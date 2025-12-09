"""
Compute standard metrics on Judge features: intrinsic dimensionality,
linear separability, cluster quality.

Tests whether simple measures detect pedagogical vs adversarial differences
when analyzing generated samples (vs Checkpoint 1 internal representations).
"""

import argparse
from pathlib import Path
import sys

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.weaver import Weaver
from src.models.baseline_gan import Generator
from src.models.judge import Judge


class JudgeFeatureExtractor:
    """Extract Judge conv features from generated images."""

    def __init__(self, judge: Judge):
        self.judge = judge
        self.features = None

        def hook_fn(module, input, output):
            self.features = output.detach()

        self.judge.features.register_forward_hook(hook_fn)

    def extract(self, images: torch.Tensor) -> np.ndarray:
        """Extract and flatten features."""
        with torch.no_grad():
            _ = self.judge(images)
        features = self.features.flatten(start_dim=1)
        return features.cpu().numpy()


def generate_and_extract_features(
    checkpoint_path: Path,
    model_type: str,
    num_classes: int,
    samples_per_class: int,
    latent_dim: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate samples and extract Judge features."""

    # Load generator
    if model_type == 'pedagogical':
        model = Weaver(latent_dim=latent_dim, num_classes=num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['models']['weaver'])
    else:  # adversarial
        model = Generator(latent_dim=latent_dim, num_classes=num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['models']['generator'])

    model.to(device)
    model.eval()

    # Generate samples
    all_images = []
    all_labels = []

    with torch.no_grad():
        for class_idx in range(num_classes):
            z = torch.randn(samples_per_class, latent_dim, device=device)
            labels = torch.full((samples_per_class,), class_idx, device=device, dtype=torch.long)

            if model_type == 'pedagogical':
                images, _ = model(z, labels)
            else:
                images = model(z, labels)

            all_images.append(images)
            all_labels.append(labels.cpu().numpy())

    images = torch.cat(all_images, dim=0)
    labels = np.concatenate(all_labels, axis=0)

    # Extract Judge features
    judge = Judge()
    judge.to(device)
    judge.eval()

    extractor = JudgeFeatureExtractor(judge)
    features = extractor.extract(images)

    return features, labels


def estimate_intrinsic_dimension_mle(X: np.ndarray, k: int = 20) -> float:
    """
    Estimate intrinsic dimensionality using MLE method.

    Based on Levina & Bickel (2004) "Maximum Likelihood Estimation of
    Intrinsic Dimension"
    """
    n_samples = X.shape[0]

    # Compute k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nbrs.kneighbors(X)

    # Remove self (distance 0)
    distances = distances[:, 1:]

    # MLE estimate for each point
    # d̂ = (k-1) / sum(log(r_k / r_i))
    r_k = distances[:, -1:] + 1e-10  # Avoid division by zero
    ratios = r_k / (distances + 1e-10)
    log_ratios = np.log(ratios)

    # Sum over neighbors (excluding r_k itself)
    local_dims = (k - 1) / np.sum(log_ratios[:, :-1], axis=1)

    # Average over all points
    intrinsic_dim = np.mean(local_dims)

    return intrinsic_dim


def compute_linear_separability(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Measure linear separability using LDA.

    Returns:
        - lda_accuracy: Classification accuracy using LDA
        - explained_variance_ratio: Ratio of between-class to total variance
    """
    # Fit LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    # Compute accuracy
    accuracy = lda.score(X, y)

    # Explained variance ratio (how much variance is between-class)
    # This measures how linearly separable the classes are
    explained_var_ratio = np.sum(lda.explained_variance_ratio_)

    return {
        'lda_accuracy': accuracy,
        'explained_variance_ratio': explained_var_ratio,
    }


def compute_cluster_quality(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Measure cluster quality using silhouette score and Davies-Bouldin index.

    Returns:
        - silhouette: Higher is better (range [-1, 1])
        - davies_bouldin: Lower is better
    """
    # Silhouette score: measures how similar samples are to their own cluster
    # vs other clusters
    silhouette = silhouette_score(X, y)

    # Davies-Bouldin index: ratio of within-cluster to between-cluster distances
    davies_bouldin = davies_bouldin_score(X, y)

    return {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
    }


def compute_all_metrics(features: np.ndarray, labels: np.ndarray, name: str) -> dict:
    """Compute all metrics for a feature set."""
    print(f"\n{'='*80}")
    print(f"Computing metrics for: {name}")
    print(f"{'='*80}")

    results = {'name': name}

    # Intrinsic dimensionality
    print("  Computing intrinsic dimensionality (MLE)...")
    results['intrinsic_dim'] = estimate_intrinsic_dimension_mle(features, k=20)
    print(f"    Intrinsic dimension: {results['intrinsic_dim']:.2f}")

    # Linear separability
    print("  Computing linear separability (LDA)...")
    sep_metrics = compute_linear_separability(features, labels)
    results.update(sep_metrics)
    print(f"    LDA accuracy: {sep_metrics['lda_accuracy']:.4f}")
    print(f"    Explained variance ratio: {sep_metrics['explained_variance_ratio']:.4f}")

    # Cluster quality
    print("  Computing cluster quality...")
    cluster_metrics = compute_cluster_quality(features, labels)
    results.update(cluster_metrics)
    print(f"    Silhouette score: {cluster_metrics['silhouette']:.4f}")
    print(f"    Davies-Bouldin index: {cluster_metrics['davies_bouldin']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Compute feature metrics')
    parser.add_argument('--pedagogical', type=Path, required=True)
    parser.add_argument('--adversarial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples-per-class', type=int, default=100)
    parser.add_argument('--latent-dim', type=int, default=64)

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')

    # Generate and extract features
    print("\nGenerating samples and extracting Judge features...")
    print("-" * 80)

    print("Pedagogical model...")
    ped_features, ped_labels = generate_and_extract_features(
        args.pedagogical, 'pedagogical', 10, args.samples_per_class,
        args.latent_dim, device
    )
    print(f"  Features shape: {ped_features.shape}")

    print("\nAdversarial model...")
    adv_features, adv_labels = generate_and_extract_features(
        args.adversarial, 'adversarial', 10, args.samples_per_class,
        args.latent_dim, device
    )
    print(f"  Features shape: {adv_features.shape}")

    # Compute metrics
    ped_results = compute_all_metrics(ped_features, ped_labels, "Pedagogical")
    adv_results = compute_all_metrics(adv_features, adv_labels, "Adversarial")

    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'Pedagogical':>15} {'Adversarial':>15} {'Difference':>15}")
    print("-" * 80)

    metrics = ['intrinsic_dim', 'lda_accuracy', 'explained_variance_ratio',
               'silhouette', 'davies_bouldin']

    for metric in metrics:
        ped_val = ped_results[metric]
        adv_val = adv_results[metric]
        diff = ped_val - adv_val
        print(f"{metric:<30} {ped_val:>15.4f} {adv_val:>15.4f} {diff:>15.4f}")

    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    print("If pedagogical differs significantly from adversarial:")
    print("  - Intrinsic dim: Lower = simpler manifold structure")
    print("  - LDA accuracy: Higher = more linearly separable")
    print("  - Explained variance: Higher = classes better separated")
    print("  - Silhouette: Higher = better cluster quality")
    print("  - Davies-Bouldin: Lower = better cluster separation")
    print()
    print("If metrics are similar despite 100% vs 81% compositional capacity,")
    print("that suggests the difference lies in topological properties not")
    print("captured by these standard geometric measures.")
    print(f"{'='*80}")

    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump({
            'pedagogical': ped_results,
            'adversarial': adv_results,
        }, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
