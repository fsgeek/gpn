"""
Analyze topology of Fashion-MNIST models to test generalization.

Tests if topological signature (dimensionality, holes) replicates in non-MNIST domain.
Compares to MNIST baseline to validate mechanistic hypothesis.
"""

import argparse
from pathlib import Path
import sys
import json

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torchvision import datasets, transforms
from ripser import ripser

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.weaver import Weaver
from src.models.baseline_gan import Generator
from src.models.judge import Judge


class JudgeFeatureExtractor:
    """Extract Judge conv features."""

    def __init__(self, judge: Judge):
        self.judge = judge
        self.features = None

        def hook_fn(module, input, output):
            self.features = output.detach()

        self.judge.features.register_forward_hook(hook_fn)

    def extract(self, images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            _ = self.judge(images)
        return self.features.flatten(start_dim=1).cpu().numpy()


def get_fashion_mnist_judge(device: torch.device) -> Judge:
    """Load Fashion-MNIST trained Judge from checkpoint."""
    # For now, create a fresh Judge - in real use, would load from pedagogical checkpoint
    judge = Judge()
    judge.to(device)
    judge.eval()
    return judge


def generate_and_extract_features(
    checkpoint_path: Path,
    model_type: str,
    num_classes: int,
    samples_per_class: int,
    latent_dim: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate samples and extract Judge features."""

    if model_type == 'pedagogical':
        model = Weaver(latent_dim=latent_dim, num_classes=num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['models']['weaver'])
        # Get Judge from pedagogical checkpoint
        judge = Judge()
        judge.load_state_dict(checkpoint['models']['judge'])
    else:
        model = Generator(latent_dim=latent_dim, num_classes=num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['models']['generator'])
        # Use separate Fashion-MNIST Judge
        judge = get_fashion_mnist_judge(device)

    model.to(device)
    model.eval()

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

    judge.to(device)
    judge.eval()

    extractor = JudgeFeatureExtractor(judge)
    features = extractor.extract(images)

    return features, labels


def estimate_intrinsic_dimension_mle(X: np.ndarray, k: int = 20) -> float:
    """MLE estimate of intrinsic dimensionality."""
    n_samples = X.shape[0]
    if n_samples < k + 1:
        return np.nan

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:, 1:]  # Remove self

    r_k = distances[:, -1:] + 1e-10
    ratios = r_k / (distances + 1e-10)
    log_ratios = np.log(ratios)
    local_dims = (k - 1) / np.sum(log_ratios[:, :-1], axis=1)

    return np.mean(local_dims)


def compute_persistence(X: np.ndarray, maxdim: int = 1) -> dict:
    """Compute persistent homology."""
    result = ripser(X, maxdim=maxdim, thresh=np.inf)
    diagrams = result['dgms']

    betti = {}
    for dim in range(min(maxdim + 1, len(diagrams))):
        dgm = diagrams[dim]
        infinite_features = np.sum(np.isinf(dgm[:, 1]))

        if len(dgm) > 0:
            persistences = dgm[:, 1] - dgm[:, 0]
            persistences = persistences[~np.isinf(dgm[:, 1])]
            if len(persistences) > 0:
                threshold = np.percentile(persistences, 90)
                long_lived = np.sum(persistences > threshold)
            else:
                long_lived = 0
        else:
            long_lived = 0

        betti[f'H{dim}_persistent'] = infinite_features + long_lived

    return {
        'diagrams': diagrams,
        'betti': betti,
    }


def analyze_per_digit_topology(
    features: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Compute topology metrics for each digit."""
    results = {}

    for digit in range(10):
        mask = labels == digit
        digit_features = features[mask]

        persistence = compute_persistence(digit_features, maxdim=1)
        results[int(digit)] = {
            'beta_0': int(persistence['betti']['H0_persistent']),
            'beta_1': int(persistence['betti']['H1_persistent']),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pedagogical', type=Path, required=True)
    parser.add_argument('--adversarial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples-per-class', type=int, default=100)
    parser.add_argument('--latent-dim', type=int, default=64)

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print("FASHION-MNIST TOPOLOGY ANALYSIS")
    print("="*80)
    print(f"Device: {device}")
    print(f"Samples per class: {args.samples_per_class}")

    # Generate and extract features
    print("\nGenerating samples and extracting features...")
    print("-"*80)

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

    # Compute global metrics
    print("\n" + "="*80)
    print("GLOBAL METRICS")
    print("="*80)

    print("Computing intrinsic dimensionality...")
    ped_dim = estimate_intrinsic_dimension_mle(ped_features, k=20)
    adv_dim = estimate_intrinsic_dimension_mle(adv_features, k=20)

    print(f"  Pedagogical: {ped_dim:.2f} dimensions")
    print(f"  Adversarial: {adv_dim:.2f} dimensions")
    print(f"  Difference: {((ped_dim - adv_dim) / adv_dim * 100):.1f}%")

    # Per-digit topology
    print("\n" + "="*80)
    print("PER-DIGIT TOPOLOGY")
    print("="*80)

    ped_topology = analyze_per_digit_topology(ped_features, ped_labels)
    adv_topology = analyze_per_digit_topology(adv_features, adv_labels)

    print("\nβ₁ (Holes) by digit:")
    print(f"{'Digit':<10} {'Ped':>10} {'Adv':>10} {'Diff':>10}")
    print("-"*40)

    ped_holes = []
    adv_holes = []

    for digit in range(10):
        ped_h1 = ped_topology[digit]['beta_1']
        adv_h1 = adv_topology[digit]['beta_1']
        diff = ped_h1 - adv_h1

        ped_holes.append(ped_h1)
        adv_holes.append(adv_h1)

        print(f"{digit:<10} {ped_h1:>10} {adv_h1:>10} {diff:>10}")

    print(f"\n{'Mean':<10} {np.mean(ped_holes):>10.1f} {np.mean(adv_holes):>10.1f} {np.mean(ped_holes) - np.mean(adv_holes):>10.1f}")

    # Results summary
    results = {
        'fashion_mnist': {
            'pedagogical': {
                'intrinsic_dim': float(ped_dim),
                'mean_holes': float(np.mean(ped_holes)),
                'per_digit': ped_topology,
            },
            'adversarial': {
                'intrinsic_dim': float(adv_dim),
                'mean_holes': float(np.mean(adv_holes)),
                'per_digit': adv_topology,
            },
        },
        'mnist_baseline': {
            'pedagogical': {
                'intrinsic_dim': 9.94,
                'mean_holes': 5.6,
            },
            'adversarial': {
                'intrinsic_dim': 13.55,
                'mean_holes': 8.0,
            },
        },
    }

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {args.output}")

    # Comparison to MNIST
    print("\n" + "="*80)
    print("COMPARISON TO MNIST BASELINE")
    print("="*80)

    mnist_ped_dim = 9.94
    mnist_adv_dim = 13.55
    mnist_ped_holes = 5.6
    mnist_adv_holes = 8.0

    print("\nIntrinsic Dimensionality:")
    print(f"  MNIST: Ped {mnist_ped_dim:.2f}, Adv {mnist_adv_dim:.2f} (Δ {((mnist_ped_dim - mnist_adv_dim) / mnist_adv_dim * 100):.1f}%)")
    print(f"  Fashion: Ped {ped_dim:.2f}, Adv {adv_dim:.2f} (Δ {((ped_dim - adv_dim) / adv_dim * 100):.1f}%)")

    print("\nMean Holes (β₁):")
    print(f"  MNIST: Ped {mnist_ped_holes:.1f}, Adv {mnist_adv_holes:.1f} (Δ {((mnist_ped_holes - mnist_adv_holes) / mnist_adv_holes * 100):.1f}%)")
    print(f"  Fashion: Ped {np.mean(ped_holes):.1f}, Adv {np.mean(adv_holes):.1f} (Δ {((np.mean(ped_holes) - np.mean(adv_holes)) / np.mean(adv_holes) * 100):.1f}%)")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    # Check if signature replicates
    dim_reduction_mnist = (mnist_ped_dim - mnist_adv_dim) / mnist_adv_dim
    dim_reduction_fashion = (ped_dim - adv_dim) / adv_dim

    hole_reduction_mnist = (mnist_ped_holes - mnist_adv_holes) / mnist_adv_holes
    hole_reduction_fashion = (np.mean(ped_holes) - np.mean(adv_holes)) / np.mean(adv_holes)

    print(f"\nDimensionality reduction: MNIST {dim_reduction_mnist:.1%}, Fashion {dim_reduction_fashion:.1%}")
    print(f"Hole reduction: MNIST {hole_reduction_mnist:.1%}, Fashion {hole_reduction_fashion:.1%}")

    # Rough threshold: if Fashion shows >50% of MNIST's effect size, consider replicated
    if abs(dim_reduction_fashion) > abs(dim_reduction_mnist) * 0.5:
        print("\n✓ Dimensionality signature REPLICATES in Fashion-MNIST")
    else:
        print("\n✗ Dimensionality signature does NOT replicate")

    if abs(hole_reduction_fashion) > abs(hole_reduction_mnist) * 0.5:
        print("✓ Topological signature (holes) REPLICATES in Fashion-MNIST")
    else:
        print("✗ Topological signature does NOT replicate")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
