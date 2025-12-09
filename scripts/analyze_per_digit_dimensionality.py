"""
Analyze per-digit intrinsic dimensionality to test if compositional failures
correlate with high-dimensional primitives.

Based on el jefe's observation that specific digits (5, 6, 9) show confusion
in GPN failures, tests whether those digits have problematic dimensionality.
"""

import argparse
from pathlib import Path
import sys

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

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
    else:
        model = Generator(latent_dim=latent_dim, num_classes=num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['models']['generator'])

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

    judge = Judge()
    judge.to(device)
    judge.eval()

    extractor = JudgeFeatureExtractor(judge)
    features = extractor.extract(images)

    return features, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pedagogical', type=Path, required=True)
    parser.add_argument('--adversarial', type=Path, required=True)
    parser.add_argument('--samples-per-class', type=int, default=100)
    parser.add_argument('--latent-dim', type=int, default=64)

    args = parser.parse_args()
    device = torch.device('cpu')

    print("\nGenerating samples and extracting features...")
    print("="*80)

    ped_features, ped_labels = generate_and_extract_features(
        args.pedagogical, 'pedagogical', 10, args.samples_per_class,
        args.latent_dim, device
    )

    adv_features, adv_labels = generate_and_extract_features(
        args.adversarial, 'adversarial', 10, args.samples_per_class,
        args.latent_dim, device
    )

    print(f"\nPedagogical features: {ped_features.shape}")
    print(f"Adversarial features: {adv_features.shape}")

    # Compute per-digit dimensionality
    print("\n" + "="*80)
    print("PER-DIGIT INTRINSIC DIMENSIONALITY")
    print("="*80)
    print(f"{'Digit':<10} {'Pedagogical':>15} {'Adversarial':>15} {'Difference':>15}")
    print("-"*80)

    ped_dims = {}
    adv_dims = {}

    for digit in range(10):
        # Pedagogical
        ped_mask = ped_labels == digit
        ped_digit_features = ped_features[ped_mask]
        ped_dim = estimate_intrinsic_dimension_mle(ped_digit_features, k=20)
        ped_dims[digit] = ped_dim

        # Adversarial
        adv_mask = adv_labels == digit
        adv_digit_features = adv_features[adv_mask]
        adv_dim = estimate_intrinsic_dimension_mle(adv_digit_features, k=20)
        adv_dims[digit] = adv_dim

        diff = ped_dim - adv_dim
        print(f"{digit:<10} {ped_dim:>15.2f} {adv_dim:>15.2f} {diff:>15.2f}")

    # Identify problematic digits
    print("\n" + "="*80)
    print("ANALYSIS: COMPOSITIONAL FAILURE HYPOTHESIS")
    print("="*80)

    print("\nEl jefe observed GPN failures on specific digit confusions:")
    print("  - 9 → 7 (target 9>7, judge sees 7>7)")
    print("  - 6 → 5 (target 6>5, judge sees 5>5)")
    print("\nChecking if confused digits (5, 6, 7, 9) have higher dimensionality...")

    confused_digits = [5, 6, 7, 9]
    other_digits = [0, 1, 2, 3, 4, 8]

    ped_confused_dims = [ped_dims[d] for d in confused_digits]
    ped_other_dims = [ped_dims[d] for d in other_digits]

    print(f"\nPedagogical:")
    print(f"  Confused digits ({confused_digits}): {np.mean(ped_confused_dims):.2f} ± {np.std(ped_confused_dims):.2f}")
    print(f"  Other digits ({other_digits}): {np.mean(ped_other_dims):.2f} ± {np.std(ped_other_dims):.2f}")

    adv_confused_dims = [adv_dims[d] for d in confused_digits]
    adv_other_dims = [adv_dims[d] for d in other_digits]

    print(f"\nAdversarial:")
    print(f"  Confused digits ({confused_digits}): {np.mean(adv_confused_dims):.2f} ± {np.std(adv_confused_dims):.2f}")
    print(f"  Other digits ({other_digits}): {np.mean(adv_other_dims):.2f} ± {np.std(adv_other_dims):.2f}")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("If confused digits have significantly higher dimensionality:")
    print("  → Supports hypothesis that high-dimensional primitives resist composition")
    print("  → Dimensionality may predict which compositions will fail")
    print("\nIf dimensionality doesn't correlate with failures:")
    print("  → Suggests other topological properties (manifold smoothness, connectivity)")
    print("  → Need to analyze manifold structure, not just dimensionality")

    # Check manifold smoothness finding
    print("\n" + "="*80)
    print("MANIFOLD SMOOTHNESS (from el jefe's Image 2 observation)")
    print("="*80)
    print("GAN: Cliff-like discontinuities in 3→7 interpolation")
    print("GPN: Smooth gradual changes")
    print("\nThis suggests GAN latent space is disconnected/non-smooth.")
    print("Composition requires traversing latent space smoothly.")
    print("GAN's cliffs prevent smooth recombination → 81% ceiling")
    print("="*80)


if __name__ == '__main__':
    main()
