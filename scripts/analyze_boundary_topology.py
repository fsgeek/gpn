"""
Test el jefe's boundary hypothesis: Do holes exist specifically in regions
between confused digit pairs?

Confused pairs from transfer experiments:
- 9 ↔ 7 (9>7 seen as 7>7)
- 6 ↔ 5 (6>5 seen as 5>5)

Test: Compute persistent homology on combined samples from digit pairs.
If holes exist in the boundary between 9 and 7, but not between unconfused
pairs like 0 and 1, that's the mechanism made visible.
"""

import argparse
from pathlib import Path
import sys

import torch
import numpy as np
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


def compute_boundary_holes(
    features: np.ndarray,
    labels: np.ndarray,
    digit1: int,
    digit2: int,
) -> dict:
    """
    Compute persistent homology on boundary region between two digits.

    Combines samples from both digits and computes topology of that region.
    Holes in this combined space represent obstacles in the boundary.
    """
    # Get samples from both digits
    mask = (labels == digit1) | (labels == digit2)
    boundary_features = features[mask]

    # Compute persistent homology
    result = ripser(boundary_features, maxdim=1, thresh=np.inf)
    diagrams = result['dgms']

    # Count holes (β₁)
    dgm_h1 = diagrams[1]

    # Infinite holes
    infinite_holes = np.sum(np.isinf(dgm_h1[:, 1]))

    # Long-lived holes (top 10% persistence)
    if len(dgm_h1) > 0:
        persistences = dgm_h1[:, 1] - dgm_h1[:, 0]
        persistences = persistences[~np.isinf(dgm_h1[:, 1])]
        if len(persistences) > 0:
            threshold = np.percentile(persistences, 90)
            long_lived = np.sum(persistences > threshold)
        else:
            long_lived = 0
    else:
        long_lived = 0

    total_holes = infinite_holes + long_lived

    # Also compute connected components (β₀)
    dgm_h0 = diagrams[0]
    infinite_components = np.sum(np.isinf(dgm_h0[:, 1]))

    return {
        'pair': f'{digit1}-{digit2}',
        'beta_0': infinite_components,
        'beta_1': total_holes,
        'samples': len(boundary_features),
    }


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

    # Define test pairs
    confused_pairs = [
        (9, 7),  # 9>7 seen as 7>7
        (6, 5),  # 6>5 seen as 5>5
    ]

    control_pairs = [
        (0, 1),  # Low confusion
        (2, 3),  # Low confusion
        (4, 8),  # Different, but no observed confusion
    ]

    print("\n" + "="*80)
    print("BOUNDARY TOPOLOGY ANALYSIS")
    print("="*80)
    print("\nTesting el jefe's hypothesis:")
    print("  Holes exist specifically in boundaries between confused digit pairs")
    print("  Confused pairs: (9,7) and (6,5)")
    print("  Control pairs: (0,1), (2,3), (4,8)")
    print()

    # Analyze pedagogical
    print("\n" + "-"*80)
    print("PEDAGOGICAL BOUNDARIES")
    print("-"*80)

    ped_results = []

    print("\nConfused pairs:")
    for d1, d2 in confused_pairs:
        result = compute_boundary_holes(ped_features, ped_labels, d1, d2)
        ped_results.append(result)
        print(f"  {result['pair']}: β₀={result['beta_0']}, β₁={result['beta_1']} holes")

    print("\nControl pairs:")
    for d1, d2 in control_pairs:
        result = compute_boundary_holes(ped_features, ped_labels, d1, d2)
        ped_results.append(result)
        print(f"  {result['pair']}: β₀={result['beta_0']}, β₁={result['beta_1']} holes")

    # Analyze adversarial
    print("\n" + "-"*80)
    print("ADVERSARIAL BOUNDARIES")
    print("-"*80)

    adv_results = []

    print("\nConfused pairs:")
    for d1, d2 in confused_pairs:
        result = compute_boundary_holes(adv_features, adv_labels, d1, d2)
        adv_results.append(result)
        print(f"  {result['pair']}: β₀={result['beta_0']}, β₁={result['beta_1']} holes")

    print("\nControl pairs:")
    for d1, d2 in control_pairs:
        result = compute_boundary_holes(adv_features, adv_labels, d1, d2)
        adv_results.append(result)
        print(f"  {result['pair']}: β₀={result['beta_0']}, β₁={result['beta_1']} holes")

    # Compare
    print("\n" + "="*80)
    print("COMPARISON: Confused vs Control Boundaries")
    print("="*80)

    # Pedagogical
    ped_confused_holes = [r['beta_1'] for r in ped_results[:2]]
    ped_control_holes = [r['beta_1'] for r in ped_results[2:]]

    print("\nPedagogical:")
    print(f"  Confused pairs (9-7, 6-5): {ped_confused_holes} (mean: {np.mean(ped_confused_holes):.1f})")
    print(f"  Control pairs: {ped_control_holes} (mean: {np.mean(ped_control_holes):.1f})")
    print(f"  Difference: {np.mean(ped_confused_holes) - np.mean(ped_control_holes):.1f} holes")

    # Adversarial
    adv_confused_holes = [r['beta_1'] for r in adv_results[:2]]
    adv_control_holes = [r['beta_1'] for r in adv_results[2:]]

    print("\nAdversarial:")
    print(f"  Confused pairs (9-7, 6-5): {adv_confused_holes} (mean: {np.mean(adv_confused_holes):.1f})")
    print(f"  Control pairs: {adv_control_holes} (mean: {np.mean(adv_control_holes):.1f})")
    print(f"  Difference: {np.mean(adv_confused_holes) - np.mean(adv_control_holes):.1f} holes")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    confused_excess = np.mean(adv_confused_holes) - np.mean(adv_control_holes)

    if confused_excess > 1.0:
        print("\n✓ SUPPORTS HYPOTHESIS:")
        print(f"  Confused boundaries have {confused_excess:.1f} more holes than control boundaries")
        print("  These holes create topological obstacles in the boundary regions")
        print("  When composition requires traversing 9→7 or 6→5 boundaries,")
        print("  the holes force detours that cross into wrong territory")
        print("  → Explains why 9>7 is seen as 7>7 (can't stay in 9-space)")
    else:
        print("\n? INCONCLUSIVE:")
        print("  Confused and control boundaries have similar hole counts")
        print("  The mechanism may be more subtle than boundary holes")
        print("  Consider: hole location, persistence, or higher-order topology")

    # Compare to pedagogical
    ped_confused_excess = np.mean(ped_confused_holes) - np.mean(ped_control_holes)

    print("\nPedagogical comparison:")
    print(f"  Pedagogical confused excess: {ped_confused_excess:.1f} holes")
    print(f"  Adversarial confused excess: {confused_excess:.1f} holes")

    if confused_excess > ped_confused_excess + 1.0:
        print("\n  ✓ Adversarial has significantly more boundary holes in confused regions")
        print("    This correlates with 81% compositional ceiling vs 100%")

    print("="*80)


if __name__ == '__main__':
    main()
