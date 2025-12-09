"""
Test generation quality of a specific checkpoint.

Quick test to check if a checkpoint can generate recognizable digits.
"""

import argparse
from pathlib import Path
import sys

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.weaver import Weaver
from src.models.judge import Judge


def test_generation_quality(
    checkpoint_path: Path,
    latent_dim: int = 64,
    samples_per_digit: int = 100,
    device: str = 'cuda',
) -> dict:
    """Test if checkpoint can generate digits recognized by Judge."""

    device = torch.device(device)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load models
    weaver = Weaver(latent_dim=latent_dim, num_classes=10)
    weaver.load_state_dict(checkpoint['models']['weaver'])
    weaver.to(device)
    weaver.eval()

    judge = Judge()
    judge.load_state_dict(checkpoint['models']['judge'])
    judge.to(device)
    judge.eval()

    # Test generation
    results = {}

    with torch.no_grad():
        for digit in range(10):
            z = torch.randn(samples_per_digit, latent_dim, device=device)
            labels = torch.full((samples_per_digit,), digit, device=device, dtype=torch.long)

            images, _ = weaver(z, labels)

            judge_logits = judge(images)
            judge_preds = judge_logits.argmax(dim=1)

            accuracy = (judge_preds == labels).float().mean().item() * 100
            results[digit] = accuracy

    mean_accuracy = np.mean(list(results.values()))

    return {
        'per_digit': results,
        'mean': mean_accuracy,
        'checkpoint_step': checkpoint.get('step', 'unknown'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--samples-per-digit', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print("="*80)
    print("CHECKPOINT GENERATION QUALITY TEST")
    print("="*80)

    results = test_generation_quality(
        args.checkpoint,
        args.latent_dim,
        args.samples_per_digit,
        args.device,
    )

    print(f"\nCheckpoint step: {results['checkpoint_step']}")
    print(f"Samples per digit: {args.samples_per_digit}")
    print("\nPer-digit accuracy:")
    print("-"*40)

    for digit, acc in results['per_digit'].items():
        print(f"  Digit {digit}: {acc:.1f}%")

    print(f"\nMean accuracy: {results['mean']:.1f}%")
    print("="*80)


if __name__ == '__main__':
    main()
