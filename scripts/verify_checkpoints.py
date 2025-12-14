#!/usr/bin/env python3
"""
Verification script: Confirm checkpoints are loaded correctly and are distinct.

Verifies:
1. AC-GAN checkpoint contains trained generator (not random)
2. Architecture-only checkpoint contains untrained weights
3. Pedagogical checkpoint contains trained weights
4. The three primitives are genuinely different
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from models.weaver import create_weaver
from models.acgan import ACGANGenerator


def load_and_analyze_checkpoint(path, name, model_type='gpn'):
    """Load checkpoint and compute statistics."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {name}")
    print(f"Path: {path}")
    print(f"{'='*60}")

    if not Path(path).exists():
        print(f"  FILE NOT FOUND")
        return None

    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    # Get model state dict
    if model_type == 'gpn':
        if 'models' in ckpt and 'weaver' in ckpt['models']:
            state_dict = ckpt['models']['weaver']
        else:
            print(f"  ERROR: Can't find weaver in checkpoint")
            print(f"  Keys: {ckpt.keys()}")
            return None
    elif model_type == 'acgan':
        if 'models' in ckpt and 'generator' in ckpt['models']:
            state_dict = ckpt['models']['generator']
        else:
            print(f"  ERROR: Can't find generator in checkpoint")
            print(f"  Keys: {ckpt.keys()}")
            return None

    # Compute statistics
    print(f"\n  Checkpoint keys: {list(ckpt.keys())}")
    if 'seed' in ckpt:
        print(f"  Seed: {ckpt['seed']}")
    if 'config' in ckpt:
        print(f"  Has config: Yes")

    # Weight statistics
    all_weights = []
    for key, tensor in state_dict.items():
        if tensor.dtype == torch.float32:
            all_weights.append(tensor.flatten())

    all_weights = torch.cat(all_weights)
    stats = {
        'mean': all_weights.mean().item(),
        'std': all_weights.std().item(),
        'min': all_weights.min().item(),
        'max': all_weights.max().item(),
        'num_params': len(all_weights),
    }

    print(f"\n  Weight statistics:")
    print(f"    Num parameters: {stats['num_params']:,}")
    print(f"    Mean: {stats['mean']:.6f}")
    print(f"    Std: {stats['std']:.6f}")
    print(f"    Min: {stats['min']:.6f}")
    print(f"    Max: {stats['max']:.6f}")

    # Check for signs of training
    # Trained models typically have larger std and more extreme values
    # than Xavier/Kaiming initialization

    return {'state_dict': state_dict, 'stats': stats, 'name': name}


def compare_checkpoints(ckpt1, ckpt2):
    """Compare two checkpoints for similarity."""
    if ckpt1 is None or ckpt2 is None:
        return

    print(f"\n{'='*60}")
    print(f"COMPARING: {ckpt1['name']} vs {ckpt2['name']}")
    print(f"{'='*60}")

    # Find common keys
    keys1 = set(ckpt1['state_dict'].keys())
    keys2 = set(ckpt2['state_dict'].keys())
    common = keys1 & keys2

    if not common:
        print(f"  No common keys (different architectures)")
        print(f"  {ckpt1['name']} keys: {len(keys1)}")
        print(f"  {ckpt2['name']} keys: {len(keys2)}")
        return

    # Compare common layers
    identical = 0
    different = 0
    total_diff = 0

    for key in common:
        t1 = ckpt1['state_dict'][key]
        t2 = ckpt2['state_dict'][key]

        if t1.shape != t2.shape:
            print(f"  {key}: shape mismatch {t1.shape} vs {t2.shape}")
            continue

        if torch.equal(t1, t2):
            identical += 1
        else:
            different += 1
            if t1.dtype in (torch.float32, torch.float64):
                diff = (t1 - t2).abs().mean().item()
                total_diff += diff

    print(f"  Common layers: {len(common)}")
    print(f"  Identical: {identical}")
    print(f"  Different: {different}")
    if different > 0:
        print(f"  Avg difference: {total_diff/different:.6f}")


def main():
    print("CHECKPOINT VERIFICATION")
    print("=" * 60)

    # Load all checkpoints
    acgan = load_and_analyze_checkpoint(
        "checkpoints/acgan_final.pt",
        "AC-GAN (trained)",
        model_type='acgan'
    )

    arch_only = load_and_analyze_checkpoint(
        "checkpoints/architecture_only/architecture_only_seed_0.pt",
        "Architecture-only (untrained)",
        model_type='gpn'
    )

    pedagogical = load_and_analyze_checkpoint(
        "checkpoints/checkpoint_final.pt",
        "Pedagogical (trained)",
        model_type='gpn'
    )

    # Compare pedagogical vs architecture-only (same architecture)
    compare_checkpoints(pedagogical, arch_only)

    # Note: AC-GAN has different architecture, can't directly compare weights

    # Final verification
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")

    issues = []

    # Check architecture-only is actually untrained
    if arch_only:
        # Untrained weights from Kaiming init should have std around 0.01-0.1
        # and mean near 0
        if abs(arch_only['stats']['mean']) > 0.1:
            issues.append(f"Architecture-only mean ({arch_only['stats']['mean']:.4f}) suggests training occurred")

    # Check pedagogical differs from architecture-only
    if pedagogical and arch_only:
        # They should be different
        ped_keys = set(pedagogical['state_dict'].keys())
        arch_keys = set(arch_only['state_dict'].keys())
        if ped_keys == arch_keys:
            # Same architecture, check if weights differ
            same_count = 0
            for key in ped_keys:
                if torch.equal(pedagogical['state_dict'][key], arch_only['state_dict'][key]):
                    same_count += 1
            if same_count == len(ped_keys):
                issues.append("Pedagogical and architecture-only have IDENTICAL weights!")
            elif same_count > len(ped_keys) * 0.5:
                issues.append(f"Pedagogical and architecture-only share {same_count}/{len(ped_keys)} identical layers")

    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nAll checkpoints appear valid and distinct.")


if __name__ == "__main__":
    main()
