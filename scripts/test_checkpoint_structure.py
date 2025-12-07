#!/usr/bin/env python3
"""
Test checkpoint structure to understand what models we're working with.
"""

import torch
from pathlib import Path

def inspect_checkpoint(path: Path, name: str):
    """Inspect a checkpoint and print its structure."""
    print(f"\n{'=' * 80}")
    print(f"CHECKPOINT: {name}")
    print(f"Path: {path}")
    print('=' * 80)

    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)

        print(f"\nTop-level keys: {list(ckpt.keys())}")

        if 'models' in ckpt:
            print(f"\nModels: {list(ckpt['models'].keys())}")

            for model_name, state_dict in ckpt['models'].items():
                print(f"\n{model_name} layers:")
                layer_names = list(state_dict.keys())
                for i, layer in enumerate(layer_names[:10]):  # First 10 layers
                    print(f"  {layer}: {state_dict[layer].shape}")
                if len(layer_names) > 10:
                    print(f"  ... ({len(layer_names) - 10} more layers)")

        if 'config' in ckpt:
            config = ckpt['config']
            print(f"\nConfig sections: {list(config.keys())}")

            if 'model' in config:
                print(f"\nModel config: {config['model']}")

        print(f"\n✓ Checkpoint loaded successfully")
        return True

    except Exception as e:
        print(f"\n✗ Failed to load: {e}")
        return False

def main():
    print("\n" + "=" * 80)
    print("CHECKPOINT STRUCTURE INSPECTION")
    print("=" * 80)

    checkpoints = [
        ('checkpoint_final.pt', 'GPN Single-Digit (Final)'),
        ('acgan_final.pt', 'AC-GAN Single-Digit (Final)'),
        ('relational_final.pt', 'GPN Relational (Final)') if Path('checkpoints/relational_final.pt').exists() else None,
        ('acgan_twodigit_final.pt', 'AC-GAN Two-Digit (Final)'),
    ]

    for item in checkpoints:
        if item is None:
            continue
        filename, description = item
        path = Path('checkpoints') / filename
        if path.exists():
            inspect_checkpoint(path, description)
        else:
            print(f"\n{description}: NOT FOUND at {path}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nFor Checkpoint 1 analysis, we should use:")
    print("  GPN: checkpoint_final.pt (single-digit Weaver)")
    print("  GAN: acgan_final.pt (AC-GAN generator)")
    print("\nBoth trained on MNIST digits [0-9]")
    print("=" * 80)

if __name__ == "__main__":
    main()
