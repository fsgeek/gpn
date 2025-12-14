#!/usr/bin/env python3
"""
Verification script: Confirm that primitives are actually frozen.

This script creates trainers for both AC-GAN and architecture-only conditions
and verifies that:
1. Primitive parameters have requires_grad=False
2. Latent splitter parameters have requires_grad=True
3. After a training step, primitive parameters haven't changed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import copy

from training.relational_trainer import (
    create_relational_trainer_holdout,
    create_relational_trainer_holdout_acgan,
)


def count_params(module, requires_grad=None):
    """Count parameters, optionally filtered by requires_grad."""
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            total += p.numel()
    return total


def verify_freezing(trainer, name):
    """Verify that primitives are frozen and splitter is trainable."""
    print(f"\n{'='*60}")
    print(f"VERIFYING: {name}")
    print(f"{'='*60}")

    weaver = trainer.weaver

    # Check single_digit_weaver parameters
    primitive_params = list(weaver.single_digit_weaver.parameters())
    primitive_frozen = all(not p.requires_grad for p in primitive_params)
    primitive_count = sum(p.numel() for p in primitive_params)

    print(f"\nPrimitive (single_digit_weaver):")
    print(f"  Total parameters: {primitive_count:,}")
    print(f"  All frozen (requires_grad=False): {primitive_frozen}")

    if not primitive_frozen:
        trainable = sum(p.numel() for p in primitive_params if p.requires_grad)
        print(f"  WARNING: {trainable:,} parameters are trainable!")

    # Check latent_splitter parameters
    splitter_params = list(weaver.latent_splitter.parameters())
    splitter_trainable = all(p.requires_grad for p in splitter_params)
    splitter_count = sum(p.numel() for p in splitter_params)

    print(f"\nLatent splitter:")
    print(f"  Total parameters: {splitter_count:,}")
    print(f"  All trainable (requires_grad=True): {splitter_trainable}")

    # Snapshot primitive weights before training step
    primitive_snapshot = {
        name: p.clone() for name, p in weaver.single_digit_weaver.named_parameters()
    }
    splitter_snapshot = {
        name: p.clone() for name, p in weaver.latent_splitter.named_parameters()
    }

    # Do one training step
    print(f"\nRunning one training step...")
    metrics = trainer.train_step(batch_size=64)
    print(f"  Loss: {metrics['loss']:.4f}")

    # Check if primitive weights changed
    primitive_changed = False
    for name, p in weaver.single_digit_weaver.named_parameters():
        if not torch.equal(p, primitive_snapshot[name]):
            primitive_changed = True
            diff = (p - primitive_snapshot[name]).abs().max().item()
            print(f"  WARNING: Primitive param '{name}' changed by max {diff:.6f}!")

    if not primitive_changed:
        print(f"  Primitive weights unchanged (GOOD - frozen)")

    # Check if splitter weights changed
    splitter_changed = False
    for name, p in weaver.latent_splitter.named_parameters():
        if not torch.equal(p, splitter_snapshot[name]):
            splitter_changed = True

    if splitter_changed:
        print(f"  Splitter weights changed (GOOD - training)")
    else:
        print(f"  WARNING: Splitter weights unchanged!")

    # Summary
    print(f"\nSUMMARY:")
    passed = primitive_frozen and splitter_trainable and not primitive_changed and splitter_changed
    if passed:
        print(f"  PASS: Freezing working correctly")
    else:
        print(f"  FAIL: Freezing NOT working correctly")
        if not primitive_frozen:
            print(f"    - Primitives not marked as frozen")
        if not splitter_trainable:
            print(f"    - Splitter not marked as trainable")
        if primitive_changed:
            print(f"    - Primitive weights changed during training")
        if not splitter_changed:
            print(f"    - Splitter weights didn't change during training")

    return passed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}

    # Test 1: AC-GAN with freeze_digits=True
    if Path("checkpoints/acgan_final.pt").exists():
        trainer_acgan = create_relational_trainer_holdout_acgan(
            acgan_checkpoint="checkpoints/acgan_final.pt",
            judge_checkpoint="checkpoints/relation_judge.pt",
            device=device,
            freeze_digits=True,
        )
        results['acgan_frozen'] = verify_freezing(trainer_acgan, "AC-GAN (freeze_digits=True)")
    else:
        print("Skipping AC-GAN: checkpoint not found")

    # Test 2: Architecture-only (random primitives)
    if Path("checkpoints/architecture_only/architecture_only_seed_0.pt").exists():
        trainer_random = create_relational_trainer_holdout(
            single_digit_checkpoint="checkpoints/architecture_only/architecture_only_seed_0.pt",
            judge_checkpoint="checkpoints/relation_judge.pt",
            device=device,
            freeze_digits=True,
        )
        results['random_frozen'] = verify_freezing(trainer_random, "Architecture-only/Random (freeze_digits=True)")
    else:
        print("Skipping architecture-only: checkpoint not found")

    # Test 3: Pedagogical
    if Path("checkpoints/checkpoint_final.pt").exists():
        trainer_ped = create_relational_trainer_holdout(
            single_digit_checkpoint="checkpoints/checkpoint_final.pt",
            judge_checkpoint="checkpoints/relation_judge.pt",
            device=device,
            freeze_digits=True,
        )
        results['pedagogical_frozen'] = verify_freezing(trainer_ped, "Pedagogical (freeze_digits=True)")
    else:
        print("Skipping pedagogical: checkpoint not found")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    if all_passed:
        print(f"\nAll freezing checks PASSED. The experiments are valid.")
    else:
        print(f"\nSome checks FAILED. Review the output above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
