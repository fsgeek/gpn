"""
Test Fashion-MNIST relational composition with holdout pairs.

Replicates MNIST experiment (100% pedagogical vs 81% adversarial) for Fashion-MNIST.

Methodology:
- Train RelationalWeaver on 41 of 45 valid X>Y pairs (all 10 digits)
- Hold out 4 pairs: {(7,3), (8,2), (9,1), (6,4)}
- Test compositional generalization to unseen combinations
- Compare pedagogical vs adversarial primitives

This is the proper apples-to-apples test for Fashion-MNIST composition.
"""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.relational_trainer import (
    create_relational_trainer_holdout,
    create_relational_trainer_holdout_acgan,
)


def train_and_test_holdout(
    single_digit_checkpoint: Path,
    relation_judge_checkpoint: Path,
    primitive_type: str,  # 'pedagogical' or 'adversarial'
    holdout_pairs: list[tuple[int, int]],
    steps: int = 5000,
    device: str = 'cuda',
) -> dict:
    """Train RelationalWeaver and test on holdout pairs."""

    device_obj = torch.device(device)

    print("="*80, flush=True)
    print(f"TRAINING {primitive_type.upper()} RELATIONAL WEAVER", flush=True)
    print("="*80, flush=True)
    print(f"Primitives: {single_digit_checkpoint}", flush=True)
    print(f"Judge: {relation_judge_checkpoint}", flush=True)
    print(f"Holdout pairs: {holdout_pairs}", flush=True)
    print(f"Training steps: {steps}", flush=True)
    print("="*80, flush=True)

    # Create trainer
    if primitive_type == 'pedagogical':
        trainer = create_relational_trainer_holdout(
            single_digit_checkpoint=str(single_digit_checkpoint),
            judge_checkpoint=str(relation_judge_checkpoint),
            latent_dim=64,
            holdout_pairs=holdout_pairs,
            device=device_obj,
            freeze_digits=True,
        )
    else:  # adversarial
        trainer = create_relational_trainer_holdout_acgan(
            acgan_checkpoint=str(single_digit_checkpoint),
            judge_checkpoint=str(relation_judge_checkpoint),
            latent_dim=64,
            holdout_pairs=holdout_pairs,
            device=device_obj,
            freeze_digits=True,
        )

    # Training loop
    for step in range(steps):
        metrics = trainer.train_step(batch_size=64)

        if (step + 1) % 1000 == 0:
            # Evaluate on training pairs
            train_metrics = trainer.evaluate(num_samples=1000, holdout_mode=False)

            print(f"Step {step+1:5d} | "
                  f"Loss: {metrics['loss']:.4f} | "
                  f"Train X: {train_metrics['x_accuracy']:.1%} "
                  f"Y: {train_metrics['y_accuracy']:.1%} "
                  f"Rel: {train_metrics['relation_accuracy']:.1%}", flush=True)

    # Final evaluation on holdout pairs
    print("\n" + "="*80, flush=True)
    print("TESTING HOLDOUT COMPOSITION", flush=True)
    print("="*80, flush=True)

    holdout_metrics = trainer.evaluate(num_samples=1000, holdout_mode=True)

    print(f"Holdout X Accuracy: {holdout_metrics['x_accuracy']:.1%}", flush=True)
    print(f"Holdout Y Accuracy: {holdout_metrics['y_accuracy']:.1%}", flush=True)
    print(f"Holdout Relation Accuracy: {holdout_metrics['relation_accuracy']:.1%}", flush=True)
    print(f"Holdout Validity: {holdout_metrics['avg_validity']:.1%}", flush=True)
    print("="*80, flush=True)

    return {
        'holdout_x_accuracy': holdout_metrics['x_accuracy'],
        'holdout_y_accuracy': holdout_metrics['y_accuracy'],
        'holdout_relation_accuracy': holdout_metrics['relation_accuracy'],
        'holdout_validity': holdout_metrics['avg_validity'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pedagogical', type=Path, required=True,
                        help='Path to pedagogical Fashion-MNIST checkpoint')
    parser.add_argument('--adversarial', type=Path, required=True,
                        help='Path to adversarial Fashion-MNIST checkpoint')
    parser.add_argument('--judge', type=Path, required=True,
                        help='Path to Fashion-MNIST RelationJudge checkpoint')
    parser.add_argument('--steps', type=int, default=5000,
                        help='Training steps for RelationalWeaver')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Same holdout pairs as MNIST experiment
    holdout_pairs = [(7, 3), (8, 2), (9, 1), (6, 4)]

    print("\n" + "="*80, flush=True)
    print("FASHION-MNIST RELATIONAL COMPOSITION TEST", flush=True)
    print("="*80, flush=True)
    print("Replicating MNIST methodology (100% ped vs 81% adv)", flush=True)
    print(f"Training pairs: 41 of 45 (all 10 digits)", flush=True)
    print(f"Holdout pairs: {holdout_pairs}", flush=True)
    print("="*80 + "\n", flush=True)

    # Test pedagogical
    ped_results = train_and_test_holdout(
        single_digit_checkpoint=args.pedagogical,
        relation_judge_checkpoint=args.judge,
        primitive_type='pedagogical',
        holdout_pairs=holdout_pairs,
        steps=args.steps,
        device=args.device,
    )

    # Test adversarial
    adv_results = train_and_test_holdout(
        single_digit_checkpoint=args.adversarial,
        relation_judge_checkpoint=args.judge,
        primitive_type='adversarial',
        holdout_pairs=holdout_pairs,
        steps=args.steps,
        device=args.device,
    )

    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Pedagogical Relation Accuracy: {ped_results['holdout_relation_accuracy']:.1%}")
    print(f"Adversarial Relation Accuracy: {adv_results['holdout_relation_accuracy']:.1%}")
    print(f"Gap: {ped_results['holdout_relation_accuracy'] - adv_results['holdout_relation_accuracy']:+.1%}")
    print("="*80)

    # Comparison to MNIST
    mnist_ped = 1.000  # 100%
    mnist_adv = 0.811  # 81.1%

    print(f"\nMNIST baseline:")
    print(f"  Pedagogical: {mnist_ped:.1%}")
    print(f"  Adversarial: {mnist_adv:.1%}")
    print(f"  Gap: {mnist_ped - mnist_adv:+.1%}")

    print(f"\nFashion-MNIST:")
    print(f"  Pedagogical: {ped_results['holdout_relation_accuracy']:.1%}")
    print(f"  Adversarial: {adv_results['holdout_relation_accuracy']:.1%}")
    print(f"  Gap: {ped_results['holdout_relation_accuracy'] - adv_results['holdout_relation_accuracy']:+.1%}")

    # Interpretation
    ped_acc = ped_results['holdout_relation_accuracy']
    adv_acc = adv_results['holdout_relation_accuracy']

    if ped_acc >= 0.95 and adv_acc < 0.90:
        print("\n✓ Pedagogical advantage REPLICATES in Fashion-MNIST")
        print("  Topology enables composition, generalizes beyond MNIST")
    elif ped_acc > adv_acc + 0.05:
        print("\n⚠ Pedagogical shows advantage but weaker than MNIST")
    else:
        print("\n✗ Compositional advantage does NOT replicate")
        print("  Topology signature insufficient for composition in Fashion-MNIST")

    print("="*80)


if __name__ == '__main__':
    main()
