#!/usr/bin/env python3
"""
Multi-seed validation for compositional transfer claims.

Runs Phase 1.6 relational holdout experiments with multiple seeds
to establish variance estimates for the 100% vs 81.1% headline result.

Uses the proper RelationalTrainerHoldout infrastructure to match
the original experiment setup.

Usage:
    python scripts/run_multiseed_validation.py

Output:
    results/multiseed_validation/
        pedagogical_seed_0.json
        pedagogical_seed_1.json
        ...
        adversarial_seed_0.json
        ...
        summary.json  (aggregated statistics)

Estimated time: ~30-45 minutes on RTX 4090
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from training.relational_trainer import (
    create_relational_trainer_holdout,
    create_relational_trainer_holdout_acgan,
)

# Configuration
N_SEEDS = 5
N_STEPS = 5000
OUTPUT_DIR = Path("results/multiseed_validation")
CHECKPOINT_DIR = Path("checkpoints/multiseed_validation")

# Hold-out pairs (same as original experiment)
HOLDOUT_PAIRS = [(7, 3), (8, 2), (9, 1), (6, 4)]


def run_relational_holdout_pedagogical(seed: int, output_name: str) -> dict:
    """Run a single pedagogical relational holdout experiment."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Running: {output_name} (seed={seed}) on {device}")
    print(f"{'='*60}")

    # Create trainer using the proper infrastructure
    trainer = create_relational_trainer_holdout(
        single_digit_checkpoint="checkpoints/checkpoint_final.pt",
        judge_checkpoint="checkpoints/relation_judge.pt",
        latent_dim=64,
        holdout_pairs=HOLDOUT_PAIRS,
        device=device,
        freeze_digits=True,  # Match original experiment
    )

    # Training loop
    for step in range(N_STEPS):
        metrics = trainer.train_step(batch_size=64)

        if step % 500 == 0:
            print(f"  Step {step}: loss={metrics['loss']:.4f}, "
                  f"x_acc={metrics['x_accuracy']:.1%}, "
                  f"y_acc={metrics['y_accuracy']:.1%}, "
                  f"validity={metrics['avg_validity']:.3f}")

    # Evaluate on holdout pairs
    print("\n  EVALUATING ON HOLDOUT PAIRS...")
    holdout_metrics = trainer.evaluate(num_samples=1000, holdout_mode=True)

    print(f"\n  HOLDOUT RESULTS:")
    print(f"    X accuracy: {holdout_metrics['x_accuracy']:.1%}")
    print(f"    Y accuracy: {holdout_metrics['y_accuracy']:.1%}")
    print(f"    Validity: {holdout_metrics['avg_validity']:.3f}")
    print(f"    Relation accuracy: {holdout_metrics['relation_accuracy']:.1%}")

    # Also evaluate on training pairs for comparison
    training_metrics = trainer.evaluate(num_samples=1000, holdout_mode=False)

    results = {
        'seed': seed,
        'primitives': 'pedagogical',
        'checkpoint': 'checkpoints/checkpoint_final.pt',
        'holdout_pairs': HOLDOUT_PAIRS,
        'holdout': {
            'x_accuracy': holdout_metrics['x_accuracy'],
            'y_accuracy': holdout_metrics['y_accuracy'],
            'validity': holdout_metrics['avg_validity'],
            'relation_accuracy': holdout_metrics['relation_accuracy'],
        },
        'training': {
            'x_accuracy': training_metrics['x_accuracy'],
            'y_accuracy': training_metrics['y_accuracy'],
            'validity': training_metrics['avg_validity'],
            'relation_accuracy': training_metrics['relation_accuracy'],
        },
        'timestamp': datetime.now().isoformat()
    }

    return results


def run_relational_holdout_adversarial(seed: int, output_name: str) -> dict:
    """Run a single adversarial (AC-GAN) relational holdout experiment."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Running: {output_name} (seed={seed}) on {device}")
    print(f"{'='*60}")

    # Create trainer using AC-GAN primitives
    trainer = create_relational_trainer_holdout_acgan(
        acgan_checkpoint="checkpoints/acgan_final.pt",
        judge_checkpoint="checkpoints/relation_judge.pt",
        holdout_pairs=HOLDOUT_PAIRS,
        device=device,
        freeze_digits=True,  # Match original experiment
    )

    # Training loop
    for step in range(N_STEPS):
        metrics = trainer.train_step(batch_size=64)

        if step % 500 == 0:
            print(f"  Step {step}: loss={metrics['loss']:.4f}, "
                  f"x_acc={metrics['x_accuracy']:.1%}, "
                  f"y_acc={metrics['y_accuracy']:.1%}, "
                  f"validity={metrics['avg_validity']:.3f}")

    # Evaluate on holdout pairs
    print("\n  EVALUATING ON HOLDOUT PAIRS...")
    holdout_metrics = trainer.evaluate(num_samples=1000, holdout_mode=True)

    print(f"\n  HOLDOUT RESULTS:")
    print(f"    X accuracy: {holdout_metrics['x_accuracy']:.1%}")
    print(f"    Y accuracy: {holdout_metrics['y_accuracy']:.1%}")
    print(f"    Validity: {holdout_metrics['avg_validity']:.3f}")
    print(f"    Relation accuracy: {holdout_metrics['relation_accuracy']:.1%}")

    # Also evaluate on training pairs
    training_metrics = trainer.evaluate(num_samples=1000, holdout_mode=False)

    results = {
        'seed': seed,
        'primitives': 'adversarial',
        'checkpoint': 'checkpoints/acgan_final.pt',
        'holdout_pairs': HOLDOUT_PAIRS,
        'holdout': {
            'x_accuracy': holdout_metrics['x_accuracy'],
            'y_accuracy': holdout_metrics['y_accuracy'],
            'validity': holdout_metrics['avg_validity'],
            'relation_accuracy': holdout_metrics['relation_accuracy'],
        },
        'training': {
            'x_accuracy': training_metrics['x_accuracy'],
            'y_accuracy': training_metrics['y_accuracy'],
            'validity': training_metrics['avg_validity'],
            'relation_accuracy': training_metrics['relation_accuracy'],
        },
        'timestamp': datetime.now().isoformat()
    }

    return results


def compute_summary(results: list[dict], name: str) -> dict:
    """Compute summary statistics for a set of results."""
    holdout_accs = [r['holdout']['relation_accuracy'] for r in results]
    training_accs = [r['training']['relation_accuracy'] for r in results]

    return {
        'name': name,
        'n_seeds': len(results),
        'holdout': {
            'mean_accuracy': float(np.mean(holdout_accs)),
            'std_accuracy': float(np.std(holdout_accs)),
            'min_accuracy': float(np.min(holdout_accs)),
            'max_accuracy': float(np.max(holdout_accs)),
            'accuracies': [float(a) for a in holdout_accs],
        },
        'training': {
            'mean_accuracy': float(np.mean(training_accs)),
            'std_accuracy': float(np.std(training_accs)),
            'accuracies': [float(a) for a in training_accs],
        }
    }


def main():
    from scipy import stats

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MULTI-SEED VALIDATION FOR COMPOSITIONAL TRANSFER")
    print("=" * 60)
    print(f"Seeds: {N_SEEDS}")
    print(f"Steps per run: {N_STEPS}")
    print(f"Holdout pairs: {HOLDOUT_PAIRS}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Start time: {datetime.now().isoformat()}")

    # Check for required checkpoints
    ped_checkpoint = Path("checkpoints/checkpoint_final.pt")
    adv_checkpoint = Path("checkpoints/acgan_final.pt")
    judge_checkpoint = Path("checkpoints/relation_judge.pt")

    missing = []
    if not ped_checkpoint.exists():
        missing.append(str(ped_checkpoint))
    if not adv_checkpoint.exists():
        missing.append(str(adv_checkpoint))
    if not judge_checkpoint.exists():
        missing.append(str(judge_checkpoint))

    if missing:
        print(f"ERROR: Missing checkpoints: {missing}")
        return

    # Run pedagogical experiments
    print("\n" + "=" * 60)
    print("PEDAGOGICAL PRIMITIVES")
    print("=" * 60)

    ped_results = []
    for seed in range(N_SEEDS):
        result = run_relational_holdout_pedagogical(seed, f"pedagogical_seed_{seed}")
        ped_results.append(result)

        # Save individual result
        with open(OUTPUT_DIR / f"pedagogical_seed_{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)

    # Run adversarial experiments
    print("\n" + "=" * 60)
    print("ADVERSARIAL PRIMITIVES")
    print("=" * 60)

    adv_results = []
    for seed in range(N_SEEDS):
        result = run_relational_holdout_adversarial(seed, f"adversarial_seed_{seed}")
        adv_results.append(result)

        # Save individual result
        with open(OUTPUT_DIR / f"adversarial_seed_{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)

    # Compute summaries
    ped_summary = compute_summary(ped_results, "pedagogical")
    adv_summary = compute_summary(adv_results, "adversarial")

    # Statistical test
    ped_accs = ped_summary['holdout']['accuracies']
    adv_accs = adv_summary['holdout']['accuracies']

    # Mann-Whitney U test (non-parametric)
    u_stat, p_value = stats.mannwhitneyu(ped_accs, adv_accs, alternative='greater')

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(ped_accs)**2 + np.std(adv_accs)**2) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(ped_accs) - np.mean(adv_accs)) / pooled_std
    else:
        cohens_d = float('inf') if np.mean(ped_accs) > np.mean(adv_accs) else 0

    summary = {
        'pedagogical': ped_summary,
        'adversarial': adv_summary,
        'comparison': {
            'difference_mean': ped_summary['holdout']['mean_accuracy'] - adv_summary['holdout']['mean_accuracy'],
            'mann_whitney_u': float(u_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant_at_05': bool(p_value < 0.05),
            'significant_at_01': bool(p_value < 0.01),
        },
        'config': {
            'n_seeds': N_SEEDS,
            'n_steps': N_STEPS,
            'holdout_pairs': HOLDOUT_PAIRS,
        },
        'timestamp': datetime.now().isoformat()
    }

    # Save summary
    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"\nPedagogical ({N_SEEDS} seeds) - HOLDOUT PAIRS:")
    print(f"  Mean: {ped_summary['holdout']['mean_accuracy']:.1%} ± {ped_summary['holdout']['std_accuracy']:.1%}")
    print(f"  Range: [{ped_summary['holdout']['min_accuracy']:.1%}, {ped_summary['holdout']['max_accuracy']:.1%}]")
    print(f"  Individual: {[f'{a:.1%}' for a in ped_accs]}")

    print(f"\nAdversarial ({N_SEEDS} seeds) - HOLDOUT PAIRS:")
    print(f"  Mean: {adv_summary['holdout']['mean_accuracy']:.1%} ± {adv_summary['holdout']['std_accuracy']:.1%}")
    print(f"  Range: [{adv_summary['holdout']['min_accuracy']:.1%}, {adv_summary['holdout']['max_accuracy']:.1%}]")
    print(f"  Individual: {[f'{a:.1%}' for a in adv_accs]}")

    print(f"\nComparison:")
    print(f"  Difference: +{summary['comparison']['difference_mean']*100:.1f} percentage points")
    print(f"  Mann-Whitney U: {u_stat}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:.2f}")

    if p_value < 0.01:
        print(f"  Significance: p < 0.01 **")
    elif p_value < 0.05:
        print(f"  Significance: p < 0.05 *")
    else:
        print(f"  Significance: Not significant")

    print(f"\nEnd time: {datetime.now().isoformat()}")
    print(f"Results saved to: {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
