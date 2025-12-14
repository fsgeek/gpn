#!/usr/bin/env python3
"""
Alignment Mechanism Ablation Experiment.

Tests whether v_pred/v_seen alignment is the mechanism that creates
bounded modification class and enables compositional transfer.

Hypothesis: If alignment creates bounded modification, removing it
should break compositional transfer.

Prediction if alignment IS the mechanism:
    Compositional transfer drops significantly (toward ~80% or worse)

Prediction if alignment ISN'T the mechanism:
    Compositional transfer remains high (~97%)

Protocol:
1. Train single-digit GPN WITHOUT alignment (alignment=0 all phases)
2. Train relational model on top of those primitives (frozen)
3. Test compositional transfer on held-out pairs
4. Compare to baseline pedagogical (with alignment)

Usage:
    python scripts/run_alignment_ablation.py

Output:
    results/alignment_ablation/
        single_digit_training.json
        relational_seed_0.json
        ...
        summary.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.weaver import create_weaver
from models.witness import create_witness
from models.judge import create_judge
from training.gpn_trainer import GPNTrainer
from training.config import TrainingConfig, LossWeights
from training.relational_trainer import create_relational_trainer_holdout
from utils.reproducibility import set_reproducibility

# Configuration
N_SEEDS = 5
SINGLE_DIGIT_STEPS = 15000
RELATIONAL_STEPS = 5000
OUTPUT_DIR = Path("results/alignment_ablation")
CHECKPOINT_DIR = Path("checkpoints/alignment_ablation")

# Hold-out pairs (same as original experiment)
HOLDOUT_PAIRS = [(7, 3), (8, 2), (9, 1), (6, 4)]


def get_mnist_loader(batch_size: int = 64) -> DataLoader:
    """Get MNIST data loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def train_single_digit_no_alignment(seed: int, device: torch.device) -> dict:
    """
    Train single-digit GPN WITHOUT alignment loss.

    This is the ablation: we keep everything else (grounding, empowerment, curriculum)
    but set alignment = 0 throughout all phases.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING SINGLE-DIGIT GPN WITHOUT ALIGNMENT (seed={seed})")
    print(f"{'='*60}")

    set_reproducibility(seed)

    # Load config from YAML and modify for ablation
    config = TrainingConfig.from_yaml("configs/gpn1_alignment_ablation.yaml")
    config.seed = seed

    # Ensure alignment is 0 in all phases (belt and suspenders)
    config.phase1_weights = LossWeights(grounding=1.0, alignment=0.0, empowerment=0.0)
    config.phase2_weights = LossWeights(grounding=1.0, alignment=0.0, empowerment=0.3)
    config.phase3_weights = LossWeights(grounding=0.0, alignment=0.0, empowerment=0.0)

    # Create models
    weaver = create_weaver(latent_dim=config.latent_dim, v_pred_dim=config.weaver.v_pred_dim, device=device)
    witness = create_witness(v_seen_dim=config.witness.v_seen_dim, dropout=config.witness.dropout, device=device)
    judge = create_judge(device=device)

    # Data
    train_loader = get_mnist_loader(batch_size=config.data.batch_size)

    # Create trainer with config
    trainer = GPNTrainer(
        config=config,
        weaver=weaver,
        witness=witness,
        judge=judge,
        train_loader=train_loader,
        device=device,
    )

    # Training loop - manually step to track metrics
    metrics_history = []
    for step in range(SINGLE_DIGIT_STEPS):
        # Update phase manager
        trainer.phase_manager.step(step)
        trainer.current_step = step

        # Do training step
        metrics = trainer.train_step()

        if step % 1000 == 0:
            loss = metrics.get('loss/total', 0)
            judge_acc = metrics.get('quality/judge_accuracy', 0)
            print(f"  Step {step} (Phase {trainer.phase_manager.current_phase}): "
                  f"loss={loss:.4f}, judge_acc={judge_acc:.1%}")
            metrics_history.append({
                'step': step,
                'phase': trainer.phase_manager.current_phase,
                'loss': float(loss),
                'judge_accuracy': float(judge_acc),
            })

    # Save checkpoint
    checkpoint_path = CHECKPOINT_DIR / f"no_alignment_seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'models': {
            'weaver': weaver.state_dict(),
            'witness': witness.state_dict(),
        },
        'seed': seed,
        'ablation': 'no_alignment',
        'final_metrics': metrics_history[-1] if metrics_history else {},
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"  Saved checkpoint: {checkpoint_path}")

    return {
        'seed': seed,
        'checkpoint_path': str(checkpoint_path),
        'final_metrics': metrics_history[-1] if metrics_history else {},
        'training_history': metrics_history,
    }


def run_relational_holdout_ablated(seed: int, ablated_checkpoint: str, device: torch.device) -> dict:
    """
    Run relational holdout experiment using ablated (no-alignment) primitives.
    """
    print(f"\n{'='*60}")
    print(f"RELATIONAL HOLDOUT WITH ABLATED PRIMITIVES (seed={seed})")
    print(f"{'='*60}")

    set_reproducibility(seed + 1000)  # Different seed for relational training

    # Create trainer using ablated primitives
    trainer = create_relational_trainer_holdout(
        single_digit_checkpoint=ablated_checkpoint,
        judge_checkpoint="checkpoints/relation_judge.pt",
        latent_dim=64,
        holdout_pairs=HOLDOUT_PAIRS,
        device=device,
        freeze_digits=True,
    )

    # Training loop
    for step in range(RELATIONAL_STEPS):
        metrics = trainer.train_step(batch_size=64)

        if step % 500 == 0:
            print(f"  Step {step}: loss={metrics['loss']:.4f}, "
                  f"x_acc={metrics['x_accuracy']:.1%}, "
                  f"y_acc={metrics['y_accuracy']:.1%}")

    # Evaluate on holdout pairs
    print("\n  EVALUATING ON HOLDOUT PAIRS...")
    holdout_metrics = trainer.evaluate(num_samples=1000, holdout_mode=True)
    training_metrics = trainer.evaluate(num_samples=1000, holdout_mode=False)

    print(f"\n  HOLDOUT RESULTS:")
    print(f"    Relation accuracy: {holdout_metrics['relation_accuracy']:.1%}")

    return {
        'seed': seed,
        'ablated_checkpoint': ablated_checkpoint,
        'holdout': {
            'relation_accuracy': holdout_metrics['relation_accuracy'],
            'x_accuracy': holdout_metrics['x_accuracy'],
            'y_accuracy': holdout_metrics['y_accuracy'],
        },
        'training': {
            'relation_accuracy': training_metrics['relation_accuracy'],
        },
    }


def main():
    from scipy import stats

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("ALIGNMENT MECHANISM ABLATION EXPERIMENT")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seeds: {N_SEEDS}")
    print(f"Single-digit steps: {SINGLE_DIGIT_STEPS}")
    print(f"Relational steps: {RELATIONAL_STEPS}")
    print(f"Holdout pairs: {HOLDOUT_PAIRS}")
    print(f"Start time: {datetime.now().isoformat()}")

    # Check for relation judge
    if not Path("checkpoints/relation_judge.pt").exists():
        print("ERROR: Missing checkpoints/relation_judge.pt")
        return

    # Phase 1: Train single-digit models without alignment
    print("\n" + "=" * 60)
    print("PHASE 1: TRAINING SINGLE-DIGIT MODELS (NO ALIGNMENT)")
    print("=" * 60)

    single_digit_results = []
    for seed in range(N_SEEDS):
        result = train_single_digit_no_alignment(seed, device)
        single_digit_results.append(result)

        with open(OUTPUT_DIR / f"single_digit_seed_{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)

    # Phase 2: Run relational holdout on each
    print("\n" + "=" * 60)
    print("PHASE 2: RELATIONAL HOLDOUT WITH ABLATED PRIMITIVES")
    print("=" * 60)

    relational_results = []
    for i, sd_result in enumerate(single_digit_results):
        result = run_relational_holdout_ablated(
            seed=i,
            ablated_checkpoint=sd_result['checkpoint_path'],
            device=device,
        )
        relational_results.append(result)

        with open(OUTPUT_DIR / f"relational_seed_{i}.json", 'w') as f:
            json.dump(result, f, indent=2)

    # Compute summary
    holdout_accs = [r['holdout']['relation_accuracy'] for r in relational_results]

    ablated_summary = {
        'n_seeds': N_SEEDS,
        'mean_accuracy': float(np.mean(holdout_accs)),
        'std_accuracy': float(np.std(holdout_accs)),
        'min_accuracy': float(np.min(holdout_accs)),
        'max_accuracy': float(np.max(holdout_accs)),
        'accuracies': [float(a) for a in holdout_accs],
    }

    # Compare to baseline pedagogical (from previous results)
    baseline_file = Path("results/multiseed_validation/summary.json")
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline = json.load(f)
        baseline_accs = baseline['pedagogical']['holdout']['accuracies']
        baseline_mean = baseline['pedagogical']['holdout']['mean_accuracy']
        baseline_std = baseline['pedagogical']['holdout']['std_accuracy']

        # Statistical comparison
        u_stat, p_value = stats.mannwhitneyu(baseline_accs, holdout_accs, alternative='greater')
        pooled_std = np.sqrt((np.std(baseline_accs)**2 + np.std(holdout_accs)**2) / 2)
        if pooled_std > 0:
            cohens_d = (baseline_mean - ablated_summary['mean_accuracy']) / pooled_std
        else:
            cohens_d = float('inf')

        comparison = {
            'baseline_pedagogical': {
                'mean': baseline_mean,
                'std': baseline_std,
                'accuracies': baseline_accs,
            },
            'ablated_no_alignment': ablated_summary,
            'difference_mean': baseline_mean - ablated_summary['mean_accuracy'],
            'mann_whitney_u': float(u_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
        }
    else:
        comparison = {
            'note': 'No baseline comparison available - run multiseed validation first',
            'ablated_no_alignment': ablated_summary,
        }

    # Determine if alignment is the mechanism
    if ablated_summary['mean_accuracy'] < 0.85:  # Significant drop from ~97%
        interpretation = "ALIGNMENT IS THE MECHANISM: Removing alignment caused compositional collapse"
    else:
        interpretation = "ALIGNMENT IS NOT THE MECHANISM: Composition survived without alignment"

    summary = {
        'experiment': 'alignment_ablation',
        'hypothesis': 'v_pred/v_seen alignment creates bounded modification class',
        'ablation': 'alignment = 0 throughout all phases',
        'ablated_results': ablated_summary,
        'comparison': comparison,
        'interpretation': interpretation,
        'timestamp': datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

    print(f"\nABLATED (no alignment) - {N_SEEDS} seeds:")
    print(f"  Mean: {ablated_summary['mean_accuracy']:.1%} ± {ablated_summary['std_accuracy']:.1%}")
    print(f"  Individual: {[f'{a:.1%}' for a in holdout_accs]}")

    if 'baseline_pedagogical' in comparison:
        print(f"\nBASELINE (with alignment):")
        print(f"  Mean: {comparison['baseline_pedagogical']['mean']:.1%} ± {comparison['baseline_pedagogical']['std']:.1%}")
        print(f"\nComparison:")
        print(f"  Difference: {comparison['difference_mean']*100:+.1f} percentage points")
        print(f"  p-value: {comparison['p_value']:.4f}")
        print(f"  Cohen's d: {comparison['cohens_d']:.2f}")

    print(f"\n{'='*60}")
    print(f"INTERPRETATION: {interpretation}")
    print(f"{'='*60}")

    print(f"\nEnd time: {datetime.now().isoformat()}")
    print(f"Results saved to: {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
