#!/usr/bin/env python3
"""
Loss Component Ablation Experiments.

Tests which loss components are necessary for compositional transfer:
1. Grounding ablation: Remove Judge supervision
2. Empowerment ablation: Remove diversity pressure

Following the alignment ablation finding that alignment is NOT the mechanism,
these experiments test the remaining candidates.

Hypothesis: At least one of grounding or empowerment is necessary for composition.

Protocol:
1. Train single-digit GPN with each ablation (5 seeds each)
2. Train relational model on top of those primitives (frozen)
3. Test compositional transfer on held-out pairs
4. Compare to baseline and alignment-ablated

Usage:
    python scripts/run_loss_ablations.py --ablation grounding
    python scripts/run_loss_ablations.py --ablation empowerment
    python scripts/run_loss_ablations.py --ablation both

Output:
    results/{ablation}_ablation/
        single_digit_seed_*.json
        relational_seed_*.json
        summary.json
"""

import argparse
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


def train_single_digit_ablated(
    seed: int,
    ablation_type: str,
    config_path: str,
    checkpoint_dir: Path,
    device: torch.device
) -> dict:
    """
    Train single-digit GPN with specified ablation.

    Args:
        seed: Random seed
        ablation_type: 'grounding' or 'empowerment'
        config_path: Path to ablation config YAML
        checkpoint_dir: Where to save checkpoints
        device: Torch device
    """
    print(f"\n{'='*60}")
    print(f"TRAINING SINGLE-DIGIT GPN - {ablation_type.upper()} ABLATED (seed={seed})")
    print(f"{'='*60}")

    set_reproducibility(seed)

    # Load config from YAML
    config = TrainingConfig.from_yaml(config_path)
    config.seed = seed

    # Belt and suspenders: ensure ablation is correctly set
    if ablation_type == 'grounding':
        config.phase1_weights = LossWeights(grounding=0.0, alignment=0.1, empowerment=0.0)
        config.phase2_weights = LossWeights(grounding=0.0, alignment=0.5, empowerment=0.3)
        config.phase3_weights = LossWeights(grounding=0.0, alignment=0.0, empowerment=0.0)
    elif ablation_type == 'empowerment':
        config.phase1_weights = LossWeights(grounding=1.0, alignment=0.1, empowerment=0.0)
        config.phase2_weights = LossWeights(grounding=1.0, alignment=0.5, empowerment=0.0)
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
    checkpoint_path = checkpoint_dir / f"{ablation_type}_ablated_seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'models': {
            'weaver': weaver.state_dict(),
            'witness': witness.state_dict(),
        },
        'seed': seed,
        'ablation': ablation_type,
        'final_metrics': metrics_history[-1] if metrics_history else {},
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"  Saved checkpoint: {checkpoint_path}")

    return {
        'seed': seed,
        'ablation': ablation_type,
        'checkpoint_path': str(checkpoint_path),
        'final_metrics': metrics_history[-1] if metrics_history else {},
        'training_history': metrics_history,
    }


def run_relational_holdout(
    seed: int,
    ablation_type: str,
    checkpoint_path: str,
    device: torch.device
) -> dict:
    """
    Run relational holdout experiment using ablated primitives.
    """
    print(f"\n{'='*60}")
    print(f"RELATIONAL HOLDOUT - {ablation_type.upper()} ABLATED PRIMITIVES (seed={seed})")
    print(f"{'='*60}")

    set_reproducibility(seed + 1000)  # Different seed for relational training

    # Create trainer using ablated primitives
    trainer = create_relational_trainer_holdout(
        single_digit_checkpoint=checkpoint_path,
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
        'ablation': ablation_type,
        'ablated_checkpoint': checkpoint_path,
        'holdout': {
            'relation_accuracy': holdout_metrics['relation_accuracy'],
            'x_accuracy': holdout_metrics['x_accuracy'],
            'y_accuracy': holdout_metrics['y_accuracy'],
        },
        'training': {
            'relation_accuracy': training_metrics['relation_accuracy'],
        },
    }


def run_ablation_experiment(ablation_type: str, device: torch.device) -> dict:
    """Run complete ablation experiment for one loss component."""
    from scipy import stats

    output_dir = Path(f"results/{ablation_type}_ablation")
    checkpoint_dir = Path(f"checkpoints/{ablation_type}_ablation")
    config_path = f"configs/gpn1_{ablation_type}_ablation.yaml"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"{ablation_type.upper()} ABLATION EXPERIMENT")
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
        return None

    # Phase 1: Train single-digit models with ablation
    print(f"\n{'='*60}")
    print(f"PHASE 1: TRAINING SINGLE-DIGIT MODELS ({ablation_type.upper()} ABLATED)")
    print("=" * 60)

    single_digit_results = []
    for seed in range(N_SEEDS):
        result = train_single_digit_ablated(
            seed=seed,
            ablation_type=ablation_type,
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
        single_digit_results.append(result)

        with open(output_dir / f"single_digit_seed_{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)

    # Phase 2: Run relational holdout on each
    print(f"\n{'='*60}")
    print(f"PHASE 2: RELATIONAL HOLDOUT WITH {ablation_type.upper()} ABLATED PRIMITIVES")
    print("=" * 60)

    relational_results = []
    for i, sd_result in enumerate(single_digit_results):
        result = run_relational_holdout(
            seed=i,
            ablation_type=ablation_type,
            checkpoint_path=sd_result['checkpoint_path'],
            device=device,
        )
        relational_results.append(result)

        with open(output_dir / f"relational_seed_{i}.json", 'w') as f:
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
    comparison = {}

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
            cohens_d = float('inf') if baseline_mean != ablated_summary['mean_accuracy'] else 0.0

        comparison = {
            'baseline_pedagogical': {
                'mean': baseline_mean,
                'std': baseline_std,
                'accuracies': baseline_accs,
            },
            f'{ablation_type}_ablated': ablated_summary,
            'difference_mean': baseline_mean - ablated_summary['mean_accuracy'],
            'mann_whitney_u': float(u_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
        }
    else:
        comparison = {
            'note': 'No baseline comparison available - run multiseed validation first',
            f'{ablation_type}_ablated': ablated_summary,
        }

    # Also compare to alignment ablation if available
    alignment_ablation_file = Path("results/alignment_ablation/summary.json")
    if alignment_ablation_file.exists():
        with open(alignment_ablation_file) as f:
            alignment_ablation = json.load(f)
        comparison['alignment_ablated'] = alignment_ablation['ablated_results']

    # Determine if this loss is the mechanism
    if ablated_summary['mean_accuracy'] < 0.85:  # Significant drop from ~97%
        interpretation = f"{ablation_type.upper()} IS THE MECHANISM: Removing {ablation_type} caused compositional collapse"
    else:
        interpretation = f"{ablation_type.upper()} IS NOT THE MECHANISM: Composition survived without {ablation_type}"

    summary = {
        'experiment': f'{ablation_type}_ablation',
        'hypothesis': f'{ablation_type} loss creates compositional capacity',
        'ablation': f'{ablation_type} = 0 throughout all phases',
        'ablated_results': ablated_summary,
        'comparison': comparison,
        'interpretation': interpretation,
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final report
    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print("=" * 60)

    print(f"\n{ablation_type.upper()} ABLATED - {N_SEEDS} seeds:")
    print(f"  Mean: {ablated_summary['mean_accuracy']:.1%} +/- {ablated_summary['std_accuracy']:.1%}")
    print(f"  Individual: {[f'{a:.1%}' for a in holdout_accs]}")

    if 'baseline_pedagogical' in comparison:
        print(f"\nBASELINE (with {ablation_type}):")
        print(f"  Mean: {comparison['baseline_pedagogical']['mean']:.1%} +/- {comparison['baseline_pedagogical']['std']:.1%}")
        print(f"\nComparison:")
        print(f"  Difference: {comparison['difference_mean']*100:+.1f} percentage points")
        print(f"  p-value: {comparison['p_value']:.4f}")
        print(f"  Cohen's d: {comparison['cohens_d']:.2f}")

    if 'alignment_ablated' in comparison:
        print(f"\nALIGNMENT ABLATED (for comparison):")
        print(f"  Mean: {comparison['alignment_ablated']['mean_accuracy']:.1%}")

    print(f"\n{'='*60}")
    print(f"INTERPRETATION: {interpretation}")
    print(f"{'='*60}")

    print(f"\nEnd time: {datetime.now().isoformat()}")
    print(f"Results saved to: {output_dir / 'summary.json'}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Run loss component ablation experiments')
    parser.add_argument('--ablation', type=str, choices=['grounding', 'empowerment', 'both'],
                        default='both', help='Which ablation to run')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    if args.ablation in ['grounding', 'both']:
        results['grounding'] = run_ablation_experiment('grounding', device)

    if args.ablation in ['empowerment', 'both']:
        results['empowerment'] = run_ablation_experiment('empowerment', device)

    # If both were run, create combined analysis
    if args.ablation == 'both':
        combined_output = Path("results/loss_ablation_combined")
        combined_output.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("COMBINED ABLATION ANALYSIS")
        print("=" * 60)

        # Load all results
        ablation_results = {}
        for ablation_type in ['alignment', 'grounding', 'empowerment']:
            result_file = Path(f"results/{ablation_type}_ablation/summary.json")
            if result_file.exists():
                with open(result_file) as f:
                    ablation_results[ablation_type] = json.load(f)

        # Summary table
        print("\n| Ablation | Mean Accuracy | Std | Interpretation |")
        print("|----------|---------------|-----|----------------|")
        for ablation_type, result in ablation_results.items():
            mean = result['ablated_results']['mean_accuracy']
            std = result['ablated_results']['std_accuracy']
            interp = "MECHANISM" if mean < 0.85 else "NOT mechanism"
            print(f"| {ablation_type:11} | {mean:.1%} | {std:.1%} | {interp} |")

        # Save combined summary
        combined = {
            'experiments': ablation_results,
            'conclusion': determine_mechanism(ablation_results),
            'timestamp': datetime.now().isoformat(),
        }

        with open(combined_output / "combined_summary.json", 'w') as f:
            json.dump(combined, f, indent=2)

        print(f"\nCombined results saved to: {combined_output / 'combined_summary.json'}")


def determine_mechanism(ablation_results: dict) -> str:
    """Determine what the mechanism is based on all ablation results."""
    mechanisms = []
    non_mechanisms = []

    for ablation_type, result in ablation_results.items():
        mean = result['ablated_results']['mean_accuracy']
        if mean < 0.85:
            mechanisms.append(ablation_type)
        else:
            non_mechanisms.append(ablation_type)

    if not mechanisms:
        return "NO SINGLE LOSS IS THE MECHANISM: Composition survives all individual ablations. Mechanism may be: (1) curriculum structure itself, (2) architecture (frozen primitives + trainable router), or (3) interaction between losses."
    elif len(mechanisms) == 1:
        return f"{mechanisms[0].upper()} IS THE MECHANISM: Only removing {mechanisms[0]} breaks composition."
    else:
        return f"MULTIPLE MECHANISMS: {', '.join(m.upper() for m in mechanisms)} are all necessary for composition."


if __name__ == "__main__":
    main()
