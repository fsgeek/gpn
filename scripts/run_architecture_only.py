#!/usr/bin/env python3
"""
Architecture-Only Ablation: The Critical Experiment.

Tests whether the mechanism for compositional transfer is purely architectural
(frozen primitives + trainable router) rather than any loss function.

If composition survives with ALL losses = 0:
    The losses were noise. The paper is about architecture, not pedagogy.
    Bounded modification is enforced by which parameters are trainable.

If composition fails:
    Something about loss interaction or curriculum timing matters.

Protocol:
1. Train single-digit GPN with ALL losses = 0 (5 seeds)
2. Train relational model on top of those primitives (frozen)
3. Test compositional transfer on held-out pairs
4. Compare to baseline and individual ablations

Usage:
    python scripts/run_architecture_only.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np

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
OUTPUT_DIR = Path("results/architecture_only")
CHECKPOINT_DIR = Path("checkpoints/architecture_only")

HOLDOUT_PAIRS = [(7, 3), (8, 2), (9, 1), (6, 4)]


def get_mnist_loader(batch_size: int = 64) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def train_architecture_only(seed: int, device: torch.device) -> dict:
    """
    Train single-digit GPN with ALL losses = 0.

    This is pure architecture: the model runs through phases but learns nothing
    from the pedagogical losses. Any capability comes from random initialization
    and architectural constraints alone.
    """
    print(f"\n{'='*60}")
    print(f"ARCHITECTURE ONLY - NO LOSSES (seed={seed})")
    print(f"{'='*60}")

    set_reproducibility(seed)

    config = TrainingConfig.from_yaml("configs/gpn1_architecture_only.yaml")
    config.seed = seed

    # Ensure ALL losses are 0 in ALL phases
    config.phase1_weights = LossWeights(grounding=0.0, alignment=0.0, empowerment=0.0)
    config.phase2_weights = LossWeights(grounding=0.0, alignment=0.0, empowerment=0.0)
    config.phase3_weights = LossWeights(grounding=0.0, alignment=0.0, empowerment=0.0)

    weaver = create_weaver(latent_dim=config.latent_dim, v_pred_dim=config.weaver.v_pred_dim, device=device)
    witness = create_witness(v_seen_dim=config.witness.v_seen_dim, dropout=config.witness.dropout, device=device)
    judge = create_judge(device=device)

    train_loader = get_mnist_loader(batch_size=config.data.batch_size)

    trainer = GPNTrainer(
        config=config,
        weaver=weaver,
        witness=witness,
        judge=judge,
        train_loader=train_loader,
        device=device,
    )

    metrics_history = []
    for step in range(SINGLE_DIGIT_STEPS):
        trainer.phase_manager.step(step)
        trainer.current_step = step
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

    checkpoint_path = CHECKPOINT_DIR / f"architecture_only_seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'models': {
            'weaver': weaver.state_dict(),
            'witness': witness.state_dict(),
        },
        'seed': seed,
        'ablation': 'architecture_only',
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


def run_relational_holdout(seed: int, checkpoint_path: str, device: torch.device) -> dict:
    """Run relational holdout with architecture-only primitives."""
    print(f"\n{'='*60}")
    print(f"RELATIONAL HOLDOUT - ARCHITECTURE ONLY PRIMITIVES (seed={seed})")
    print(f"{'='*60}")

    set_reproducibility(seed + 1000)

    trainer = create_relational_trainer_holdout(
        single_digit_checkpoint=checkpoint_path,
        judge_checkpoint="checkpoints/relation_judge.pt",
        latent_dim=64,
        holdout_pairs=HOLDOUT_PAIRS,
        device=device,
        freeze_digits=True,
    )

    for step in range(RELATIONAL_STEPS):
        metrics = trainer.train_step(batch_size=64)

        if step % 500 == 0:
            print(f"  Step {step}: loss={metrics['loss']:.4f}, "
                  f"x_acc={metrics['x_accuracy']:.1%}, "
                  f"y_acc={metrics['y_accuracy']:.1%}")

    print("\n  EVALUATING ON HOLDOUT PAIRS...")
    holdout_metrics = trainer.evaluate(num_samples=1000, holdout_mode=True)
    training_metrics = trainer.evaluate(num_samples=1000, holdout_mode=False)

    print(f"\n  HOLDOUT RESULTS:")
    print(f"    Relation accuracy: {holdout_metrics['relation_accuracy']:.1%}")

    return {
        'seed': seed,
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


def main():
    from scipy import stats

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("ARCHITECTURE-ONLY ABLATION: THE CRITICAL EXPERIMENT")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seeds: {N_SEEDS}")
    print(f"ALL LOSSES = 0 in ALL PHASES")
    print(f"Start time: {datetime.now().isoformat()}")

    if not Path("checkpoints/relation_judge.pt").exists():
        print("ERROR: Missing checkpoints/relation_judge.pt")
        return

    # Phase 1: Train with no losses
    print(f"\n{'='*60}")
    print("PHASE 1: TRAINING WITH NO LOSSES (ARCHITECTURE ONLY)")
    print("=" * 60)

    single_digit_results = []
    for seed in range(N_SEEDS):
        result = train_architecture_only(seed, device)
        single_digit_results.append(result)

        with open(OUTPUT_DIR / f"single_digit_seed_{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)

    # Phase 2: Relational holdout
    print(f"\n{'='*60}")
    print("PHASE 2: RELATIONAL HOLDOUT")
    print("=" * 60)

    relational_results = []
    for i, sd_result in enumerate(single_digit_results):
        result = run_relational_holdout(
            seed=i,
            checkpoint_path=sd_result['checkpoint_path'],
            device=device,
        )
        relational_results.append(result)

        with open(OUTPUT_DIR / f"relational_seed_{i}.json", 'w') as f:
            json.dump(result, f, indent=2)

    # Compute summary
    holdout_accs = [r['holdout']['relation_accuracy'] for r in relational_results]

    results_summary = {
        'n_seeds': N_SEEDS,
        'mean_accuracy': float(np.mean(holdout_accs)),
        'std_accuracy': float(np.std(holdout_accs)),
        'min_accuracy': float(np.min(holdout_accs)),
        'max_accuracy': float(np.max(holdout_accs)),
        'accuracies': [float(a) for a in holdout_accs],
    }

    # Load comparisons
    comparisons = {}

    baseline_file = Path("results/multiseed_validation/summary.json")
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline = json.load(f)
        comparisons['baseline'] = {
            'mean': baseline['pedagogical']['holdout']['mean_accuracy'],
            'std': baseline['pedagogical']['holdout']['std_accuracy'],
            'accuracies': baseline['pedagogical']['holdout']['accuracies'],
        }

    for ablation in ['alignment', 'grounding', 'empowerment']:
        ablation_file = Path(f"results/{ablation}_ablation/summary.json")
        if ablation_file.exists():
            with open(ablation_file) as f:
                data = json.load(f)
            comparisons[f'{ablation}_ablated'] = data['ablated_results']

    # Interpretation
    if results_summary['mean_accuracy'] >= 0.95:
        interpretation = "ARCHITECTURE IS THE MECHANISM: Composition survives with NO losses. The losses were noise. Bounded modification is enforced by frozen parameters, not gradients."
    elif results_summary['mean_accuracy'] >= 0.85:
        interpretation = "PARTIAL EFFECT: Composition partially survives. Some loss signal may be necessary, but architecture does most of the work."
    else:
        interpretation = "LOSSES MATTER: Composition fails without losses. Some aspect of loss interaction or curriculum is necessary."

    summary = {
        'experiment': 'architecture_only',
        'hypothesis': 'Compositional transfer requires only frozen primitives + trainable router',
        'ablation': 'ALL losses = 0 throughout ALL phases',
        'results': results_summary,
        'comparisons': comparisons,
        'interpretation': interpretation,
        'timestamp': datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final report
    print(f"\n{'='*60}")
    print("FINAL REPORT: ARCHITECTURE-ONLY ABLATION")
    print("=" * 60)

    print(f"\nARCHITECTURE ONLY (no losses) - {N_SEEDS} seeds:")
    print(f"  Mean: {results_summary['mean_accuracy']:.1%} +/- {results_summary['std_accuracy']:.1%}")
    print(f"  Individual: {[f'{a:.1%}' for a in holdout_accs]}")

    print(f"\nCOMPARISON:")
    if 'baseline' in comparisons:
        print(f"  Baseline (all losses): {comparisons['baseline']['mean']:.1%}")
    for name, data in comparisons.items():
        if name != 'baseline':
            print(f"  {name}: {data['mean_accuracy']:.1%}")

    print(f"\n{'='*60}")
    print(f"INTERPRETATION: {interpretation}")
    print(f"{'='*60}")

    # The paper implication
    if results_summary['mean_accuracy'] >= 0.95:
        print(f"\n{'='*60}")
        print("PAPER IMPLICATION:")
        print("The paper is NOT about 'pedagogical training bounds modification.'")
        print("The paper is about 'architectural constraints bound modification.'")
        print("The losses we studied were irrelevant noise.")
        print("This is simpler. Less flattering to original framing. More true.")
        print(f"{'='*60}")

    print(f"\nEnd time: {datetime.now().isoformat()}")
    print(f"Results saved to: {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
