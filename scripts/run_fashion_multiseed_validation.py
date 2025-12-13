#!/usr/bin/env python3
"""
Multi-seed validation for Fashion-MNIST compositional transfer.
Mirrors MNIST validation for cross-domain consistency.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from training.relational_trainer import (
    create_relational_trainer_holdout,
    create_relational_trainer_holdout_acgan,
)

N_SEEDS = 5
N_STEPS = 5000
OUTPUT_DIR = Path("results/fashion_multiseed_validation")
HOLDOUT_PAIRS = [(7, 3), (8, 2), (9, 1), (6, 4)]


def run_pedagogical(seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Fashion-MNIST Pedagogical seed={seed} on {device}")
    print(f"{'='*60}")

    trainer = create_relational_trainer_holdout(
        single_digit_checkpoint="checkpoints/fashion_mnist_pedagogical.pt",
        judge_checkpoint="checkpoints/fashion_relation_judge.pt",
        holdout_pairs=HOLDOUT_PAIRS,
        device=device,
        freeze_digits=True,
    )

    for step in range(N_STEPS):
        metrics = trainer.train_step(batch_size=64)
        if step % 1000 == 0:
            print(f"  Step {step}: loss={metrics['loss']:.4f}, "
                  f"validity={metrics['avg_validity']:.3f}")

    holdout = trainer.evaluate(num_samples=1000, holdout_mode=True)
    print(f"  HOLDOUT: {holdout['relation_accuracy']:.1%}")
    
    return {
        'seed': seed,
        'holdout': holdout,
        'timestamp': datetime.now().isoformat()
    }


def run_adversarial(seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Fashion-MNIST Adversarial seed={seed} on {device}")
    print(f"{'='*60}")

    trainer = create_relational_trainer_holdout_acgan(
        acgan_checkpoint="checkpoints/fashion_mnist_adversarial.pt",
        judge_checkpoint="checkpoints/fashion_relation_judge.pt",
        holdout_pairs=HOLDOUT_PAIRS,
        device=device,
        freeze_digits=True,
    )

    for step in range(N_STEPS):
        metrics = trainer.train_step(batch_size=64)
        if step % 1000 == 0:
            print(f"  Step {step}: loss={metrics['loss']:.4f}, "
                  f"validity={metrics['avg_validity']:.3f}")

    holdout = trainer.evaluate(num_samples=1000, holdout_mode=True)
    print(f"  HOLDOUT: {holdout['relation_accuracy']:.1%}")
    
    return {
        'seed': seed,
        'holdout': holdout,
        'timestamp': datetime.now().isoformat()
    }


def main():
    from scipy import stats
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check checkpoints exist
    required = [
        "checkpoints/fashion_mnist_pedagogical.pt",
        "checkpoints/fashion_mnist_adversarial.pt",
        "checkpoints/fashion_relation_judge.pt",
    ]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        print(f"Missing checkpoints: {missing}")
        print("Fashion-MNIST multi-seed validation requires trained Fashion models.")
        return
    
    print(f"Start: {datetime.now().isoformat()}")
    
    ped_results = []
    for seed in range(N_SEEDS):
        result = run_pedagogical(seed)
        ped_results.append(result)
        with open(OUTPUT_DIR / f"pedagogical_seed_{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    adv_results = []
    for seed in range(N_SEEDS):
        result = run_adversarial(seed)
        adv_results.append(result)
        with open(OUTPUT_DIR / f"adversarial_seed_{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    ped_accs = [r['holdout']['relation_accuracy'] for r in ped_results]
    adv_accs = [r['holdout']['relation_accuracy'] for r in adv_results]
    
    u_stat, p_value = stats.mannwhitneyu(ped_accs, adv_accs, alternative='greater')
    pooled_std = np.sqrt((np.std(ped_accs)**2 + np.std(adv_accs)**2) / 2)
    cohens_d = (np.mean(ped_accs) - np.mean(adv_accs)) / pooled_std if pooled_std > 0 else float('inf')
    
    summary = {
        'pedagogical': {
            'mean': float(np.mean(ped_accs)),
            'std': float(np.std(ped_accs)),
            'accuracies': [float(a) for a in ped_accs],
        },
        'adversarial': {
            'mean': float(np.mean(adv_accs)),
            'std': float(np.std(adv_accs)),
            'accuracies': [float(a) for a in adv_accs],
        },
        'comparison': {
            'gap': float(np.mean(ped_accs) - np.mean(adv_accs)),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("FASHION-MNIST MULTI-SEED SUMMARY")
    print(f"{'='*60}")
    print(f"Pedagogical: {np.mean(ped_accs)*100:.1f}% ± {np.std(ped_accs)*100:.1f}%")
    print(f"Adversarial: {np.mean(adv_accs)*100:.1f}% ± {np.std(adv_accs)*100:.1f}%")
    print(f"Gap: +{(np.mean(ped_accs)-np.mean(adv_accs))*100:.1f}pp, p={p_value:.4f}")
    print(f"End: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
