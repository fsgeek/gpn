"""Quick test of Fashion-MNIST relational holdout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Starting...", flush=True)

import torch
print("Torch imported", flush=True)

from src.training.relational_trainer import create_relational_trainer_holdout
print("Imports done", flush=True)

holdout_pairs = [(7, 3), (8, 2), (9, 1), (6, 4)]
device = torch.device('cuda')

print("Creating trainer...", flush=True)
trainer = create_relational_trainer_holdout(
    single_digit_checkpoint='checkpoints/fashion_mnist_pedagogical.pt',
    judge_checkpoint='checkpoints/fashion_relation_judge.pt',
    latent_dim=64,
    holdout_pairs=holdout_pairs,
    device=device,
    freeze_digits=True,
)
print("Trainer created!", flush=True)

print("Training one step...", flush=True)
metrics = trainer.train_step(batch_size=64)
print(f"Step completed: {metrics}", flush=True)
