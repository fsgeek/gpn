#!/usr/bin/env python3
"""
Training CLI for GPN-1.

Usage:
    python -m src.cli.train [--config PATH] [--mode gpn|gan] [--steps N]
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.weaver import create_weaver
from src.models.witness import create_witness
from src.models.judge import create_judge
from src.models.baseline_gan import create_baseline_gan
from src.training.config import TrainingConfig
from src.training.gpn_trainer import GPNTrainer
from src.training.gan_trainer import GANTrainer
from src.utils.reproducibility import set_reproducibility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_mnist_loader(
    batch_size: int = 64,
    num_workers: int = 4,
    train: bool = True,
    data_dir: str = "data",
) -> DataLoader:
    """Get MNIST data loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ])

    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_gpn(config: TrainingConfig, resume_from: str | None = None) -> dict:
    """
    Train GPN model.

    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from

    Returns:
        Final metrics
    """
    device = config.get_device()
    logger.info(f"Training GPN on {device}")

    # Set reproducibility
    set_reproducibility(config.seed)

    # Data
    train_loader = get_mnist_loader(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    # Models
    weaver = create_weaver(
        latent_dim=config.latent_dim,
        v_pred_dim=config.weaver.v_pred_dim,
        device=device,
    )
    witness = create_witness(
        v_seen_dim=config.witness.v_seen_dim,
        dropout=config.witness.dropout,
        device=device,
    )
    judge = create_judge(
        checkpoint_path=config.judge.checkpoint_path or "checkpoints/judge.pt",
        device=device,
        freeze=True,
    )

    logger.info(f"Weaver parameters: {sum(p.numel() for p in weaver.parameters()):,}")
    logger.info(f"Witness parameters: {sum(p.numel() for p in witness.parameters()):,}")
    logger.info(f"Judge parameters: {sum(p.numel() for p in judge.parameters()):,}")

    # Trainer
    trainer = GPNTrainer(
        config=config,
        weaver=weaver,
        witness=witness,
        judge=judge,
        train_loader=train_loader,
        device=device,
    )

    # Train
    final_metrics = trainer.train(resume_from=resume_from)

    # Evaluate
    eval_metrics = trainer.evaluate()
    final_metrics.update(eval_metrics)

    logger.info(f"Final metrics: {final_metrics}")

    return final_metrics


def train_gan(config: TrainingConfig) -> dict:
    """
    Train baseline GAN model.

    Args:
        config: Training configuration

    Returns:
        Final metrics
    """
    device = config.get_device()
    logger.info(f"Training baseline GAN on {device}")

    # Set reproducibility
    set_reproducibility(config.seed)

    # Data
    train_loader = get_mnist_loader(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    # Models
    generator, discriminator = create_baseline_gan(
        latent_dim=config.latent_dim,
        device=device,
    )
    judge = create_judge(
        checkpoint_path=config.judge.checkpoint_path or "checkpoints/judge.pt",
        device=device,
        freeze=True,
    )

    logger.info(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    logger.info(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Trainer
    trainer = GANTrainer(
        config=config,
        generator=generator,
        discriminator=discriminator,
        judge=judge,
        train_loader=train_loader,
        device=device,
    )

    # Train
    final_metrics = trainer.train()

    # Evaluate
    eval_metrics = trainer.evaluate()
    final_metrics.update(eval_metrics)

    logger.info(f"Final metrics: {final_metrics}")

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Train GPN-1 or baseline GAN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gpn1_default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gpn", "gan"],
        default="gpn",
        help="Training mode",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override total training steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Override device",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = TrainingConfig.from_yaml(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = TrainingConfig()
        logger.info("Using default config")

    # Override config with CLI args
    if args.steps is not None:
        config.total_steps = args.steps
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device

    # Train
    if args.mode == "gpn":
        metrics = train_gpn(config, resume_from=args.resume)
    else:
        metrics = train_gan(config)

    # Print final summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
