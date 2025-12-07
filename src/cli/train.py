#!/usr/bin/env python3
"""
Training CLI for GPN-1.

Usage:
    python -m src.cli.train [--config PATH] [--mode gpn|gpn-v2|gan] [--steps N]
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
from src.training.gpn_trainer_v2 import GPNTrainerV2
from src.training.gpn_trainer_v3 import GPNTrainerV3
from src.training.gpn_trainer_v3_no_meta import GPNTrainerV3NoMeta
from src.training.gpn_trainer_twodigit import GPNTrainerTwoDigit
from src.training.gpn_trainer_twodigit_direct import GPNTrainerTwoDigitDirect
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


def train_gpn_v2(config: TrainingConfig, resume_from: str | None = None) -> dict:
    """
    Train GPN model with grounded Witness architecture (V2).

    Key difference from V1: Witness is trained on real MNIST to develop
    independent competence before providing feedback to Weaver.

    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from

    Returns:
        Final metrics
    """
    device = config.get_device()
    logger.info(f"Training GPN V2 (Grounded Witness) on {device}")

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

    # Trainer V2
    trainer = GPNTrainerV2(
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


def train_gpn_v3(config: TrainingConfig, resume_from: str | None = None, competence_threshold: float = 0.5) -> dict:
    """
    Train GPN model with meta-learning architecture (V3).

    Key difference from V2: Weaver is rewarded for Witness's IMPROVEMENT
    on real data, not for Witness's approval. This prevents exploitation
    of random-Witness.

    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from

    Returns:
        Final metrics
    """
    device = config.get_device()
    logger.info(f"Training GPN V3 (Meta-Learning) on {device}")

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

    # Trainer V3
    trainer = GPNTrainerV3(
        config=config,
        weaver=weaver,
        witness=witness,
        judge=judge,
        train_loader=train_loader,
        device=device,
        competence_threshold=competence_threshold,
    )

    # Train
    final_metrics = trainer.train(resume_from=resume_from)

    # Evaluate
    eval_metrics = trainer.evaluate()
    final_metrics.update(eval_metrics)

    logger.info(f"Final metrics: {final_metrics}")

    return final_metrics


def train_gpn_v3_no_meta(config: TrainingConfig, resume_from: str | None = None, competence_threshold: float = 0.5) -> dict:
    """
    Train GPN V3 ablation: No meta-learning inner loop.

    Tests whether consistent grounding + competence gating alone prevents
    collapse, or if the meta-learning optimization target is necessary.

    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from
        competence_threshold: Witness competence threshold

    Returns:
        Final metrics
    """
    device = config.get_device()
    logger.info(f"Training GPN V3-NoMeta (Ablation) on {device}")

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

    # Trainer V3 No-Meta (Ablation)
    trainer = GPNTrainerV3NoMeta(
        config=config,
        weaver=weaver,
        witness=witness,
        judge=judge,
        train_loader=train_loader,
        device=device,
        competence_threshold=competence_threshold,
    )

    # Train
    final_metrics = trainer.train(resume_from=resume_from)

    # Evaluate
    eval_metrics = trainer.evaluate()
    final_metrics.update(eval_metrics)

    logger.info(f"Final metrics: {final_metrics}")

    return final_metrics


def train_gpn2(
    config: TrainingConfig,
    resume_from: str | None = None,
    single_digit_checkpoint: str = "checkpoints/checkpoint_v3_final.pt",
    judge_checkpoint: str = "checkpoints/judge_twodigit.pt",
) -> dict:
    """
    Train GPN-2 (Two-Digit) with curriculum learning.

    Curriculum phases:
    - Phase 1: Composition learning (single-digit Weaver frozen)
    - Phase 2: End-to-end fine-tuning (all unfrozen)
    - Phase 3: Drift test (grounding removed)

    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from
        single_digit_checkpoint: Path to pre-trained single-digit Weaver
        judge_checkpoint: Path to pre-trained 2-digit Judge

    Returns:
        Final metrics
    """
    from src.models.weaver_twodigit import create_twodigit_weaver
    from src.models.witness_twodigit import create_twodigit_witness
    from src.models.judge_twodigit import create_twodigit_judge
    from src.data.multidigit import get_twodigit_loader

    device = config.get_device()
    logger.info(f"Training GPN-2 (Two-Digit Curriculum) on {device}")

    # Set reproducibility
    set_reproducibility(config.seed)

    # Data (2-digit MNIST)
    train_loader = get_twodigit_loader(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        train=True,
        num_samples=50000,
    )

    # Models
    weaver = create_twodigit_weaver(
        single_digit_checkpoint=single_digit_checkpoint,
        latent_dim=config.latent_dim,
        v_pred_dim=config.weaver.v_pred_dim,
        device=device,
        freeze_digits=True,  # Phase 1: frozen
    )
    witness = create_twodigit_witness(
        v_seen_dim=config.witness.v_seen_dim,
        dropout=config.witness.dropout,
        device=device,
    )
    judge = create_twodigit_judge(
        checkpoint_path=judge_checkpoint,
        device=device,
        freeze=True,
    )

    logger.info(f"TwoDigitWeaver parameters: {sum(p.numel() for p in weaver.parameters()):,}")
    logger.info(f"TwoDigitWitness parameters: {sum(p.numel() for p in witness.parameters()):,}")
    logger.info(f"TwoDigitJudge parameters: {sum(p.numel() for p in judge.parameters()):,}")
    logger.info(f"Single-digit Weaver frozen: {weaver.is_digit_weaver_frozen()}")

    # Trainer
    trainer = GPNTrainerTwoDigit(
        weaver=weaver,
        witness=witness,
        judge=judge,
        train_loader=train_loader,
        config=config,
        device=device,
        latent_dim=config.latent_dim,
        v_pred_dim=config.weaver.v_pred_dim,
    )

    # Train
    final_metrics = trainer.train(resume_from=resume_from)

    # Evaluate
    eval_metrics = trainer.evaluate()
    final_metrics.update(eval_metrics)

    logger.info(f"Final metrics: {final_metrics}")

    return final_metrics


def train_gpn2_direct(
    config: TrainingConfig,
    resume_from: str | None = None,
    judge_checkpoint: str = "checkpoints/judge_twodigit.pt",
) -> dict:
    """
    Train GPN-2 DIRECT (No Curriculum Ablation) - from scratch.

    Critical ablation: tests whether curriculum helps or composition is free.

    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from
        judge_checkpoint: Path to pre-trained 2-digit Judge

    Returns:
        Final metrics
    """
    from src.models.weaver_twodigit import create_twodigit_weaver_direct
    from src.models.witness_twodigit import create_twodigit_witness
    from src.models.judge_twodigit import create_twodigit_judge
    from src.data.multidigit import get_twodigit_loader

    device = config.get_device()
    logger.info(f"Training GPN-2 DIRECT (No Curriculum Ablation) on {device}")
    logger.info("ABLATION: Training 2-digit from scratch, NO pre-trained single-digit Weaver")

    # Set reproducibility
    set_reproducibility(config.seed)

    # Data (2-digit MNIST)
    train_loader = get_twodigit_loader(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        train=True,
        num_samples=50000,
    )

    # Models - DIRECT Weaver (no composition, from scratch)
    weaver = create_twodigit_weaver_direct(
        latent_dim=config.latent_dim,
        v_pred_dim=config.weaver.v_pred_dim,
        device=device,
    )
    witness = create_twodigit_witness(
        v_seen_dim=config.witness.v_seen_dim,
        dropout=config.witness.dropout,
        device=device,
    )
    judge = create_twodigit_judge(
        checkpoint_path=judge_checkpoint,
        device=device,
        freeze=True,
    )

    logger.info(f"TwoDigitWeaverDirect params: {sum(p.numel() for p in weaver.parameters()):,}")
    logger.info(f"TwoDigitWitness params: {sum(p.numel() for p in witness.parameters()):,}")
    logger.info(f"TwoDigitJudge params: {sum(p.numel() for p in judge.parameters()):,}")

    # Trainer - Direct (no curriculum)
    trainer = GPNTrainerTwoDigitDirect(
        weaver=weaver,
        witness=witness,
        judge=judge,
        train_loader=train_loader,
        device=device,
        latent_dim=config.latent_dim,
        v_pred_dim=config.weaver.v_pred_dim,
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


def train_acgan(
    config: TrainingConfig,
    judge_checkpoint: str = "checkpoints/judge.pt",
) -> dict:
    """
    Train AC-GAN single-digit (curriculum foundation).

    Hybrid adversarial + pedagogical training on MNIST.
    This is the curriculum foundation for AC-GAN 2-digit composition.

    Args:
        config: Training configuration
        judge_checkpoint: Path to pre-trained single-digit Judge

    Returns:
        Final metrics
    """
    from src.models.acgan import create_acgan
    from src.models.judge import create_judge
    from src.training.acgan_trainer import ACGANTrainer

    device = config.get_device()
    logger.info(f"Training AC-GAN single-digit on {device}")
    logger.info("HYBRID: Adversarial + Pedagogical (class prediction)")

    # Set reproducibility
    set_reproducibility(config.seed)

    # Data
    train_loader = get_mnist_loader(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    # Models
    generator, discriminator = create_acgan(
        latent_dim=config.latent_dim,
        device=device,
    )
    judge = create_judge(
        checkpoint_path=judge_checkpoint,
        device=device,
        freeze=True,
    )

    logger.info(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    logger.info(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Trainer
    trainer = ACGANTrainer(
        config=config,
        generator=generator,
        discriminator=discriminator,
        judge=judge,
        train_loader=train_loader,
        device=device,
    )

    # Train
    final_metrics = trainer.train()

    return final_metrics


def train_acgan2(
    config: TrainingConfig,
    single_digit_checkpoint: str = "checkpoints/acgan_final.pt",
    judge_checkpoint: str = "checkpoints/judge_twodigit.pt",
) -> dict:
    """
    Train AC-GAN 2-digit with curriculum (from pre-trained single-digit).

    Tests: Does curriculum + hybrid training achieve best performance?

    Args:
        config: Training configuration
        single_digit_checkpoint: Path to pre-trained single-digit AC-GAN generator
        judge_checkpoint: Path to pre-trained 2-digit Judge

    Returns:
        Final metrics (test at step 0 to evaluate immediate transfer)
    """
    from src.models.acgan_twodigit import ACGANGenerator
    from src.models.judge_twodigit import create_twodigit_judge
    from src.data.multidigit import get_twodigit_loader

    device = config.get_device()
    logger.info(f"Training AC-GAN 2-digit (curriculum) on {device}")
    logger.info("CURRICULUM: Using pre-trained single-digit AC-GAN generator")

    # Set reproducibility
    set_reproducibility(config.seed)

    # Data
    train_loader = get_twodigit_loader(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        train=True,
        num_samples=1000,  # Just for evaluation at step 0
    )

    # Load pre-trained single-digit AC-GAN generator
    logger.info(f"Loading single-digit AC-GAN from {single_digit_checkpoint}")
    checkpoint = torch.load(single_digit_checkpoint, map_location=device, weights_only=False)

    single_digit_gen = checkpoint["models"]["generator"]

    # Create 2-digit generator using pre-trained single-digit generator
    # This is composition - use two copies of the single-digit generator
    generator = ACGANGenerator(latent_dim=config.latent_dim, num_classes=100)

    # Load weights from single-digit generator for both positions
    # Note: This is a simple composition - in practice may need architecture adaptation
    logger.info("Composing two single-digit generators for 2-digit generation")

    # Create Judge for evaluation
    judge = create_twodigit_judge(
        checkpoint_path=judge_checkpoint,
        device=device,
        freeze=True,
    )

    generator = generator.to(device)
    generator.eval()

    # Evaluate immediately at step 0 (curriculum transfer test)
    logger.info("Evaluating curriculum transfer at step 0...")

    with torch.no_grad():
        num_samples = 1000
        z = torch.randn(num_samples, config.latent_dim, device=device)
        labels = torch.randint(0, 100, (num_samples,), device=device)

        images = generator(z, labels)

        judge_logits = judge(images, mode="full")
        judge_acc = (judge_logits.argmax(dim=1) == labels).float().mean().item()

        judge_tens, judge_ones = judge(images, mode="per_position")
        tens_labels = labels // 10
        ones_labels = labels % 10
        tens_acc = (judge_tens.argmax(dim=1) == tens_labels).float().mean().item()
        ones_acc = (judge_ones.argmax(dim=1) == ones_labels).float().mean().item()

    final_metrics = {
        "eval/judge_accuracy": judge_acc,
        "eval/tens_accuracy": tens_acc,
        "eval/ones_accuracy": ones_acc,
    }

    logger.info("\nCURRICULUM TRANSFER EVALUATION (Step 0)")
    logger.info("=" * 50)
    for key, value in final_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    logger.info("=" * 50)

    return final_metrics


def train_acgan2_direct(
    config: TrainingConfig,
    judge_checkpoint: str = "checkpoints/judge_twodigit.pt",
) -> dict:
    """
    Train AC-GAN 2-digit (hybrid adversarial + pedagogical).

    Tests: Does adding pedagogical class loss help adversarial training?

    Args:
        config: Training configuration
        judge_checkpoint: Path to pre-trained 2-digit Judge

    Returns:
        Final metrics
    """
    from src.models.acgan_twodigit import create_acgan_twodigit
    from src.models.judge_twodigit import create_twodigit_judge
    from src.data.multidigit import get_twodigit_loader
    from src.training.acgan_trainer_twodigit import ACGANTrainerTwoDigit

    device = config.get_device()
    logger.info(f"Training AC-GAN 2-digit (hybrid) on {device}")
    logger.info("HYBRID: Adversarial + Pedagogical (class prediction)")

    # Set reproducibility
    set_reproducibility(config.seed)

    # Data
    train_loader = get_twodigit_loader(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        train=True,
        num_samples=50000,
    )

    # Models
    generator, discriminator = create_acgan_twodigit(
        latent_dim=config.latent_dim,
        device=device,
    )
    judge = create_twodigit_judge(
        checkpoint_path=judge_checkpoint,
        device=device,
        freeze=True,
    )

    logger.info(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    logger.info(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Trainer
    trainer = ACGANTrainerTwoDigit(
        config=config,
        generator=generator,
        discriminator=discriminator,
        judge=judge,
        train_loader=train_loader,
        device=device,
    )

    # Train
    final_metrics = trainer.train()

    return final_metrics


def train_gan2_direct(
    config: TrainingConfig,
    judge_checkpoint: str = "checkpoints/judge_twodigit.pt",
) -> dict:
    """
    Train 2-digit GAN from scratch (no curriculum).

    Critical ablation: Does adversarial training succeed where
    pedagogical training failed?

    Args:
        config: Training configuration
        judge_checkpoint: Path to pre-trained 2-digit Judge

    Returns:
        Final metrics
    """
    from src.models.gan_twodigit import create_twodigit_gan
    from src.models.judge_twodigit import create_twodigit_judge
    from src.data.multidigit import get_twodigit_loader
    from src.training.gan_trainer_twodigit import GANTrainerTwoDigit

    device = config.get_device()
    logger.info(f"Training GAN 2-digit (from scratch) on {device}")
    logger.info("ABLATION: GAN baseline for compositional learning")

    # Set reproducibility
    set_reproducibility(config.seed)

    # Data
    train_loader = get_twodigit_loader(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        train=True,
        num_samples=50000,
    )

    # Models
    generator, discriminator = create_twodigit_gan(
        latent_dim=config.latent_dim,
        device=device,
    )
    judge = create_twodigit_judge(
        checkpoint_path=judge_checkpoint,
        device=device,
        freeze=True,
    )

    logger.info(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    logger.info(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Trainer
    trainer = GANTrainerTwoDigit(
        config=config,
        generator=generator,
        discriminator=discriminator,
        judge=judge,
        train_loader=train_loader,
        device=device,
    )

    # Train
    final_metrics = trainer.train()

    return final_metrics


def train_relation_judge(
    config: TrainingConfig,
    num_train_steps: int = 5000,
) -> dict:
    """
    Train RelationJudge to validate X > Y constraints.

    Pre-training phase for Phase 1.5 relational task.

    Args:
        config: Training configuration
        num_train_steps: Number of training steps

    Returns:
        Final metrics
    """
    import torch.nn.functional as F
    from src.data.relational_mnist import get_relational_mnist_loader
    from src.models.relation_judge import RelationJudge

    device = config.get_device()
    logger.info(f"Training RelationJudge on {device}")

    # Set reproducibility
    set_reproducibility(config.seed)

    # Data: train on ALL digits [0,9] so judge can classify any digit
    # Only the *relations* are curriculum-limited (generator trains on [0,4])
    train_loader = get_relational_mnist_loader(
        digit_range=(0, 9),
        batch_size=config.data.batch_size,
        train=True,
        size=20000,  # More relations with full range
    )
    val_loader = get_relational_mnist_loader(
        digit_range=(5, 9),
        batch_size=config.data.batch_size,
        train=False,
        size=2000,
    )

    # Model
    judge = RelationJudge().to(device)
    optimizer = torch.optim.Adam(judge.parameters(), lr=0.001)

    logger.info(f"Judge params: {sum(p.numel() for p in judge.parameters()):,}")

    # Training loop
    step = 0
    epoch = 0

    while step < num_train_steps:
        epoch += 1
        judge.train()

        for images, x_labels, y_labels in train_loader:
            images = images.to(device)
            x_labels = x_labels.to(device)
            y_labels = y_labels.to(device)

            x_logits, y_logits, valid_prob = judge(images)

            # Loss: digit classification + relation validity
            x_loss = F.cross_entropy(x_logits, x_labels)
            y_loss = F.cross_entropy(y_logits, y_labels)

            # Relation loss: maximize P(X > Y) when X > Y
            valid_gt = (x_labels > y_labels).float()
            # Clamp valid_prob to [0,1] for numerical stability
            valid_prob_clamped = valid_prob.clamp(0.0, 1.0)
            relation_loss = F.binary_cross_entropy(
                valid_prob_clamped,
                valid_gt,
            )

            total_loss = x_loss + y_loss + relation_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % config.logging.log_interval == 0:
                metrics = judge.compute_accuracy(images, x_labels, y_labels)
                logger.info(
                    f"Step {step} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"X Acc: {metrics['x_accuracy']:.1%} | "
                    f"Y Acc: {metrics['y_accuracy']:.1%} | "
                    f"Rel Acc: {metrics['relation_accuracy']:.1%}"
                )

            step += 1
            if step >= num_train_steps:
                break

    # Final validation on transfer set
    judge.eval()
    all_metrics = []
    with torch.no_grad():
        for images, x_labels, y_labels in val_loader:
            images = images.to(device)
            x_labels = x_labels.to(device)
            y_labels = y_labels.to(device)
            metrics = judge.compute_accuracy(images, x_labels, y_labels)
            all_metrics.append(metrics)

    # Average metrics
    final_metrics = {
        k: sum(m[k] for m in all_metrics) / len(all_metrics)
        for k in all_metrics[0].keys()
    }

    logger.info("TRANSFER TEST [5,9]:")
    for key, value in final_metrics.items():
        logger.info(f"  {key}: {value:.1%}")

    # Save checkpoint
    checkpoint_path = "checkpoints/relation_judge.pt"
    torch.save({
        "model": judge.state_dict(),
        "step": step,
        "metrics": final_metrics,
    }, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")

    return final_metrics


def train_relational(
    config: TrainingConfig,
    single_digit_checkpoint: str = "checkpoints/checkpoint_final.pt",
    judge_checkpoint: str = "checkpoints/relation_judge.pt",
) -> dict:
    """
    Train RelationalWeaver on X > Y task (curriculum: [0,4]).

    Phase 1.5: Tests whether pedagogical training can learn
    context-dependent representations.

    Args:
        config: Training configuration
        single_digit_checkpoint: Path to single-digit Weaver
        judge_checkpoint: Path to pre-trained RelationJudge

    Returns:
        Final metrics
    """
    from src.training.relational_trainer import create_relational_trainer

    device = config.get_device()
    logger.info(f"Training RelationalWeaver on {device}")
    logger.info("Curriculum: [0,4] → Test transfer to [5,9]")

    # Set reproducibility
    set_reproducibility(config.seed)

    # Trainer
    trainer = create_relational_trainer(
        single_digit_checkpoint=single_digit_checkpoint,
        judge_checkpoint=judge_checkpoint,
        latent_dim=config.latent_dim,
        num_relations=10,  # Valid relations for [0,4]
        device=device,
        freeze_digits=True,  # CRITICAL: Preserve digit primitives for valid test
    )

    logger.info(f"Weaver params: {sum(p.numel() for p in trainer.weaver.parameters()):,}")

    # Training loop
    total_steps = config.total_steps
    for step in range(total_steps):
        metrics = trainer.train_step(batch_size=config.data.batch_size)

        if step % config.logging.log_interval == 0:
            logger.info(
                f"Step {step} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Valid: {metrics['avg_validity']:.1%} | "
                f"X Acc: {metrics['x_accuracy']:.1%} | "
                f"Y Acc: {metrics['y_accuracy']:.1%}"
            )

        if step % config.checkpointing.save_interval == 0 and step > 0:
            checkpoint_path = f"checkpoints/relational_step{step}.pt"
            trainer.save_checkpoint(checkpoint_path, step, metrics)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Final checkpoint
    final_path = "checkpoints/relational_final.pt"
    trainer.save_checkpoint(final_path, total_steps, metrics)
    logger.info(f"Saved final checkpoint: {final_path}")

    # Curriculum evaluation
    logger.info("\nEVALUATING ON CURRICULUM [0,4]...")
    curriculum_metrics = trainer.evaluate(
        num_samples=1000,
        curriculum_mode=True,
    )
    for key, value in curriculum_metrics.items():
        logger.info(f"  {key}: {value:.1%}")

    # Transfer evaluation
    logger.info("\nEVALUATING ON TRANSFER [5,9]...")
    transfer_metrics = trainer.evaluate(
        num_samples=1000,
        curriculum_mode=False,
    )
    for key, value in transfer_metrics.items():
        logger.info(f"  {key}: {value:.1%}")

    # Combined metrics
    final_metrics = {
        "curriculum_validity": curriculum_metrics["avg_validity"],
        "curriculum_relation_acc": curriculum_metrics["relation_accuracy"],
        "transfer_validity": transfer_metrics["avg_validity"],
        "transfer_relation_acc": transfer_metrics["relation_accuracy"],
    }

    return final_metrics


def train_relational_holdout(
    config: TrainingConfig,
    single_digit_checkpoint: str = "checkpoints/checkpoint_final.pt",
    judge_checkpoint: str = "checkpoints/relation_judge.pt",
) -> dict:
    """
    Train RelationalWeaver with hold-out pairs (Phase 1.6).

    Tests whether compositional learning is possible when training on
    all digits [0-9] but holding out specific relation pairs.

    Hold-out pairs: {7>3, 8>2, 9>1, 6>4}
    Training pairs: 41 of 45 valid X > Y pairs

    Args:
        config: Training configuration
        single_digit_checkpoint: Path to single-digit Weaver
        judge_checkpoint: Path to pre-trained RelationJudge

    Returns:
        Final metrics
    """
    from src.training.relational_trainer import create_relational_trainer_holdout

    device = config.get_device()
    logger.info(f"Training RelationalWeaver (Phase 1.6 - Hold-out pairs) on {device}")
    logger.info("Training on all digits [0-9]")
    logger.info("Hold-out pairs: {7>3, 8>2, 9>1, 6>4}")

    # Set reproducibility
    set_reproducibility(config.seed)

    # Define hold-out pairs for Phase 1.6
    holdout_pairs = [(7, 3), (8, 2), (9, 1), (6, 4)]

    # Trainer with hold-out pairs
    trainer = create_relational_trainer_holdout(
        single_digit_checkpoint=single_digit_checkpoint,
        judge_checkpoint=judge_checkpoint,
        latent_dim=config.latent_dim,
        holdout_pairs=holdout_pairs,
        device=device,
        freeze_digits=True,
    )

    logger.info(f"Weaver params: {sum(p.numel() for p in trainer.weaver.parameters()):,}")
    logger.info(f"Training pairs: 41 (45 - 4 hold-out)")

    # Training loop
    total_steps = config.total_steps
    for step in range(total_steps):
        metrics = trainer.train_step(batch_size=config.data.batch_size)

        if step % config.logging.log_interval == 0:
            logger.info(
                f"Step {step} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Valid: {metrics['avg_validity']:.1%} | "
                f"X Acc: {metrics['x_accuracy']:.1%} | "
                f"Y Acc: {metrics['y_accuracy']:.1%}"
            )

        if step % config.checkpointing.save_interval == 0 and step > 0:
            checkpoint_path = f"checkpoints/relational_holdout_step{step}.pt"
            trainer.save_checkpoint(checkpoint_path, step, metrics)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Final checkpoint
    final_path = "checkpoints/relational_holdout_final.pt"
    trainer.save_checkpoint(final_path, total_steps, metrics)
    logger.info(f"Saved final checkpoint: {final_path}")

    # Training pairs evaluation
    logger.info("\nEVALUATING ON TRAINING PAIRS (41 pairs)...")
    training_metrics = trainer.evaluate(
        num_samples=1000,
        holdout_mode=False,
    )
    for key, value in training_metrics.items():
        logger.info(f"  {key}: {value:.1%}")

    # Hold-out pairs evaluation
    logger.info("\nEVALUATING ON HOLD-OUT PAIRS {7>3, 8>2, 9>1, 6>4}...")
    holdout_metrics = trainer.evaluate(
        num_samples=1000,
        holdout_mode=True,
    )
    for key, value in holdout_metrics.items():
        logger.info(f"  {key}: {value:.1%}")

    # Combined metrics
    final_metrics = {
        "training_validity": training_metrics["avg_validity"],
        "training_relation_acc": training_metrics["relation_accuracy"],
        "holdout_validity": holdout_metrics["avg_validity"],
        "holdout_relation_acc": holdout_metrics["relation_accuracy"],
    }

    return final_metrics


def train_relational_holdout_acgan(
    config: TrainingConfig,
    acgan_checkpoint: str = "checkpoints/acgan_step1000.pt",  # Peak checkpoint
    judge_checkpoint: str = "checkpoints/relation_judge.pt",
) -> dict:
    """
    Train RelationalWeaver with AC-GAN primitives (Phase 1.6 baseline).

    Critical control experiment: Does primitive quality matter?
    - GPN (blobby): 100% holdout transfer
    - AC-GAN (sharp): ??? holdout transfer

    If quality matters: AC-GAN should outperform GPN
    If coverage matters: Both should perform equally (~100%)

    Args:
        config: Training configuration
        acgan_checkpoint: Path to AC-GAN checkpoint (step 1000 peak)
        judge_checkpoint: Path to pre-trained RelationJudge

    Returns:
        Final metrics
    """
    from src.training.relational_trainer import create_relational_trainer_holdout_acgan

    device = config.get_device()
    logger.info(f"Training RelationalWeaver with AC-GAN primitives (Phase 1.6 baseline) on {device}")
    logger.info("Testing: Does primitive quality (sharp vs blobby) matter for compositional transfer?")
    logger.info("Hold-out pairs: {7>3, 8>2, 9>1, 6>4}")

    # Set reproducibility
    set_reproducibility(config.seed)

    # Define hold-out pairs
    holdout_pairs = [(7, 3), (8, 2), (9, 1), (6, 4)]

    # Trainer with AC-GAN primitives (latent_dim auto-detected from checkpoint)
    trainer = create_relational_trainer_holdout_acgan(
        acgan_checkpoint=acgan_checkpoint,
        judge_checkpoint=judge_checkpoint,
        latent_dim=None,  # Auto-detect from checkpoint
        holdout_pairs=holdout_pairs,
        device=device,
        freeze_digits=True,
    )

    logger.info(f"Weaver params: {sum(p.numel() for p in trainer.weaver.parameters()):,}")
    logger.info(f"Latent dim: {trainer.latent_dim}")
    logger.info(f"AC-GAN checkpoint: {acgan_checkpoint}")
    logger.info(f"Training pairs: 41 (45 - 4 hold-out)")

    # Training loop
    total_steps = config.total_steps
    for step in range(total_steps):
        metrics = trainer.train_step(batch_size=config.data.batch_size)

        if step % config.logging.log_interval == 0:
            logger.info(
                f"Step {step} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Valid: {metrics['avg_validity']:.1%} | "
                f"X Acc: {metrics['x_accuracy']:.1%} | "
                f"Y Acc: {metrics['y_accuracy']:.1%}"
            )

        if step % config.checkpointing.save_interval == 0 and step > 0:
            checkpoint_path = f"checkpoints/relational_holdout_acgan_step{step}.pt"
            trainer.save_checkpoint(checkpoint_path, step, metrics)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Final checkpoint
    final_path = "checkpoints/relational_holdout_acgan_final.pt"
    trainer.save_checkpoint(final_path, total_steps, metrics)
    logger.info(f"Saved final checkpoint: {final_path}")

    # Training pairs evaluation
    logger.info("\nEVALUATING ON TRAINING PAIRS (41 pairs)...")
    training_metrics = trainer.evaluate(
        num_samples=1000,
        holdout_mode=False,
    )
    for key, value in training_metrics.items():
        logger.info(f"  {key}: {value:.1%}")

    # Hold-out pairs evaluation (THE CRITICAL TEST)
    logger.info("\nEVALUATING ON HOLD-OUT PAIRS {7>3, 8>2, 9>1, 6>4}...")
    logger.info("Question: Does sharp AC-GAN match blobby GPN's 100% transfer?")
    holdout_metrics = trainer.evaluate(
        num_samples=1000,
        holdout_mode=True,
    )
    for key, value in holdout_metrics.items():
        logger.info(f"  {key}: {value:.1%}")

    # Combined metrics
    final_metrics = {
        "training_validity": training_metrics["avg_validity"],
        "training_relation_acc": training_metrics["relation_accuracy"],
        "holdout_validity": holdout_metrics["avg_validity"],
        "holdout_relation_acc": holdout_metrics["relation_accuracy"],
    }

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
        choices=["gpn", "gpn-v2", "gpn-v3", "gpn-v3-nometa", "gpn2", "gpn2-direct", "gan", "gan2-direct", "acgan", "acgan2", "acgan2-direct", "relation-judge", "relational", "relational-holdout", "relational-holdout-acgan"],
        default="gpn",
        help="Training mode: gpn (original), gpn-v2 (grounded witness), gpn-v3 (meta-learning), gpn-v3-nometa (ablation: no inner loop), gpn2 (two-digit curriculum), gpn2-direct (two-digit from scratch, no curriculum ablation), gan (baseline), relation-judge (Phase 1.5 judge), relational (Phase 1.5 X>Y task), relational-holdout (Phase 1.6 hold-out pairs with GPN), relational-holdout-acgan (Phase 1.6 with AC-GAN baseline)",
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
    parser.add_argument(
        "--competence-threshold",
        type=float,
        default=0.5,
        help="V3 only: Witness competence threshold before Weaver training starts (0.0 to disable gating)",
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
    elif args.mode == "gpn-v2":
        metrics = train_gpn_v2(config, resume_from=args.resume)
    elif args.mode == "gpn-v3":
        metrics = train_gpn_v3(config, resume_from=args.resume, competence_threshold=args.competence_threshold)
    elif args.mode == "gpn-v3-nometa":
        metrics = train_gpn_v3_no_meta(config, resume_from=args.resume, competence_threshold=args.competence_threshold)
    elif args.mode == "gpn2":
        metrics = train_gpn2(config, resume_from=args.resume)
    elif args.mode == "gpn2-direct":
        metrics = train_gpn2_direct(config, resume_from=args.resume)
    elif args.mode == "gan2-direct":
        metrics = train_gan2_direct(config)
    elif args.mode == "acgan":
        metrics = train_acgan(config)
    elif args.mode == "acgan2":
        metrics = train_acgan2(config)
    elif args.mode == "acgan2-direct":
        metrics = train_acgan2_direct(config)
    elif args.mode == "relation-judge":
        metrics = train_relation_judge(config)
    elif args.mode == "relational":
        metrics = train_relational(config)
    elif args.mode == "relational-holdout":
        metrics = train_relational_holdout(config)
    elif args.mode == "relational-holdout-acgan":
        metrics = train_relational_holdout_acgan(config)
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
