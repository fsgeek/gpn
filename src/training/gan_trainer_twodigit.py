"""
Two-digit GAN Trainer for GPN-2 baseline comparison.

Tests whether adversarial training can learn compositional
2-digit generation from scratch (no curriculum).
"""

from pathlib import Path
from typing import Optional, Iterator
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.gan_twodigit import TwoDigitGenerator, TwoDigitDiscriminator
from src.models.judge_twodigit import TwoDigitJudge
from src.training.config import TrainingConfig
from src.utils.logging import MetricsLogger
from src.utils.checkpointing import save_checkpoint

logger = logging.getLogger(__name__)


class GANTrainerTwoDigit:
    """
    GAN trainer for 2-digit MNIST.

    Critical ablation: Does adversarial training succeed where
    pedagogical training failed (from-scratch compositional learning)?
    """

    def __init__(
        self,
        config: TrainingConfig,
        generator: TwoDigitGenerator,
        discriminator: TwoDigitDiscriminator,
        judge: TwoDigitJudge,
        train_loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize trainer.

        Args:
            config: Training configuration
            generator: Generator model
            discriminator: Discriminator model
            judge: Frozen Judge for evaluation
            train_loader: Training data
            device: Training device
        """
        self.config = config
        self.device = device or config.get_device()

        # Models
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.judge = judge.to(self.device)

        # Data
        self.train_loader = train_loader

        # Optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
        )

        # Loss
        self.adversarial_loss = nn.BCEWithLogitsLoss()

        # Logging
        self.logger: Optional[MetricsLogger] = None

        # Training state
        self.current_step = 0
        self._data_iterator: Optional[Iterator] = None

        logger.info("GANTrainerTwoDigit initialized")
        logger.info(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
        logger.info(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get next batch from data loader."""
        if self._data_iterator is None:
            self._data_iterator = iter(self.train_loader)

        try:
            images, labels = next(self._data_iterator)
        except StopIteration:
            self._data_iterator = iter(self.train_loader)
            images, labels = next(self._data_iterator)

        return images.to(self.device), labels.to(self.device)

    def train_step(self) -> dict[str, float]:
        """
        Execute single GAN training step.

        Returns:
            Dictionary of metrics
        """
        self.generator.train()
        self.discriminator.train()

        # Get real data
        real_images, labels = self._get_batch()
        batch_size = real_images.size(0)

        # Labels for adversarial training
        real_label = torch.ones(batch_size, 1, device=self.device)
        fake_label = torch.zeros(batch_size, 1, device=self.device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        self.d_optimizer.zero_grad()

        # Real images
        real_output = self.discriminator(real_images, labels)
        d_real_loss = self.adversarial_loss(real_output, real_label)

        # Fake images
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        fake_images = self.generator(z, labels)
        fake_output = self.discriminator(fake_images.detach(), labels)
        d_fake_loss = self.adversarial_loss(fake_output, fake_label)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.d_optimizer.step()

        # ---------------------
        # Train Generator
        # ---------------------
        self.g_optimizer.zero_grad()

        # Generate and try to fool discriminator
        fake_output = self.discriminator(fake_images, labels)
        g_loss = self.adversarial_loss(fake_output, real_label)

        g_loss.backward()
        self.g_optimizer.step()

        # Evaluate with Judge
        with torch.no_grad():
            self.judge.eval()
            judge_logits = self.judge(fake_images, mode="full")
            judge_acc = (judge_logits.argmax(dim=1) == labels).float().mean().item()

            # Also evaluate per-position accuracy
            judge_tens, judge_ones = self.judge(fake_images, mode="per_position")
            tens_labels = labels // 10
            ones_labels = labels % 10
            tens_acc = (judge_tens.argmax(dim=1) == tens_labels).float().mean().item()
            ones_acc = (judge_ones.argmax(dim=1) == ones_labels).float().mean().item()

        return {
            "loss/generator": g_loss.item(),
            "loss/discriminator": d_loss.item(),
            "loss/d_real": d_real_loss.item(),
            "loss/d_fake": d_fake_loss.item(),
            "quality/judge_accuracy": judge_acc,
            "quality/tens_accuracy": tens_acc,
            "quality/ones_accuracy": ones_acc,
        }

    def train(
        self,
        total_steps: Optional[int] = None,
        log_dir: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Run full training loop.

        Args:
            total_steps: Total training steps
            log_dir: Logging directory

        Returns:
            Final metrics
        """
        total_steps = total_steps or self.config.total_steps
        log_dir = log_dir or self.config.logging.log_dir

        # Setup logging
        self.logger = MetricsLogger(log_dir, experiment_name="gan_twodigit")

        logger.info(f"Starting GAN 2-digit training for {total_steps} steps")
        logger.info("ABLATION: Training 2-digit from scratch, adversarial baseline")

        final_metrics = {}

        while self.current_step < total_steps:
            metrics = self.train_step()
            final_metrics = metrics

            # Logging
            if self.current_step % self.config.logging.log_interval == 0:
                self._log_metrics(metrics)
                logger.info(
                    f"Step {self.current_step} | "
                    f"Judge: {metrics['quality/judge_accuracy']:.3f} | "
                    f"Tens: {metrics['quality/tens_accuracy']:.3f} | "
                    f"Ones: {metrics['quality/ones_accuracy']:.3f}"
                )

            # Sample generation
            if self.current_step % self.config.logging.sample_interval == 0:
                self._log_samples()

            # Checkpointing
            if self.current_step % self.config.checkpointing.save_interval == 0:
                self._save_checkpoint()

            self.current_step += 1

        # Final checkpoint
        self._save_checkpoint("final")

        # Final evaluation
        eval_metrics = self.evaluate()
        final_metrics.update(eval_metrics)

        logger.info("\nTRAINING COMPLETE")
        logger.info("=" * 50)
        for key, value in eval_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("=" * 50)

        if self.logger:
            self.logger.close()

        return final_metrics

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to TensorBoard."""
        if self.logger is None:
            return

        for name, value in metrics.items():
            self.logger.log_scalar(name, value, self.current_step)
        self.logger.flush()

    def _log_samples(self) -> None:
        """Generate and log sample images."""
        if self.logger is None:
            return

        self.generator.eval()

        with torch.no_grad():
            num_samples = 64
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)

            # Sample across label space
            labels = torch.randint(0, 100, (num_samples,), device=self.device)

            images = self.generator(z, labels)
            images = (images + 1) / 2  # [-1,1] -> [0,1]

            self.logger.log_images(
                "samples/generated",
                images,
                self.current_step,
                nrow=8,
            )

        self.generator.train()

    def _save_checkpoint(self, suffix: Optional[str] = None) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpointing.checkpoint_dir)

        if suffix:
            path = checkpoint_dir / f"gan_twodigit_{suffix}.pt"
        else:
            path = checkpoint_dir / f"gan_twodigit_step{self.current_step}.pt"

        save_checkpoint(
            path=path,
            step=self.current_step,
            phase=0,
            models={
                "generator": self.generator,
                "discriminator": self.discriminator,
            },
            optimizers={
                "generator": self.g_optimizer,
                "discriminator": self.d_optimizer,
            },
            config=self.config.to_dict(),
        )

        logger.info(f"Saved checkpoint to {path}")

    def evaluate(self, num_samples: int = 1000) -> dict[str, float]:
        """
        Evaluate generated samples.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Evaluation metrics
        """
        self.generator.eval()
        self.judge.eval()

        with torch.no_grad():
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
            labels = torch.randint(0, 100, (num_samples,), device=self.device)

            images = self.generator(z, labels)

            # Full number accuracy
            judge_logits = self.judge(images, mode="full")
            judge_acc = (judge_logits.argmax(dim=1) == labels).float().mean().item()

            # Per-position accuracy
            judge_tens, judge_ones = self.judge(images, mode="per_position")
            tens_labels = labels // 10
            ones_labels = labels % 10
            tens_acc = (judge_tens.argmax(dim=1) == tens_labels).float().mean().item()
            ones_acc = (judge_ones.argmax(dim=1) == ones_labels).float().mean().item()

        self.generator.train()

        return {
            "eval/judge_accuracy": judge_acc,
            "eval/tens_accuracy": tens_acc,
            "eval/ones_accuracy": ones_acc,
        }
