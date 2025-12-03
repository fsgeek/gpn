"""
GAN Trainer for baseline adversarial training.

Implements standard GAN training loop for comparison with GPN.

Exports:
    - GANTrainer: Standard adversarial training loop
"""

from pathlib import Path
from typing import Optional, Iterator
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.baseline_gan import Generator, Discriminator
from src.models.judge import Judge
from src.training.config import TrainingConfig
from src.metrics.quality import QualityMetrics
from src.metrics.mode_diversity import ModeDiversity
from src.metrics.convergence import ConvergenceMetrics
from src.utils.logging import MetricsLogger
from src.utils.checkpointing import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class GANTrainer:
    """
    Standard GAN training loop for baseline comparison.

    Uses adversarial training (generator vs discriminator) without
    the cooperative signaling of GPN.
    """

    def __init__(
        self,
        config: TrainingConfig,
        generator: Generator,
        discriminator: Discriminator,
        judge: Judge,
        train_loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize GAN trainer.

        Args:
            config: Training configuration
            generator: Generator model
            discriminator: Discriminator model
            judge: Frozen Judge for quality evaluation
            train_loader: Training data loader
            device: Training device
        """
        self.config = config
        self.device = device or config.get_device()

        # Models
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.judge = judge.to(self.device)
        self.judge.freeze()

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

        # Metrics
        self.quality_metrics = QualityMetrics(self.judge)
        self.mode_diversity = ModeDiversity(self.judge)
        self.convergence_metrics = ConvergenceMetrics()

        # Logging
        self.logger: Optional[MetricsLogger] = None

        # Training state
        self.current_step = 0
        self._data_iterator: Optional[Iterator] = None

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
        Execute a single GAN training step.

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

        # Generate images and try to fool discriminator
        fake_output = self.discriminator(fake_images, labels)
        g_loss = self.adversarial_loss(fake_output, real_label)

        g_loss.backward()
        self.g_optimizer.step()

        # Compute quality metric
        with torch.no_grad():
            judge_logits = self.judge(fake_images)
            quality = (judge_logits.argmax(dim=1) == labels).float().mean().item()

        # Update convergence tracking
        self.convergence_metrics.update(self.current_step, quality)

        return {
            "loss/generator": g_loss.item(),
            "loss/discriminator": d_loss.item(),
            "loss/d_real": d_real_loss.item(),
            "loss/d_fake": d_fake_loss.item(),
            "quality/judge_accuracy": quality,
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
            Final metrics dictionary
        """
        total_steps = total_steps or self.config.total_steps
        log_dir = log_dir or self.config.logging.log_dir

        # Setup logging
        self.logger = MetricsLogger(log_dir, experiment_name="gan_baseline")

        logger.info(f"Starting GAN training for {total_steps} steps")

        final_metrics = {}

        while self.current_step < total_steps:
            metrics = self.train_step()
            final_metrics = metrics

            # Logging
            if self.current_step % self.config.logging.log_interval == 0:
                self._log_metrics(metrics)

            # Sample generation
            if self.current_step % self.config.logging.sample_interval == 0:
                self._log_samples()

            # Checkpointing
            if self.current_step % self.config.checkpointing.save_interval == 0:
                self._save_checkpoint()

            self.current_step += 1

        # Final checkpoint
        self._save_checkpoint("gan_final")

        self.logger.close()

        logger.info("GAN training complete")
        return final_metrics

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to TensorBoard."""
        if self.logger is None:
            return

        for name, value in metrics.items():
            self.logger.log_scalar(name, value, self.current_step)

    def _log_samples(self) -> None:
        """Generate and log sample images."""
        if self.logger is None:
            return

        self.generator.eval()

        with torch.no_grad():
            num_samples = self.config.logging.num_samples
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
            labels = torch.arange(10, device=self.device).repeat(num_samples // 10 + 1)[:num_samples]

            images = self.generator(z, labels)
            images = (images + 1) / 2

            self.logger.log_images(
                "samples/generated",
                images,
                self.current_step,
                nrow=10,
            )

        self.generator.train()

    def _save_checkpoint(self, suffix: Optional[str] = None) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpointing.checkpoint_dir)

        if suffix:
            path = checkpoint_dir / f"gan_checkpoint_{suffix}.pt"
        else:
            path = checkpoint_dir / f"gan_checkpoint_step{self.current_step}.pt"

        save_checkpoint(
            path=path,
            step=self.current_step,
            phase=0,  # GAN has no phases
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

    def evaluate(self, num_samples: int = 1000) -> dict[str, float]:
        """
        Run evaluation on generated samples.

        Args:
            num_samples: Number of samples to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        self.generator.eval()

        with torch.no_grad():
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
            labels = torch.randint(0, 10, (num_samples,), device=self.device)

            images = self.generator(z, labels)

            quality = self.quality_metrics.evaluate(images, labels)
            diversity = self.mode_diversity.evaluate(images, labels)

        self.generator.train()

        return {
            "eval/judge_accuracy": quality.judge_accuracy,
            "eval/mode_coverage": diversity.mode_coverage,
            "eval/entropy": diversity.entropy,
            "eval/is_collapsed": float(diversity.is_collapsed),
        }
