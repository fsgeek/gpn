"""
AC-GAN Trainer for 2-digit MNIST.

Hybrid adversarial + pedagogical training:
- Discriminator learns real/fake AND class prediction
- Generator optimized to fool discriminator AND produce correct class

Tests: Does pedagogical signal help adversarial training?
"""

from pathlib import Path
from typing import Optional, Iterator
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.acgan_twodigit import ACGANGenerator, ACGANDiscriminator
from src.models.judge_twodigit import TwoDigitJudge
from src.training.config import TrainingConfig
from src.utils.logging import MetricsLogger
from src.utils.checkpointing import save_checkpoint

logger = logging.getLogger(__name__)


class ACGANTrainerTwoDigit:
    """
    AC-GAN trainer for 2-digit MNIST.

    Key difference from standard GAN:
    - Discriminator has two objectives: adversarial + class prediction
    - Generator optimized against both
    - This adds pedagogical guidance to adversarial training
    """

    def __init__(
        self,
        config: TrainingConfig,
        generator: ACGANGenerator,
        discriminator: ACGANDiscriminator,
        judge: TwoDigitJudge,
        train_loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> None:
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

        # Losses
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.class_loss = nn.CrossEntropyLoss()

        # Logging
        self.logger: Optional[MetricsLogger] = None

        # Training state
        self.current_step = 0
        self._data_iterator: Optional[Iterator] = None

        logger.info("ACGANTrainerTwoDigit initialized")
        logger.info(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
        logger.info(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._data_iterator is None:
            self._data_iterator = iter(self.train_loader)

        try:
            images, labels = next(self._data_iterator)
        except StopIteration:
            self._data_iterator = iter(self.train_loader)
            images, labels = next(self._data_iterator)

        return images.to(self.device), labels.to(self.device)

    def train_step(self) -> dict[str, float]:
        self.generator.train()
        self.discriminator.train()

        real_images, labels = self._get_batch()
        batch_size = real_images.size(0)

        real_label = torch.ones(batch_size, 1, device=self.device)
        fake_label = torch.zeros(batch_size, 1, device=self.device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        self.d_optimizer.zero_grad()

        # Real images - should be classified as real AND correct class
        d_real_adv, d_real_class = self.discriminator(real_images)
        d_real_loss_adv = self.adversarial_loss(d_real_adv, real_label)
        d_real_loss_class = self.class_loss(d_real_class, labels)
        d_real_loss = d_real_loss_adv + d_real_loss_class

        # Fake images - should be classified as fake (class loss still applies!)
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        fake_images = self.generator(z, labels)
        d_fake_adv, d_fake_class = self.discriminator(fake_images.detach())
        d_fake_loss_adv = self.adversarial_loss(d_fake_adv, fake_label)
        d_fake_loss_class = self.class_loss(d_fake_class, labels)
        d_fake_loss = d_fake_loss_adv + d_fake_loss_class

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.d_optimizer.step()

        # ---------------------
        # Train Generator
        # ---------------------
        self.g_optimizer.zero_grad()

        # Generator wants to fool discriminator (real) AND produce correct class
        g_adv, g_class = self.discriminator(fake_images)
        g_loss_adv = self.adversarial_loss(g_adv, real_label)
        g_loss_class = self.class_loss(g_class, labels)
        g_loss = g_loss_adv + g_loss_class

        g_loss.backward()
        self.g_optimizer.step()

        # Evaluate with Judge
        with torch.no_grad():
            self.judge.eval()
            judge_logits = self.judge(fake_images, mode="full")
            judge_acc = (judge_logits.argmax(dim=1) == labels).float().mean().item()

            judge_tens, judge_ones = self.judge(fake_images, mode="per_position")
            tens_labels = labels // 10
            ones_labels = labels % 10
            tens_acc = (judge_tens.argmax(dim=1) == tens_labels).float().mean().item()
            ones_acc = (judge_ones.argmax(dim=1) == ones_labels).float().mean().item()

            # Also check discriminator's class accuracy on fakes
            d_class_acc = (g_class.argmax(dim=1) == labels).float().mean().item()

        return {
            "loss/generator": g_loss.item(),
            "loss/g_adversarial": g_loss_adv.item(),
            "loss/g_class": g_loss_class.item(),
            "loss/discriminator": d_loss.item(),
            "loss/d_adversarial": (d_real_loss_adv + d_fake_loss_adv).item() / 2,
            "loss/d_class": (d_real_loss_class + d_fake_loss_class).item() / 2,
            "quality/judge_accuracy": judge_acc,
            "quality/tens_accuracy": tens_acc,
            "quality/ones_accuracy": ones_acc,
            "quality/discriminator_class_accuracy": d_class_acc,
        }

    def train(
        self,
        total_steps: Optional[int] = None,
        log_dir: Optional[str] = None,
    ) -> dict[str, float]:
        total_steps = total_steps or self.config.total_steps
        log_dir = log_dir or self.config.logging.log_dir

        self.logger = MetricsLogger(log_dir, experiment_name="acgan_twodigit")

        logger.info(f"Starting AC-GAN 2-digit training for {total_steps} steps")
        logger.info("HYBRID: Adversarial + Pedagogical (class prediction)")

        final_metrics = {}

        while self.current_step < total_steps:
            metrics = self.train_step()
            final_metrics = metrics

            if self.current_step % self.config.logging.log_interval == 0:
                self._log_metrics(metrics)
                logger.info(
                    f"Step {self.current_step} | "
                    f"Judge: {metrics['quality/judge_accuracy']:.3f} | "
                    f"Tens: {metrics['quality/tens_accuracy']:.3f} | "
                    f"Ones: {metrics['quality/ones_accuracy']:.3f} | "
                    f"D-Class: {metrics['quality/discriminator_class_accuracy']:.3f}"
                )

            if self.current_step % self.config.logging.sample_interval == 0:
                self._log_samples()

            if self.current_step % self.config.checkpointing.save_interval == 0:
                self._save_checkpoint()

            self.current_step += 1

        self._save_checkpoint("final")

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
        if self.logger is None:
            return

        for name, value in metrics.items():
            self.logger.log_scalar(name, value, self.current_step)
        self.logger.flush()

    def _log_samples(self) -> None:
        if self.logger is None:
            return

        self.generator.eval()

        with torch.no_grad():
            num_samples = 64
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
            labels = torch.randint(0, 100, (num_samples,), device=self.device)

            images = self.generator(z, labels)
            images = (images + 1) / 2

            self.logger.log_images(
                "samples/generated",
                images,
                self.current_step,
                nrow=8,
            )

        self.generator.train()

    def _save_checkpoint(self, suffix: Optional[str] = None) -> None:
        checkpoint_dir = Path(self.config.checkpointing.checkpoint_dir)

        if suffix:
            path = checkpoint_dir / f"acgan_twodigit_{suffix}.pt"
        else:
            path = checkpoint_dir / f"acgan_twodigit_step{self.current_step}.pt"

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
        self.generator.eval()
        self.judge.eval()

        with torch.no_grad():
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
            labels = torch.randint(0, 100, (num_samples,), device=self.device)

            images = self.generator(z, labels)

            judge_logits = self.judge(images, mode="full")
            judge_acc = (judge_logits.argmax(dim=1) == labels).float().mean().item()

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
