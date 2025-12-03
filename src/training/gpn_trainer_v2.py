"""
GPN Trainer V2: Grounded Witness Architecture.

Key changes from V1:
- Witness is grounded on real MNIST data (independent competence)
- Weaver learns from Witness's grounded judgment
- Phases now control the ratio of real vs synthetic training, not just loss weights

Pedagogical framing:
- Witness (student) studies the textbook (real MNIST)
- Witness develops independent judgment
- Weaver (teacher) learns to produce examples Witness can understand
- Trust emerges from demonstrated competence, not scheduling

Exports:
    - GPNTrainerV2: Grounded-witness GPN training loop
"""

from pathlib import Path
from typing import Optional, Iterator
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.weaver import Weaver
from src.models.witness import Witness
from src.models.judge import Judge
from src.training.config import TrainingConfig
from src.training.ema import EMAState
from src.metrics.quality import QualityMetrics, CollusionDetector
from src.metrics.mode_diversity import ModeDiversity
from src.utils.logging import MetricsLogger
from src.utils.checkpointing import save_checkpoint, load_checkpoint, find_latest_checkpoint

logger = logging.getLogger(__name__)


class GPNTrainerV2:
    """
    Grounded-witness GPN training loop.

    Key architectural change: Witness is trained on real MNIST to develop
    independent competence, then provides grounded feedback to Weaver.

    Training structure per step:
    1. Witness grounding: Train Witness on real MNIST (classification)
    2. Weaver training: Train Weaver to produce images Witness classifies correctly
    3. Quality monitoring: Track Judge accuracy (no training signal)

    Attributes:
        config: Training configuration
        weaver: Generator model
        witness: Classifier model (grounded on real data)
        judge: Frozen evaluation classifier
        device: Training device
    """

    def __init__(
        self,
        config: TrainingConfig,
        weaver: Weaver,
        witness: Witness,
        judge: Judge,
        train_loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize GPN trainer V2.

        Args:
            config: Training configuration
            weaver: Weaver (generator) model
            witness: Witness (classifier) model
            judge: Frozen Judge model for evaluation
            train_loader: Training data loader (real MNIST)
            device: Training device (default: from config)
        """
        self.config = config
        self.device = device or config.get_device()

        # Models
        self.weaver = weaver.to(self.device)
        self.witness = witness.to(self.device)
        self.judge = judge.to(self.device)
        self.judge.freeze()

        # Data
        self.train_loader = train_loader

        # Optimizers
        self.weaver_optimizer = self._create_optimizer(self.weaver)
        self.witness_optimizer = self._create_optimizer(self.witness)

        # EMA for Witness accuracy (tracks grounding quality)
        self.witness_accuracy_ema = 0.0
        self.witness_accuracy_ema_decay = 0.99

        # EMA state for v_seen tracking
        self.ema_state = EMAState(
            dim=config.weaver.v_pred_dim,
            decay=config.ema.decay,
            variance_threshold=config.ema.variance_threshold,
            window_size=config.ema.window_size,
            device=self.device,
        )

        # Metrics
        self.quality_metrics = QualityMetrics(self.judge, self.witness)
        self.mode_diversity = ModeDiversity(self.judge)
        self.collusion_detector = CollusionDetector(
            alignment_drop_threshold=config.collusion.alignment_drop_threshold,
            quality_stagnation_threshold=config.collusion.quality_stagnation_threshold,
        )

        # Logging
        self.logger: Optional[MetricsLogger] = None

        # Training state
        self.current_step = 0
        self._data_iterator: Optional[Iterator] = None

    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer for a model."""
        return optim.Adam(
            model.parameters(),
            lr=self.config.optimizer.lr,
            betas=self.config.optimizer.betas,
            weight_decay=self.config.optimizer.weight_decay,
        )

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

    def _get_phase(self) -> int:
        """Determine current phase based on step count."""
        if self.current_step < self.config.phase1_steps:
            return 1
        elif self.current_step < self.config.phase2_steps:
            return 2
        else:
            return 3

    def _get_grounding_ratio(self) -> float:
        """
        Get ratio of Witness grounding steps vs Weaver training steps.

        Early training: Heavy grounding (Witness studies textbook)
        Later training: More balanced (Witness evaluates teacher)
        Phase 3: No grounding (test if Witness maintains judgment)
        """
        phase = self._get_phase()
        if phase == 1:
            return 0.8  # 80% grounding, 20% Weaver training
        elif phase == 2:
            return 0.5  # 50/50
        else:
            return 0.0  # No grounding - drift test

    def train_step(self) -> dict[str, float]:
        """
        Execute a single training step.

        Structure:
        1. Witness grounding on real data (probabilistic based on phase)
        2. Weaver training with Witness feedback
        3. Quality monitoring via Judge

        Returns:
            Dictionary of metrics
        """
        phase = self._get_phase()
        is_drift_test = phase >= 3

        metrics = {}

        # Get real images and labels
        real_images, labels = self._get_batch()
        batch_size = real_images.size(0)

        # =====================================================================
        # Step 1: Witness Grounding (on real MNIST)
        # =====================================================================
        grounding_ratio = self._get_grounding_ratio()
        do_grounding = torch.rand(1).item() < grounding_ratio

        if do_grounding and not is_drift_test:
            self.witness.train()

            # Forward pass on real images
            witness_logits_real, _ = self.witness(real_images)

            # Classification loss on real data
            witness_grounding_loss = F.cross_entropy(witness_logits_real, labels)

            # Update Witness
            self.witness_optimizer.zero_grad()
            witness_grounding_loss.backward()
            self.witness_optimizer.step()

            # Track Witness accuracy on real data
            with torch.no_grad():
                witness_acc_real = (witness_logits_real.argmax(dim=1) == labels).float().mean().item()
                self.witness_accuracy_ema = (
                    self.witness_accuracy_ema_decay * self.witness_accuracy_ema
                    + (1 - self.witness_accuracy_ema_decay) * witness_acc_real
                )

            metrics["witness/grounding_loss"] = witness_grounding_loss.item()
            metrics["witness/accuracy_real"] = witness_acc_real
            metrics["witness/accuracy_ema"] = self.witness_accuracy_ema
        else:
            metrics["witness/grounding_loss"] = 0.0
            metrics["witness/accuracy_real"] = 0.0
            metrics["witness/accuracy_ema"] = self.witness_accuracy_ema

        # =====================================================================
        # Step 2: Weaver Training (Witness provides grounded feedback)
        # =====================================================================
        if not is_drift_test:
            self.weaver.train()
            self.witness.eval()  # Witness evaluates, doesn't learn from Weaver

            # Generate fake images
            z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
            fake_images, v_pred = self.weaver(z, labels)

            # Witness evaluates Weaver's output (gradients flow to Weaver, not Witness)
            witness_logits_fake, v_seen = self.witness(fake_images)

            # Weaver loss: Witness should correctly classify Weaver's output
            # This is the key: Witness has grounded knowledge, so this signal is meaningful
            weaver_loss = F.cross_entropy(witness_logits_fake, labels)

            # Optional: alignment loss (Weaver predicts what Witness will see)
            alignment_weight = self.config.phase1_weights.alignment if phase == 1 else \
                              self.config.phase2_weights.alignment if phase == 2 else 0.0
            if alignment_weight > 0:
                alignment_loss = F.mse_loss(v_pred, v_seen.detach())
                weaver_loss = weaver_loss + alignment_weight * alignment_loss
                metrics["loss/alignment"] = alignment_loss.item()
            else:
                metrics["loss/alignment"] = 0.0

            # Update Weaver only
            self.weaver_optimizer.zero_grad()
            weaver_loss.backward()
            self.weaver_optimizer.step()

            metrics["loss/weaver"] = weaver_loss.item()

            # Update EMA
            self.ema_state.update(v_seen.detach())
        else:
            # Phase 3: Observation only
            self.weaver.eval()
            self.witness.eval()

            with torch.no_grad():
                z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
                fake_images, v_pred = self.weaver(z, labels)
                witness_logits_fake, v_seen = self.witness(fake_images)

            metrics["loss/weaver"] = 0.0
            metrics["loss/alignment"] = 0.0

        # =====================================================================
        # Step 3: Quality Monitoring (Judge evaluates, no training signal)
        # =====================================================================
        with torch.no_grad():
            if is_drift_test:
                # Already have fake_images from above
                pass
            judge_logits = self.judge(fake_images)

            # Judge accuracy: Does Judge recognize the intended digit?
            judge_accuracy = (judge_logits.argmax(dim=1) == labels).float().mean().item()

            # Witness accuracy on fake: Does grounded Witness recognize it?
            witness_accuracy_fake = (witness_logits_fake.argmax(dim=1) == labels).float().mean().item()

            # Agreement: Do Judge and Witness agree?
            agreement = (judge_logits.argmax(dim=1) == witness_logits_fake.argmax(dim=1)).float().mean().item()

        metrics["quality/judge_accuracy"] = judge_accuracy
        metrics["quality/witness_accuracy_fake"] = witness_accuracy_fake
        metrics["quality/agreement"] = agreement
        metrics["training/phase"] = float(phase)
        metrics["ema/mean_norm"] = self.ema_state.mean.norm().item()
        metrics["ema/variance_mean"] = self.ema_state.variance.mean().item()

        return metrics

    def train(
        self,
        total_steps: Optional[int] = None,
        log_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Run full training loop.

        Args:
            total_steps: Total training steps (default: from config)
            log_dir: Logging directory (default: from config)
            resume_from: Path to checkpoint to resume from

        Returns:
            Final metrics dictionary
        """
        total_steps = total_steps or self.config.total_steps
        log_dir = log_dir or self.config.logging.log_dir

        # Setup logging
        self.logger = MetricsLogger(log_dir)

        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)

        logger.info(f"Starting GPN V2 training from step {self.current_step}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Device: {self.device}")
        logger.info("Architecture: Grounded Witness (Witness trained on real MNIST)")

        # Training loop
        final_metrics = {}
        last_phase = self._get_phase()

        while self.current_step < total_steps:
            # Check for phase transition
            current_phase = self._get_phase()
            if current_phase != last_phase:
                logger.info(f"Phase transition: {last_phase} -> {current_phase}")
                self.logger.set_phase(current_phase)
                self._save_checkpoint(f"phase{current_phase}_start")
                last_phase = current_phase

            # Training step
            metrics = self.train_step()
            final_metrics = metrics

            # Logging
            if self.current_step % self.config.logging.log_interval == 0:
                self._log_metrics(metrics)

                # Console output
                phase = self._get_phase()
                logger.info(
                    f"Step {self.current_step} | Phase {phase} | "
                    f"Judge: {metrics['quality/judge_accuracy']:.3f} | "
                    f"Witness(fake): {metrics['quality/witness_accuracy_fake']:.3f} | "
                    f"Witness(real): {metrics['witness/accuracy_ema']:.3f}"
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

        # Cleanup
        self.logger.close()

        logger.info("Training complete")
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

        self.weaver.eval()

        with torch.no_grad():
            num_samples = self.config.logging.num_samples
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
            labels = torch.arange(10, device=self.device).repeat(num_samples // 10 + 1)[:num_samples]

            images, _ = self.weaver(z, labels)
            images = (images + 1) / 2  # Normalize to [0, 1]

            self.logger.log_images(
                "samples/generated",
                images,
                self.current_step,
                nrow=10,
            )

        self.weaver.train()

    def _save_checkpoint(self, suffix: Optional[str] = None) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpointing.checkpoint_dir)

        if suffix:
            path = checkpoint_dir / f"checkpoint_v2_{suffix}.pt"
        else:
            path = checkpoint_dir / f"checkpoint_v2_step{self.current_step}.pt"

        save_checkpoint(
            path=path,
            step=self.current_step,
            phase=self._get_phase(),
            models={
                "weaver": self.weaver,
                "witness": self.witness,
            },
            optimizers={
                "weaver": self.weaver_optimizer,
                "witness": self.witness_optimizer,
            },
            ema_state=self.ema_state.state_dict(),
            metrics={
                "witness_accuracy_ema": self.witness_accuracy_ema,
            },
            config=self.config.to_dict(),
        )

    def _load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        meta = load_checkpoint(
            path=path,
            models={
                "weaver": self.weaver,
                "witness": self.witness,
            },
            optimizers={
                "weaver": self.weaver_optimizer,
                "witness": self.witness_optimizer,
            },
            device=self.device,
        )

        self.current_step = meta["step"]

        if "ema_state" in meta:
            self.ema_state.load_state_dict(meta["ema_state"])

        if "metrics" in meta and "witness_accuracy_ema" in meta["metrics"]:
            self.witness_accuracy_ema = meta["metrics"]["witness_accuracy_ema"]

        logger.info(f"Resumed from step {self.current_step}, phase {meta['phase']}")

    def evaluate(self, num_samples: int = 1000) -> dict[str, float]:
        """
        Run evaluation on generated samples.

        Args:
            num_samples: Number of samples to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        self.weaver.eval()
        self.witness.eval()

        with torch.no_grad():
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
            labels = torch.randint(0, 10, (num_samples,), device=self.device)

            images, _ = self.weaver(z, labels)

            # Quality metrics
            quality = self.quality_metrics.evaluate(images, labels)

            # Mode diversity
            diversity = self.mode_diversity.evaluate(images, labels)

        return {
            "eval/judge_accuracy": quality.judge_accuracy,
            "eval/witness_accuracy": quality.witness_accuracy,
            "eval/agreement_rate": quality.agreement_rate,
            "eval/mode_coverage": diversity.mode_coverage,
            "eval/entropy": diversity.entropy,
            "eval/is_collapsed": float(diversity.is_collapsed),
        }
