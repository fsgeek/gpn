"""
GPN Trainer for three-phase training.

Implements the complete GPN training loop with:
- Phase 1: Scaffolding (heavy grounding, light alignment)
- Phase 2: Relationship (balanced, with empowerment)
- Phase 3: Drift Test (scaffolding removed)

Exports:
    - GPNTrainer: Three-phase GPN training loop
"""

from pathlib import Path
from typing import Optional, Iterator
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.weaver import Weaver
from src.models.witness import Witness
from src.models.judge import Judge
from src.training.config import TrainingConfig
from src.training.curriculum import PhaseManager
from src.training.ema import EMAState
from src.training.losses import CombinedLoss
from src.metrics.quality import QualityMetrics, CollusionDetector
from src.metrics.mode_diversity import ModeDiversity
from src.utils.logging import MetricsLogger
from src.utils.checkpointing import save_checkpoint, load_checkpoint, find_latest_checkpoint

logger = logging.getLogger(__name__)


class GPNTrainer:
    """
    Three-phase GPN training loop.

    Manages the complete training process including:
    - Phase transitions and weight scheduling
    - EMA state tracking (updated only on Witness forward)
    - Collusion detection
    - Checkpointing with full state
    - Metrics logging

    Attributes:
        config: Training configuration
        weaver: Generator model
        witness: Classifier model
        judge: Frozen grounding classifier
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
        Initialize GPN trainer.

        Args:
            config: Training configuration
            weaver: Weaver (generator) model
            witness: Witness (classifier) model
            judge: Frozen Judge model
            train_loader: Training data loader
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

        # Training state
        self.phase_manager = PhaseManager(
            phase1_steps=config.phase1_steps,
            phase2_steps=config.phase2_steps,
            phase1_weights=(
                config.phase1_weights.grounding,
                config.phase1_weights.alignment,
                config.phase1_weights.empowerment,
            ),
            phase2_weights=(
                config.phase2_weights.grounding,
                config.phase2_weights.alignment,
                config.phase2_weights.empowerment,
            ),
            phase3_weights=(
                config.phase3_weights.grounding,
                config.phase3_weights.alignment,
                config.phase3_weights.empowerment,
            ),
            on_phase_change=self._on_phase_change,
        )

        self.ema_state = EMAState(
            dim=config.weaver.v_pred_dim,
            decay=config.ema.decay,
            variance_threshold=config.ema.variance_threshold,
            window_size=config.ema.window_size,
            device=self.device,
        )

        # Loss function
        weights = self.phase_manager.get_weights()
        self.loss_fn = CombinedLoss(
            grounding_weight=weights[0],
            alignment_weight=weights[1],
            empowerment_weight=weights[2],
            target_kl=config.empowerment.target_kl,
            tolerance=config.empowerment.tolerance,
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

    def _on_phase_change(self, old_phase: int, new_phase: int) -> None:
        """Handle phase transition."""
        logger.info(f"Phase transition: {old_phase} -> {new_phase}")

        # Update loss weights
        weights = self.phase_manager.get_weights()
        self.loss_fn.update_weights(
            grounding=weights[0],
            alignment=weights[1],
            empowerment=weights[2],
        )

        # Update EMA and collusion detector
        self.ema_state.set_phase(new_phase)
        self.collusion_detector.set_phase(new_phase)

        # Save checkpoint on phase transition
        if self.config.checkpointing.save_on_phase_transition:
            self._save_checkpoint(f"phase{new_phase}_start")

        # Log phase transition
        if self.logger:
            self.logger.log_text(
                "phase_transition",
                f"Step {self.current_step}: Phase {old_phase} -> {new_phase}",
                self.current_step,
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

    def train_step(self) -> dict[str, float]:
        """
        Execute a single training step.

        Returns:
            Dictionary of loss components and metrics
        """
        current_phase = self.phase_manager.current_phase
        is_drift_test = current_phase >= 3

        # In Phase 3 (Drift Test), we observe without training
        if is_drift_test:
            self.weaver.eval()
            self.witness.eval()
        else:
            self.weaver.train()
            self.witness.train()

        # Get real images and labels
        real_images, labels = self._get_batch()
        batch_size = real_images.size(0)

        # Generate noise
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)

        # Use no_grad in Phase 3 to save memory
        context = torch.no_grad() if is_drift_test else torch.enable_grad()

        with context:
            # Forward pass through Weaver
            fake_images, v_pred = self.weaver(z, labels)

            # Forward pass through Witness (this triggers EMA update logic)
            witness_logits, v_seen = self.witness(fake_images)

            # Judge provides grounding signal
            with torch.no_grad():
                judge_logits = self.judge(fake_images)

            # Compute losses (for metrics, even in Phase 3)
            ema_mean = self.ema_state.mean if self.ema_state.initialized else None
            ema_var = self.ema_state.variance if self.ema_state.initialized else None

            total_loss, loss_components = self.loss_fn(
                witness_logits, labels, judge_logits, v_pred, v_seen,
                ema_mean, ema_var,
            )

        # Backward pass (skip in Phase 3 - we're observing, not training)
        if not is_drift_test and total_loss.requires_grad:
            self.weaver_optimizer.zero_grad()
            self.witness_optimizer.zero_grad()
            total_loss.backward()
            self.weaver_optimizer.step()
            self.witness_optimizer.step()

        # Update EMA (only after Witness forward - per spec)
        self.ema_state.update(v_seen.detach())

        # Update collusion detector
        quality = (judge_logits.argmax(dim=1) == labels).float().mean().item()
        self.collusion_detector.update(
            loss_components["alignment"].item(),
            quality,
        )

        # Prepare metrics
        metrics = {
            "loss/total": loss_components["total"].item(),
            "loss/grounding": loss_components["grounding"].item(),
            "loss/alignment": loss_components["alignment"].item(),
            "loss/empowerment": loss_components["empowerment"].item(),
            "quality/judge_accuracy": quality,
            "ema/mean_norm": self.ema_state.mean.norm().item(),
            "ema/variance_mean": self.ema_state.variance.mean().item(),
        }

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
        self.logger.set_phase(self.phase_manager.current_phase)

        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)
        else:
            # Check for latest checkpoint
            latest = find_latest_checkpoint(self.config.checkpointing.checkpoint_dir)
            if latest:
                logger.info(f"Found checkpoint: {latest}")
                # Don't auto-resume, let user decide

        logger.info(f"Starting training from step {self.current_step}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Device: {self.device}")

        # Training loop
        final_metrics = {}

        while self.current_step < total_steps:
            # Check for phase transition
            phase_changed = self.phase_manager.step(self.current_step)
            if phase_changed:
                self.logger.set_phase(self.phase_manager.current_phase)

            # Training step
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

        # Log phase info
        self.logger.log_scalar(
            "training/phase",
            self.phase_manager.current_phase,
            self.current_step,
        )
        self.logger.log_scalar(
            "training/phase_progress",
            self.phase_manager.progress_in_phase(),
            self.current_step,
        )

    def _log_samples(self) -> None:
        """Generate and log sample images."""
        if self.logger is None:
            return

        self.weaver.eval()

        with torch.no_grad():
            # Generate samples for each class
            num_samples = self.config.logging.num_samples
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
            labels = torch.arange(10, device=self.device).repeat(num_samples // 10 + 1)[:num_samples]

            images, _ = self.weaver(z, labels)

            # Normalize to [0, 1] for visualization
            images = (images + 1) / 2

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
            path = checkpoint_dir / f"checkpoint_{suffix}.pt"
        else:
            path = checkpoint_dir

        save_checkpoint(
            path=path,
            step=self.current_step,
            phase=self.phase_manager.current_phase,
            models={
                "weaver": self.weaver,
                "witness": self.witness,
            },
            optimizers={
                "weaver": self.weaver_optimizer,
                "witness": self.witness_optimizer,
            },
            ema_state=self.ema_state.state_dict(),
            metrics=self.collusion_detector.get_summary(),
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

        # Restore EMA state
        if "ema_state" in meta:
            self.ema_state.load_state_dict(meta["ema_state"])

        # Sync phase manager
        self.phase_manager.step(self.current_step)

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

        with torch.no_grad():
            # Generate samples
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
            labels = torch.randint(0, 10, (num_samples,), device=self.device)

            images, _ = self.weaver(z, labels)

            # Quality metrics
            quality = self.quality_metrics.evaluate(images, labels)

            # Mode diversity
            diversity = self.mode_diversity.evaluate(images, labels)

        self.weaver.train()

        return {
            "eval/judge_accuracy": quality.judge_accuracy,
            "eval/witness_accuracy": quality.witness_accuracy,
            "eval/agreement_rate": quality.agreement_rate,
            "eval/mode_coverage": diversity.mode_coverage,
            "eval/entropy": diversity.entropy,
            "eval/is_collapsed": float(diversity.is_collapsed),
        }
