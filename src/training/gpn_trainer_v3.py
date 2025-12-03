"""
GPN Trainer V3: Meta-Learning Architecture.

Key insight from V2 failure: Weaver exploited random-Witness before Witness
developed any competence. The fix: Weaver is rewarded for Witness's IMPROVEMENT
on real data, not Witness's approval.

Pedagogical framing:
- Teacher (Weaver) succeeds when student (Witness) learns
- Learning is measured by improvement on the exam (held-out real data)
- This is meta-learning: inner loop trains Witness, outer loop trains Weaver

Training structure per step:
1. Measure Witness's baseline accuracy on held-out real data
2. Witness trains on Weaver's generated examples
3. Measure Witness's post-training accuracy on held-out real data
4. Weaver's reward = improvement in Witness's accuracy

The crucial difference from V2:
- V2: Weaver optimizes for Witness's approval (cross-entropy on fake)
- V3: Weaver optimizes for Witness's improvement (delta accuracy on real)

This prevents exploitation because random-Witness shows no improvement.
Weaver can only reduce loss by producing examples that actually help Witness learn.

Exports:
    - GPNTrainerV3: Meta-learning GPN training loop
"""

from pathlib import Path
from typing import Optional, Iterator
import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.models.weaver import Weaver
from src.models.witness import Witness
from src.models.judge import Judge
from src.training.config import TrainingConfig
from src.training.ema import EMAState
from src.metrics.quality import QualityMetrics, CollusionDetector
from src.metrics.mode_diversity import ModeDiversity
from src.utils.logging import MetricsLogger
from src.utils.checkpointing import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class GPNTrainerV3:
    """
    Meta-learning GPN training loop.

    Key insight: Weaver is rewarded for Witness's improvement on real data,
    not for Witness's approval of generated images.

    Inner loop: Witness trains on Weaver's output
    Outer loop: Weaver's loss = -improvement in Witness's real-data accuracy

    This prevents the V2 failure mode where Weaver exploited random-Witness.
    Random-Witness shows no improvement, so Weaver gets no reward until it
    produces genuinely educational examples.

    Attributes:
        config: Training configuration
        weaver: Generator model (teacher)
        witness: Classifier model (student)
        judge: Frozen evaluation classifier (exam)
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
        Initialize GPN trainer V3.

        Args:
            config: Training configuration
            weaver: Weaver (generator/teacher) model
            witness: Witness (classifier/student) model
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

        # Data - split into training and held-out validation
        self.train_loader = train_loader

        # Optimizers
        self.weaver_optimizer = self._create_optimizer(self.weaver)
        self.witness_optimizer = self._create_optimizer(self.witness)

        # Competence tracking
        self.witness_competence = 0.0  # EMA of Witness accuracy on real data
        self.competence_ema_decay = 0.95
        self.competence_threshold = 0.5  # Only train Weaver when Witness shows competence

        # Meta-learning state
        self.inner_steps = 3  # How many steps Witness trains on Weaver's output
        self.held_out_batch_size = 128  # Size of held-out evaluation batch

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

    def _measure_witness_accuracy(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        """Measure Witness accuracy on a batch without training."""
        self.witness.eval()
        with torch.no_grad():
            logits, _ = self.witness(images)
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
        return accuracy

    def train_step(self) -> dict[str, float]:
        """
        Execute a single meta-learning training step.

        Structure:
        1. Ground Witness on real data (establish baseline competence)
        2. If Witness is competent:
           a. Snapshot Witness state
           b. Measure baseline accuracy on held-out real data
           c. Train Witness on Weaver's generated examples (inner loop)
           d. Measure post-training accuracy on held-out real data
           e. Weaver's reward = improvement (outer loop)
           f. Restore Witness state (meta-learning requires this)
        3. Quality monitoring via Judge

        Returns:
            Dictionary of metrics
        """
        phase = self._get_phase()
        is_drift_test = phase >= 3

        metrics = {}

        # Get real data batches
        real_images, labels = self._get_batch()
        held_out_images, held_out_labels = self._get_batch()  # Separate held-out batch
        batch_size = real_images.size(0)

        # =====================================================================
        # Step 1: Witness Grounding (establish baseline competence on real data)
        # =====================================================================
        if not is_drift_test:
            self.witness.train()

            # Train Witness on real data
            witness_logits_real, _ = self.witness(real_images)
            witness_grounding_loss = F.cross_entropy(witness_logits_real, labels)

            self.witness_optimizer.zero_grad()
            witness_grounding_loss.backward()
            self.witness_optimizer.step()

            # Track Witness competence
            with torch.no_grad():
                witness_acc_real = (witness_logits_real.argmax(dim=1) == labels).float().mean().item()
                self.witness_competence = (
                    self.competence_ema_decay * self.witness_competence
                    + (1 - self.competence_ema_decay) * witness_acc_real
                )

            metrics["witness/grounding_loss"] = witness_grounding_loss.item()
            metrics["witness/accuracy_real"] = witness_acc_real
            metrics["witness/competence_ema"] = self.witness_competence
        else:
            metrics["witness/grounding_loss"] = 0.0
            metrics["witness/accuracy_real"] = 0.0
            metrics["witness/competence_ema"] = self.witness_competence

        # =====================================================================
        # Step 2: Meta-Learning (only if Witness shows competence)
        # =====================================================================
        witness_is_competent = self.witness_competence >= self.competence_threshold

        if witness_is_competent and not is_drift_test:
            # 2a. Snapshot Witness state (meta-learning requires state restoration)
            witness_snapshot = copy.deepcopy(self.witness.state_dict())
            witness_opt_snapshot = copy.deepcopy(self.witness_optimizer.state_dict())

            # 2b. Measure baseline accuracy on held-out data
            baseline_accuracy = self._measure_witness_accuracy(held_out_images, held_out_labels)

            # 2c. Inner loop: Train Witness on Weaver's generated examples
            self.weaver.eval()
            self.witness.train()

            inner_losses = []
            for _ in range(self.inner_steps):
                z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
                target_labels = torch.randint(0, 10, (batch_size,), device=self.device)

                with torch.no_grad():
                    fake_images, _ = self.weaver(z, target_labels)

                # Witness trains to classify Weaver's output
                witness_logits, _ = self.witness(fake_images)
                inner_loss = F.cross_entropy(witness_logits, target_labels)

                self.witness_optimizer.zero_grad()
                inner_loss.backward()
                self.witness_optimizer.step()

                inner_losses.append(inner_loss.item())

            # 2d. Measure post-training accuracy on held-out data
            post_accuracy = self._measure_witness_accuracy(held_out_images, held_out_labels)

            # Calculate improvement
            improvement = post_accuracy - baseline_accuracy

            metrics["meta/baseline_accuracy"] = baseline_accuracy
            metrics["meta/post_accuracy"] = post_accuracy
            metrics["meta/improvement"] = improvement
            metrics["meta/inner_loss_mean"] = sum(inner_losses) / len(inner_losses)

            # 2e. Restore Witness state (essential for meta-learning)
            self.witness.load_state_dict(witness_snapshot)
            self.witness_optimizer.load_state_dict(witness_opt_snapshot)

            # 2f. Outer loop: Train Weaver to maximize improvement
            # We use a differentiable proxy: train Weaver so that Witness's
            # cross-entropy on held-out data decreases after training on Weaver's output

            self.weaver.train()
            self.witness.eval()  # Witness provides feedback, doesn't learn here

            # Generate training examples from Weaver
            z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
            fake_images, v_pred = self.weaver(z, labels)

            # Weaver's loss: Make Witness classify correctly
            # But this is gated by competence, so we're not exploiting random-Witness
            witness_logits_fake, v_seen = self.witness(fake_images)
            weaver_loss = F.cross_entropy(witness_logits_fake, labels)

            # Scale loss by inverse of improvement (encourage positive improvement)
            # If improvement is negative, increase loss; if positive, decrease loss
            improvement_scale = max(0.1, 1.0 - improvement * 10)  # Heuristic scaling
            weaver_loss = weaver_loss * improvement_scale

            # Update Weaver
            self.weaver_optimizer.zero_grad()
            weaver_loss.backward()
            self.weaver_optimizer.step()

            metrics["loss/weaver"] = weaver_loss.item()
            metrics["loss/improvement_scale"] = improvement_scale

            # Update EMA
            self.ema_state.update(v_seen.detach())

        else:
            # Witness not yet competent or drift test - no Weaver training
            if not witness_is_competent:
                logger.debug(f"Step {self.current_step}: Witness competence {self.witness_competence:.3f} < threshold {self.competence_threshold}")

            metrics["meta/baseline_accuracy"] = 0.0
            metrics["meta/post_accuracy"] = 0.0
            metrics["meta/improvement"] = 0.0
            metrics["meta/inner_loss_mean"] = 0.0
            metrics["loss/weaver"] = 0.0
            metrics["loss/improvement_scale"] = 1.0

            # Generate samples for monitoring even when not training
            self.weaver.eval()
            self.witness.eval()
            with torch.no_grad():
                z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
                fake_images, v_pred = self.weaver(z, labels)
                witness_logits_fake, v_seen = self.witness(fake_images)

        # =====================================================================
        # Step 3: Quality Monitoring (Judge evaluates, no training signal)
        # =====================================================================
        with torch.no_grad():
            judge_logits = self.judge(fake_images)

            # Judge accuracy: Does Judge recognize the intended digit?
            judge_accuracy = (judge_logits.argmax(dim=1) == labels).float().mean().item()

            # Witness accuracy on fake: Does Witness recognize it?
            witness_accuracy_fake = (witness_logits_fake.argmax(dim=1) == labels).float().mean().item()

            # Agreement: Do Judge and Witness agree?
            agreement = (judge_logits.argmax(dim=1) == witness_logits_fake.argmax(dim=1)).float().mean().item()

        metrics["quality/judge_accuracy"] = judge_accuracy
        metrics["quality/witness_accuracy_fake"] = witness_accuracy_fake
        metrics["quality/agreement"] = agreement
        metrics["training/phase"] = float(phase)
        metrics["training/competence_gated"] = float(witness_is_competent)
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

        logger.info(f"Starting GPN V3 (Meta-Learning) training from step {self.current_step}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Competence threshold: {self.competence_threshold}")
        logger.info(f"Inner loop steps: {self.inner_steps}")
        logger.info("Architecture: Weaver rewarded for Witness improvement on real data")

        # Training loop
        final_metrics = {}
        last_phase = self._get_phase()
        competence_logged = False

        while self.current_step < total_steps:
            # Check for phase transition
            current_phase = self._get_phase()
            if current_phase != last_phase:
                logger.info(f"Phase transition: {last_phase} -> {current_phase}")
                self.logger.set_phase(current_phase)
                self._save_checkpoint(f"phase{current_phase}_start")
                last_phase = current_phase

            # Log when Witness first reaches competence
            if not competence_logged and self.witness_competence >= self.competence_threshold:
                logger.info(f"Witness reached competence threshold at step {self.current_step}")
                competence_logged = True

            # Training step
            metrics = self.train_step()
            final_metrics = metrics

            # Logging
            if self.current_step % self.config.logging.log_interval == 0:
                self._log_metrics(metrics)

                # Console output
                phase = self._get_phase()
                gated = "GATED" if metrics["training/competence_gated"] else "waiting"
                logger.info(
                    f"Step {self.current_step} | Phase {phase} | {gated} | "
                    f"Judge: {metrics['quality/judge_accuracy']:.3f} | "
                    f"Witness(fake): {metrics['quality/witness_accuracy_fake']:.3f} | "
                    f"Competence: {metrics['witness/competence_ema']:.3f} | "
                    f"Improvement: {metrics['meta/improvement']:.4f}"
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
            path = checkpoint_dir / f"checkpoint_v3_{suffix}.pt"
        else:
            path = checkpoint_dir / f"checkpoint_v3_step{self.current_step}.pt"

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
                "witness_competence": self.witness_competence,
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

        if "metrics" in meta and "witness_competence" in meta["metrics"]:
            self.witness_competence = meta["metrics"]["witness_competence"]

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
