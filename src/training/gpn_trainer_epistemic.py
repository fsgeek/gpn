"""
GPN Trainer with Epistemic Instrumentation

Extends GPNTrainer to add epistemic honesty tracking during training.
Supports comparative analysis of multiple epistemic approaches.

This trainer is used for Phase 1 epistemic study experiments.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.weaver import Weaver
from src.models.witness import Witness
from src.models.judge import Judge
from src.training.config import TrainingConfig
from src.training.gpn_trainer import GPNTrainer
from src.metrics.epistemic import ComparativeTracker


class GPNTrainerEpistemic(GPNTrainer):
    """
    GPN Trainer with epistemic instrumentation.

    Extends base GPNTrainer to track epistemic honesty metrics:
    - Simple 2D (alignment + correctness)
    - Neutrosophic {T, I, F}
    - Bayesian uncertainty (Week 2)

    Usage:
        trainer = GPNTrainerEpistemic(
            config=config,
            weaver=weaver,
            witness=witness,
            judge=judge,
            train_loader=train_loader,
            enable_epistemic=True,
        )

        trainer.train(total_steps=10000, log_dir='logs/epistemic_baseline')

        # Access epistemic data
        summary = trainer.epistemic_tracker.get_summary_statistics()
    """

    def __init__(
        self,
        config: TrainingConfig,
        weaver: Weaver,
        witness: Witness,
        judge: Judge,
        train_loader: DataLoader,
        device: Optional[torch.device] = None,
        # Epistemic tracking options
        enable_epistemic: bool = True,
        enable_simple_2d: bool = True,
        enable_neutrosophic: bool = True,
        enable_bayesian: bool = False,  # Week 2
        random_sampling: bool = False,
        sample_rate: float = 0.5,
    ) -> None:
        """
        Initialize epistemic GPN trainer.

        Args:
            config: Training configuration
            weaver: Weaver (generator) model
            witness: Witness (classifier) model
            judge: Frozen Judge model
            train_loader: Training data loader
            device: Training device
            enable_epistemic: Enable epistemic tracking
            enable_simple_2d: Enable Simple 2D approach
            enable_neutrosophic: Enable Neutrosophic approach
            enable_bayesian: Enable Bayesian approach (not yet implemented)
            random_sampling: Enable sparse tensor sampling (test for gaming)
            sample_rate: Fraction of metrics to compute when sampling
        """
        # Initialize base trainer
        super().__init__(
            config=config,
            weaver=weaver,
            witness=witness,
            judge=judge,
            train_loader=train_loader,
            device=device,
        )

        # Initialize epistemic tracker
        self.enable_epistemic = enable_epistemic

        if self.enable_epistemic:
            self.epistemic_tracker = ComparativeTracker(
                device=self.device,
                enable_simple_2d=enable_simple_2d,
                enable_neutrosophic=enable_neutrosophic,
                enable_bayesian=enable_bayesian,
                random_sampling=random_sampling,
                sample_rate=sample_rate,
            )
        else:
            self.epistemic_tracker = None

        # Cache tensors from last train_step (for epistemic computation)
        self._last_v_pred: Optional[torch.Tensor] = None
        self._last_v_seen: Optional[torch.Tensor] = None
        self._last_judge_logits: Optional[torch.Tensor] = None
        self._last_witness_logits: Optional[torch.Tensor] = None
        self._last_labels: Optional[torch.Tensor] = None
        self._last_fake_images: Optional[torch.Tensor] = None

    def train_step(self) -> dict[str, float]:
        """
        Execute a single training step with epistemic tracking.

        Extends base train_step to compute epistemic metrics.

        Returns:
            Dictionary of loss components, metrics, and epistemic metrics
        """
        current_phase = self.phase_manager.current_phase
        is_drift_test = current_phase >= 3

        # Phase setup
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

        # Forward pass
        context = torch.no_grad() if is_drift_test else torch.enable_grad()

        with context:
            # Weaver forward
            fake_images, v_pred = self.weaver(z, labels)

            # Witness forward
            witness_logits, v_seen = self.witness(fake_images)

            # Judge forward
            with torch.no_grad():
                judge_logits = self.judge(fake_images)

            # Compute losses
            ema_mean = self.ema_state.mean if self.ema_state.initialized else None
            ema_var = self.ema_state.variance if self.ema_state.initialized else None

            total_loss, loss_components = self.loss_fn(
                witness_logits, labels, judge_logits, v_pred, v_seen,
                ema_mean, ema_var,
            )

        # Cache tensors for epistemic computation
        if self.enable_epistemic:
            self._last_v_pred = v_pred.detach()
            self._last_v_seen = v_seen.detach()
            self._last_judge_logits = judge_logits.detach()
            self._last_witness_logits = witness_logits.detach()
            self._last_labels = labels.detach()
            self._last_fake_images = fake_images.detach()

        # Backward pass
        if not is_drift_test and total_loss.requires_grad:
            self.weaver_optimizer.zero_grad()
            self.witness_optimizer.zero_grad()
            total_loss.backward()
            self.weaver_optimizer.step()
            self.witness_optimizer.step()

        # Update EMA
        self.ema_state.update(v_seen.detach())

        # Update collusion detector
        quality = (judge_logits.argmax(dim=1) == labels).float().mean().item()
        self.collusion_detector.update(
            loss_components["alignment"].item(),
            quality,
        )

        # Base metrics
        metrics = {
            "loss/total": loss_components["total"].item(),
            "loss/grounding": loss_components["grounding"].item(),
            "loss/alignment": loss_components["alignment"].item(),
            "loss/empowerment": loss_components["empowerment"].item(),
            "quality/judge_accuracy": quality,
            "ema/mean_norm": self.ema_state.mean.norm().item(),
            "ema/variance_mean": self.ema_state.variance.mean().item(),
        }

        # Compute epistemic metrics
        if self.enable_epistemic:
            epistemic_states = self.epistemic_tracker.compute_all(
                step=self.current_step,
                v_pred=self._last_v_pred,
                v_seen=self._last_v_seen,
                judge_logits=self._last_judge_logits,
                witness_logits=self._last_witness_logits,
                labels=self._last_labels,
                fake_images=self._last_fake_images,
            )

            # Flatten epistemic metrics into main metrics dict
            for approach_name, state in epistemic_states.items():
                for metric_key, metric_value in state.metrics.items():
                    if isinstance(metric_value, (int, float)):
                        metrics[f'epistemic/{approach_name}/{metric_key}'] = metric_value

        return metrics

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """
        Log metrics to TensorBoard (including epistemic metrics).

        Extends base _log_metrics to handle epistemic metrics.

        Args:
            metrics: Dictionary of metrics (includes epistemic metrics)
        """
        # Call parent logging
        super()._log_metrics(metrics)

        # Additional epistemic logging (if tracker exists)
        # Note: epistemic metrics are already in metrics dict from train_step,
        # so they'll be logged by parent. This is just for any additional
        # visualization or summary stats.
        pass

    def _save_checkpoint(self, suffix: Optional[str] = None) -> None:
        """
        Save training checkpoint (including epistemic state).

        Extends base _save_checkpoint to include epistemic tracker history.

        Args:
            suffix: Optional suffix for checkpoint filename
        """
        # Call parent to save base checkpoint
        super()._save_checkpoint(suffix)

        # Save epistemic states separately
        if self.enable_epistemic and self.epistemic_tracker:
            checkpoint_dir = Path(self.config.checkpointing.checkpoint_dir)
            epistemic_dir = checkpoint_dir / 'epistemic'

            if suffix:
                save_path = epistemic_dir / f'epistemic_{suffix}'
            else:
                save_path = epistemic_dir / f'epistemic_step_{self.current_step}'

            self.epistemic_tracker.save_states(str(save_path))

    def save_epistemic_summary(self, output_path: str):
        """
        Save summary statistics for epistemic tracking.

        Args:
            output_path: Path to save summary JSON
        """
        if not self.enable_epistemic or not self.epistemic_tracker:
            return

        import json
        from pathlib import Path

        summary = self.epistemic_tracker.get_summary_statistics()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def load_epistemic_states(self, input_path: str):
        """
        Load epistemic states from disk.

        Args:
            input_path: Path to directory containing epistemic states
        """
        if not self.enable_epistemic or not self.epistemic_tracker:
            return

        self.epistemic_tracker.load_states(input_path)
