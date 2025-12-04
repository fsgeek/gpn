"""
GPN Trainer V3 Ablation: No Meta-Learning Inner Loop.

This ablation tests whether the meta-learning structure is necessary,
or if consistent grounding + competence gating alone prevents collapse.

Key differences from full V3:
- NO inner loop (Witness doesn't train on Weaver's output)
- NO improvement measurement
- NO improvement scaling
- Just standard cross-entropy loss like V2

What's kept from V3:
- Consistent Witness grounding every step
- Competence gating (wait for threshold before training Weaver)

If this works: Grounding + gating is sufficient. Meta-learning is optimization.
If this fails: The inner loop / optimization target matters.

Exports:
    - GPNTrainerV3NoMeta: Ablation trainer without meta-learning
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
from src.utils.checkpointing import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class GPNTrainerV3NoMeta:
    """
    Ablation: V3 without meta-learning inner loop.

    Tests whether consistent grounding + competence gating alone
    prevents collapse, or if the meta-learning structure is necessary.
    """

    def __init__(
        self,
        config: TrainingConfig,
        weaver: Weaver,
        witness: Witness,
        judge: Judge,
        train_loader: DataLoader,
        device: Optional[torch.device] = None,
        competence_threshold: float = 0.5,
    ) -> None:
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

        # Competence tracking
        self.witness_competence = 0.0
        self.competence_ema_decay = 0.95
        self.competence_threshold = competence_threshold

        # EMA state
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

        # Logging
        self.logger: Optional[MetricsLogger] = None

        # Training state
        self.current_step = 0
        self._data_iterator: Optional[Iterator] = None

    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.Adam(
            model.parameters(),
            lr=self.config.optimizer.lr,
            betas=self.config.optimizer.betas,
            weight_decay=self.config.optimizer.weight_decay,
        )

    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._data_iterator is None:
            self._data_iterator = iter(self.train_loader)
        try:
            images, labels = next(self._data_iterator)
        except StopIteration:
            self._data_iterator = iter(self.train_loader)
            images, labels = next(self._data_iterator)
        return images.to(self.device), labels.to(self.device)

    def _get_phase(self) -> int:
        if self.current_step < self.config.phase1_steps:
            return 1
        elif self.current_step < self.config.phase2_steps:
            return 2
        else:
            return 3

    def train_step(self) -> dict[str, float]:
        """
        Training step WITHOUT meta-learning.

        Structure (like V2 but with consistent grounding + gating):
        1. Witness grounding on real data (EVERY step, not probabilistic)
        2. If Witness is competent: Train Weaver with standard cross-entropy
        3. Quality monitoring via Judge
        """
        phase = self._get_phase()
        is_drift_test = phase >= 3

        metrics = {}

        # Get real images and labels
        real_images, labels = self._get_batch()
        batch_size = real_images.size(0)

        # =====================================================================
        # Step 1: Witness Grounding (CONSISTENT - every step)
        # =====================================================================
        if not is_drift_test:
            self.witness.train()

            witness_logits_real, _ = self.witness(real_images)
            witness_grounding_loss = F.cross_entropy(witness_logits_real, labels)

            self.witness_optimizer.zero_grad()
            witness_grounding_loss.backward()
            self.witness_optimizer.step()

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
        # Step 2: Weaver Training (NO meta-learning - just standard cross-entropy)
        # =====================================================================
        witness_is_competent = self.witness_competence >= self.competence_threshold

        if witness_is_competent and not is_drift_test:
            self.weaver.train()
            self.witness.eval()

            # Generate fake images
            z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
            fake_images, v_pred = self.weaver(z, labels)

            # Witness evaluates (NO inner loop, NO improvement measurement)
            witness_logits_fake, v_seen = self.witness(fake_images)

            # Standard cross-entropy loss (like V2, NO improvement scaling)
            weaver_loss = F.cross_entropy(witness_logits_fake, labels)

            # Update Weaver
            self.weaver_optimizer.zero_grad()
            weaver_loss.backward()
            self.weaver_optimizer.step()

            metrics["loss/weaver"] = weaver_loss.item()

            # Update EMA
            self.ema_state.update(v_seen.detach())
        else:
            metrics["loss/weaver"] = 0.0

            # Generate samples for monitoring
            self.weaver.eval()
            self.witness.eval()
            with torch.no_grad():
                z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
                fake_images, v_pred = self.weaver(z, labels)
                witness_logits_fake, v_seen = self.witness(fake_images)

        # =====================================================================
        # Step 3: Quality Monitoring
        # =====================================================================
        with torch.no_grad():
            judge_logits = self.judge(fake_images)
            judge_accuracy = (judge_logits.argmax(dim=1) == labels).float().mean().item()
            witness_accuracy_fake = (witness_logits_fake.argmax(dim=1) == labels).float().mean().item()
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
        total_steps = total_steps or self.config.total_steps
        log_dir = log_dir or self.config.logging.log_dir

        self.logger = MetricsLogger(log_dir)

        if resume_from:
            self._load_checkpoint(resume_from)

        logger.info(f"Starting GPN V3-NoMeta (Ablation) training from step {self.current_step}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Competence threshold: {self.competence_threshold}")
        logger.info("ABLATION: No meta-learning inner loop - just grounding + gating + cross-entropy")

        final_metrics = {}
        last_phase = self._get_phase()
        competence_logged = False

        while self.current_step < total_steps:
            current_phase = self._get_phase()
            if current_phase != last_phase:
                logger.info(f"Phase transition: {last_phase} -> {current_phase}")
                self.logger.set_phase(current_phase)
                self._save_checkpoint(f"phase{current_phase}_start")
                last_phase = current_phase

            if not competence_logged and self.witness_competence >= self.competence_threshold:
                logger.info(f"Witness reached competence threshold at step {self.current_step}")
                competence_logged = True

            metrics = self.train_step()
            final_metrics = metrics

            if self.current_step % self.config.logging.log_interval == 0:
                self._log_metrics(metrics)
                phase = self._get_phase()
                gated = "GATED" if metrics["training/competence_gated"] else "waiting"
                logger.info(
                    f"Step {self.current_step} | Phase {phase} | {gated} | "
                    f"Judge: {metrics['quality/judge_accuracy']:.3f} | "
                    f"Witness(fake): {metrics['quality/witness_accuracy_fake']:.3f} | "
                    f"Competence: {metrics['witness/competence_ema']:.3f}"
                )

            if self.current_step % self.config.logging.sample_interval == 0:
                self._log_samples()

            if self.current_step % self.config.checkpointing.save_interval == 0:
                self._save_checkpoint()

            self.current_step += 1

        self._save_checkpoint("final")
        self.logger.close()

        logger.info("Training complete")
        return final_metrics

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        if self.logger is None:
            return
        for name, value in metrics.items():
            self.logger.log_scalar(name, value, self.current_step)

    def _log_samples(self) -> None:
        if self.logger is None:
            return
        self.weaver.eval()
        with torch.no_grad():
            num_samples = self.config.logging.num_samples
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
            labels = torch.arange(10, device=self.device).repeat(num_samples // 10 + 1)[:num_samples]
            images, _ = self.weaver(z, labels)
            images = (images + 1) / 2
            self.logger.log_images("samples/generated", images, self.current_step, nrow=10)
        self.weaver.train()

    def _save_checkpoint(self, suffix: Optional[str] = None) -> None:
        checkpoint_dir = Path(self.config.checkpointing.checkpoint_dir)
        if suffix:
            path = checkpoint_dir / f"checkpoint_v3nometa_{suffix}.pt"
        else:
            path = checkpoint_dir / f"checkpoint_v3nometa_step{self.current_step}.pt"

        save_checkpoint(
            path=path,
            step=self.current_step,
            phase=self._get_phase(),
            models={"weaver": self.weaver, "witness": self.witness},
            optimizers={"weaver": self.weaver_optimizer, "witness": self.witness_optimizer},
            ema_state=self.ema_state.state_dict(),
            metrics={"witness_competence": self.witness_competence},
            config=self.config.to_dict(),
        )

    def _load_checkpoint(self, path: str) -> None:
        meta = load_checkpoint(
            path=path,
            models={"weaver": self.weaver, "witness": self.witness},
            optimizers={"weaver": self.weaver_optimizer, "witness": self.witness_optimizer},
            device=self.device,
        )
        self.current_step = meta["step"]
        if "ema_state" in meta:
            self.ema_state.load_state_dict(meta["ema_state"])
        if "metrics" in meta and "witness_competence" in meta["metrics"]:
            self.witness_competence = meta["metrics"]["witness_competence"]
        logger.info(f"Resumed from step {self.current_step}, phase {meta['phase']}")

    def evaluate(self, num_samples: int = 1000) -> dict[str, float]:
        self.weaver.eval()
        self.witness.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
            labels = torch.randint(0, 10, (num_samples,), device=self.device)
            images, _ = self.weaver(z, labels)
            quality = self.quality_metrics.evaluate(images, labels)
            diversity = self.mode_diversity.evaluate(images, labels)
        return {
            "eval/judge_accuracy": quality.judge_accuracy,
            "eval/witness_accuracy": quality.witness_accuracy,
            "eval/agreement_rate": quality.agreement_rate,
            "eval/mode_coverage": diversity.mode_coverage,
            "eval/entropy": diversity.entropy,
            "eval/is_collapsed": float(diversity.is_collapsed),
        }
