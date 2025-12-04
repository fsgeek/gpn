"""
GPN-2 Trainer: Curriculum-based training for 2-digit number generation.

Tests whether curriculum learning (single digits → composition) produces
better multi-digit generators than training from scratch.

Curriculum Phases:
- Phase 1: Freeze single-digit Weaver, train only composition layer
- Phase 2: Unfreeze everything, end-to-end fine-tuning
- Phase 3: Drift test (remove grounding, test stability)

Based on V3-NoMeta findings: meta-learning inner loop not needed,
classification grounding + competence gating is sufficient.

Exports:
    - GPNTrainerTwoDigit: Curriculum trainer for 2-digit generation
"""

from pathlib import Path
from typing import Optional, Iterator
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.weaver_twodigit import TwoDigitWeaver
from src.models.witness_twodigit import TwoDigitWitness
from src.models.judge_twodigit import TwoDigitJudge
from src.data.multidigit import TwoDigitMNIST
from src.training.config import TrainingConfig
from src.training.ema import EMAState
from src.utils.logging import MetricsLogger
from src.utils.checkpointing import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class GPNTrainerTwoDigit:
    """
    Curriculum trainer for 2-digit MNIST generation.

    Phase 1: Composition learning (frozen digit Weaver)
    Phase 2: End-to-end fine-tuning (unfrozen)
    Phase 3: Drift test (no grounding)
    """

    def __init__(
        self,
        weaver: TwoDigitWeaver,
        witness: TwoDigitWitness,
        judge: TwoDigitJudge,
        train_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
        competence_threshold: float = 0.5,
        phase1_steps: int = 2000,
        phase2_steps: int = 6000,
        total_steps: int = 8000,
        latent_dim: int = 64,
        v_pred_dim: int = 16,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.config = config

        # Models
        self.weaver = weaver.to(self.device)
        self.witness = witness.to(self.device)
        self.judge = judge.to(self.device)
        self.judge.freeze()

        # Data
        self.train_loader = train_loader

        # Training parameters
        self.latent_dim = latent_dim
        self.v_pred_dim = v_pred_dim
        self.phase1_steps = phase1_steps
        self.phase2_steps = phase2_steps
        self.total_steps = total_steps

        # Optimizers (will be recreated when unfreezing)
        self.weaver_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.weaver.parameters()),
            lr=0.0002,
            betas=(0.5, 0.999),
        )
        self.witness_optimizer = optim.Adam(
            self.witness.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999),
        )

        # Competence tracking
        self.witness_competence = 0.0
        self.competence_ema_decay = 0.95
        self.competence_threshold = competence_threshold

        # EMA state
        self.ema_state = EMAState(
            dim=v_pred_dim * 2,  # Two digits
            decay=0.99,
            variance_threshold=0.000001,
            window_size=100,
            device=self.device,
        )

        # Logging
        self.logger: Optional[MetricsLogger] = None

        # Training state
        self.current_step = 0
        self._data_iterator: Optional[Iterator] = None
        self._phase_transitioned = {2: False, 3: False}

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
        if self.current_step < self.phase1_steps:
            return 1
        elif self.current_step < self.phase2_steps:
            return 2
        else:
            return 3

    def _handle_phase_transition(self, phase: int) -> None:
        """Handle phase transitions (unfreezing, etc.)."""
        if phase == 2 and not self._phase_transitioned[2]:
            logger.info("Phase 2: Unfreezing single-digit Weaver for end-to-end training")
            self.weaver.unfreeze_digit_weaver()
            # Recreate optimizer with all parameters
            self.weaver_optimizer = optim.Adam(
                self.weaver.parameters(),
                lr=0.0001,  # Lower LR for fine-tuning
                betas=(0.5, 0.999),
            )
            self._phase_transitioned[2] = True

        elif phase == 3 and not self._phase_transitioned[3]:
            logger.info("Phase 3: Drift test - removing grounding")
            self._phase_transitioned[3] = True

    def train_step(self) -> dict[str, float]:
        """
        Training step with curriculum phases.

        Phase 1: Train composition only (digits frozen)
        Phase 2: End-to-end training (all unfrozen)
        Phase 3: Drift test (no grounding)
        """
        phase = self._get_phase()
        self._handle_phase_transition(phase)
        is_drift_test = phase >= 3

        metrics = {}

        # Get real images and labels
        real_images, labels = self._get_batch()
        batch_size = real_images.size(0)

        # =====================================================================
        # Step 1: Witness Grounding (on real 2-digit images)
        # =====================================================================
        if not is_drift_test:
            self.witness.train()

            witness_logits_real, _ = self.witness(real_images, mode="full")
            witness_grounding_loss = F.cross_entropy(witness_logits_real, labels)

            self.witness_optimizer.zero_grad()
            witness_grounding_loss.backward()
            self.witness_optimizer.step()

            with torch.no_grad():
                witness_acc_real = (
                    (witness_logits_real.argmax(dim=1) == labels).float().mean().item()
                )
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
        # Step 2: Weaver Training
        # =====================================================================
        witness_is_competent = self.witness_competence >= self.competence_threshold

        if witness_is_competent and not is_drift_test:
            self.weaver.train()
            self.witness.eval()

            # Generate fake 2-digit images
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_images, v_pred = self.weaver(z, labels)

            # Witness evaluates
            witness_logits_fake, v_seen = self.witness(fake_images, mode="full")

            # Standard cross-entropy loss
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
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_images, v_pred = self.weaver(z, labels)
                witness_logits_fake, v_seen = self.witness(fake_images, mode="full")

        # =====================================================================
        # Step 3: Quality Monitoring
        # =====================================================================
        with torch.no_grad():
            judge_logits = self.judge(fake_images, mode="full")
            judge_accuracy = (judge_logits.argmax(dim=1) == labels).float().mean().item()
            witness_accuracy_fake = (
                (witness_logits_fake.argmax(dim=1) == labels).float().mean().item()
            )
            agreement = (
                (judge_logits.argmax(dim=1) == witness_logits_fake.argmax(dim=1))
                .float()
                .mean()
                .item()
            )

            # Per-digit accuracy (decompose labels)
            tens = labels // 10
            ones = labels % 10
            pred_labels = judge_logits.argmax(dim=1)
            pred_tens = pred_labels // 10
            pred_ones = pred_labels % 10
            tens_accuracy = (pred_tens == tens).float().mean().item()
            ones_accuracy = (pred_ones == ones).float().mean().item()

        metrics["quality/judge_accuracy"] = judge_accuracy
        metrics["quality/witness_accuracy_fake"] = witness_accuracy_fake
        metrics["quality/agreement"] = agreement
        metrics["quality/tens_accuracy"] = tens_accuracy
        metrics["quality/ones_accuracy"] = ones_accuracy
        metrics["training/phase"] = float(phase)
        metrics["training/competence_gated"] = float(witness_is_competent)
        metrics["training/digit_weaver_frozen"] = float(self.weaver.is_digit_weaver_frozen())
        metrics["ema/mean_norm"] = self.ema_state.mean.norm().item()
        metrics["ema/variance_mean"] = self.ema_state.variance.mean().item()

        return metrics

    def train(
        self,
        total_steps: Optional[int] = None,
        log_dir: str = "experiments/gpn2",
        resume_from: Optional[str] = None,
    ) -> dict[str, float]:
        total_steps = total_steps or self.total_steps

        self.logger = MetricsLogger(log_dir)

        if resume_from:
            self._load_checkpoint(resume_from)

        logger.info(f"Starting GPN-2 (Two-Digit) curriculum training from step {self.current_step}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Phase 1 (composition): 0-{self.phase1_steps}")
        logger.info(f"Phase 2 (fine-tuning): {self.phase1_steps}-{self.phase2_steps}")
        logger.info(f"Phase 3 (drift test): {self.phase2_steps}+")
        logger.info(f"Device: {self.device}")
        logger.info(f"Digit Weaver frozen: {self.weaver.is_digit_weaver_frozen()}")

        final_metrics = {}
        last_phase = self._get_phase()
        competence_logged = False

        while self.current_step < total_steps:
            current_phase = self._get_phase()
            if current_phase != last_phase:
                logger.info(f"Phase transition: {last_phase} -> {current_phase}")
                if self.logger:
                    self.logger.set_phase(current_phase)
                self._save_checkpoint(f"phase{current_phase}_start")
                last_phase = current_phase

            if not competence_logged and self.witness_competence >= self.competence_threshold:
                logger.info(f"Witness reached competence threshold at step {self.current_step}")
                competence_logged = True

            metrics = self.train_step()
            final_metrics = metrics

            if self.current_step % 100 == 0:
                self._log_metrics(metrics)
                phase = self._get_phase()
                gated = "GATED" if metrics["training/competence_gated"] else "waiting"
                frozen = "frozen" if metrics["training/digit_weaver_frozen"] else "unfrozen"
                logger.info(
                    f"Step {self.current_step} | Phase {phase} | {gated} | {frozen} | "
                    f"Judge: {metrics['quality/judge_accuracy']:.3f} | "
                    f"Tens: {metrics['quality/tens_accuracy']:.3f} | "
                    f"Ones: {metrics['quality/ones_accuracy']:.3f}"
                )

            if self.current_step % 500 == 0:
                self._log_samples()

            if self.current_step % 1000 == 0:
                self._save_checkpoint()

            self.current_step += 1

        self._save_checkpoint("final")
        if self.logger:
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
            num_samples = 20
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            # Sample various 2-digit numbers
            labels = torch.tensor(
                [0, 11, 22, 33, 44, 55, 66, 77, 88, 99,
                 10, 21, 32, 43, 54, 65, 76, 87, 98, 9],
                device=self.device,
            )
            images, _ = self.weaver(z, labels)
            images = (images + 1) / 2
            self.logger.log_images("samples/generated", images, self.current_step, nrow=10)
        self.weaver.train()

    def _save_checkpoint(self, suffix: Optional[str] = None) -> None:
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if suffix:
            path = checkpoint_dir / f"checkpoint_twodigit_{suffix}.pt"
        else:
            path = checkpoint_dir / f"checkpoint_twodigit_step{self.current_step}.pt"

        save_checkpoint(
            path=path,
            step=self.current_step,
            phase=self._get_phase(),
            models={"weaver": self.weaver, "witness": self.witness},
            optimizers={"weaver": self.weaver_optimizer, "witness": self.witness_optimizer},
            ema_state=self.ema_state.state_dict(),
            metrics={
                "witness_competence": self.witness_competence,
                "phase_transitioned": self._phase_transitioned,
            },
            config=None,
        )
        logger.info(f"Saved checkpoint to {path}")

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
        if "metrics" in meta:
            if "witness_competence" in meta["metrics"]:
                self.witness_competence = meta["metrics"]["witness_competence"]
            if "phase_transitioned" in meta["metrics"]:
                self._phase_transitioned = meta["metrics"]["phase_transitioned"]
        logger.info(f"Resumed from step {self.current_step}, phase {meta['phase']}")

    def evaluate(self, num_samples: int = 1000) -> dict[str, float]:
        """Evaluate generation quality."""
        self.weaver.eval()
        self.witness.eval()

        total_judge_correct = 0
        total_tens_correct = 0
        total_ones_correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(num_samples // 64 + 1):
                batch_size = min(64, num_samples - total)
                if batch_size <= 0:
                    break

                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                labels = torch.randint(0, 100, (batch_size,), device=self.device)
                images, _ = self.weaver(z, labels)

                # Judge evaluation
                judge_logits = self.judge(images, mode="full")
                preds = judge_logits.argmax(dim=1)

                total_judge_correct += (preds == labels).sum().item()

                # Per-digit accuracy
                tens = labels // 10
                ones = labels % 10
                pred_tens = preds // 10
                pred_ones = preds % 10
                total_tens_correct += (pred_tens == tens).sum().item()
                total_ones_correct += (pred_ones == ones).sum().item()
                total += batch_size

        return {
            "eval/judge_accuracy": total_judge_correct / total,
            "eval/tens_accuracy": total_tens_correct / total,
            "eval/ones_accuracy": total_ones_correct / total,
        }
