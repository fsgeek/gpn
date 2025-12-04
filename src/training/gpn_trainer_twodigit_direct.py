"""
GPN-2 Trainer (Direct/No-Curriculum Ablation): Train 2-digit from scratch.

This is the critical ablation for GPN-2:
- NO pre-trained single-digit Weaver
- NO curriculum phases
- Direct generation of 28x56 images

If this matches the curriculum version → curriculum doesn't help
If this is worse → curriculum (single-digit mastery first) matters

Based on V3-NoMeta findings: classification grounding + competence gating.
"""

from pathlib import Path
from typing import Optional, Iterator
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.weaver_twodigit import TwoDigitWeaverDirect
from src.models.witness_twodigit import TwoDigitWitness
from src.models.judge_twodigit import TwoDigitJudge
from src.training.ema import EMAState
from src.utils.logging import MetricsLogger
from src.utils.checkpointing import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class GPNTrainerTwoDigitDirect:
    """
    Direct trainer for 2-digit MNIST - NO curriculum, from scratch.

    Ablation to test: is composition free once you have atomic competence,
    or does the curriculum (single-digit → composition) actually help?
    """

    def __init__(
        self,
        weaver: TwoDigitWeaverDirect,
        witness: TwoDigitWitness,
        judge: TwoDigitJudge,
        train_loader: DataLoader,
        device: Optional[torch.device] = None,
        competence_threshold: float = 0.5,
        total_steps: int = 5000,
        latent_dim: int = 64,
        v_pred_dim: int = 16,
    ) -> None:
        self.device = device or torch.device("cpu")

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
        self.total_steps = total_steps

        # Optimizers
        self.weaver_optimizer = optim.Adam(
            self.weaver.parameters(),
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
            dim=v_pred_dim,
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
        """
        Training step - direct generation, no curriculum.
        """
        metrics = {}

        # Get real images and labels
        real_images, labels = self._get_batch()
        batch_size = real_images.size(0)

        # =====================================================================
        # Step 1: Witness Grounding (on real 2-digit images)
        # =====================================================================
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

        # =====================================================================
        # Step 2: Weaver Training (when Witness is competent)
        # =====================================================================
        witness_is_competent = self.witness_competence >= self.competence_threshold

        if witness_is_competent:
            self.weaver.train()
            self.witness.eval()

            # Generate fake 2-digit images (DIRECT - no composition)
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

            # Per-digit accuracy
            tens = labels // 10
            ones = labels % 10
            pred_labels = judge_logits.argmax(dim=1)
            pred_tens = pred_labels // 10
            pred_ones = pred_labels % 10
            tens_accuracy = (pred_tens == tens).float().mean().item()
            ones_accuracy = (pred_ones == ones).float().mean().item()

        metrics["quality/judge_accuracy"] = judge_accuracy
        metrics["quality/witness_accuracy_fake"] = witness_accuracy_fake
        metrics["quality/tens_accuracy"] = tens_accuracy
        metrics["quality/ones_accuracy"] = ones_accuracy
        metrics["training/competence_gated"] = float(witness_is_competent)

        return metrics

    def train(
        self,
        total_steps: Optional[int] = None,
        log_dir: str = "experiments/gpn2_direct",
        resume_from: Optional[str] = None,
    ) -> dict[str, float]:
        total_steps = total_steps or self.total_steps

        self.logger = MetricsLogger(log_dir)

        if resume_from:
            self._load_checkpoint(resume_from)

        logger.info(f"Starting GPN-2 DIRECT (No Curriculum) training from step {self.current_step}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Device: {self.device}")
        logger.info("ABLATION: Training 2-digit from scratch, no pre-trained single-digit Weaver")

        final_metrics = {}
        competence_logged = False

        while self.current_step < total_steps:
            if not competence_logged and self.witness_competence >= self.competence_threshold:
                logger.info(f"Witness reached competence threshold at step {self.current_step}")
                competence_logged = True

            metrics = self.train_step()
            final_metrics = metrics

            if self.current_step % 100 == 0:
                self._log_metrics(metrics)
                gated = "GATED" if metrics["training/competence_gated"] else "waiting"
                logger.info(
                    f"Step {self.current_step} | {gated} | "
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
        self.logger.flush()

    def _log_samples(self) -> None:
        if self.logger is None:
            return
        self.weaver.eval()
        with torch.no_grad():
            num_samples = 20
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
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
            path = checkpoint_dir / f"checkpoint_twodigit_direct_{suffix}.pt"
        else:
            path = checkpoint_dir / f"checkpoint_twodigit_direct_step{self.current_step}.pt"

        save_checkpoint(
            path=path,
            step=self.current_step,
            phase=1,  # No phases in direct version
            models={"weaver": self.weaver, "witness": self.witness},
            optimizers={"weaver": self.weaver_optimizer, "witness": self.witness_optimizer},
            ema_state=self.ema_state.state_dict(),
            metrics={"witness_competence": self.witness_competence},
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
        if "metrics" in meta and "witness_competence" in meta["metrics"]:
            self.witness_competence = meta["metrics"]["witness_competence"]
        logger.info(f"Resumed from step {self.current_step}")

    def evaluate(self, num_samples: int = 1000) -> dict[str, float]:
        """Evaluate generation quality."""
        self.weaver.eval()

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

                judge_logits = self.judge(images, mode="full")
                preds = judge_logits.argmax(dim=1)

                total_judge_correct += (preds == labels).sum().item()

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
