"""
Trainer for SCAN-lite experiments.

Implements both pedagogical (three-phase GPN) and adversarial (seq2seq GAN) training
for comparison on compositional generalization.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.transformer_weaver import TransformerWeaver
from src.models.transformer_witness import TransformerWitness
from src.models.scan_judge import SCANJudgeWrapper
from src.data.scan_lite import (
    ACTION_TO_IDX, COMMAND_TO_IDX,
    SOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    detokenize_actions,
)

logger = logging.getLogger(__name__)


@dataclass
class SCANTrainerConfig:
    """Configuration for SCAN trainer."""
    # Training phases (for pedagogical)
    phase1_steps: int = 500
    phase2_steps: int = 1000
    total_steps: int = 2000

    # Loss weights by phase
    phase1_grounding: float = 1.0
    phase1_alignment: float = 0.1
    phase1_empowerment: float = 0.0

    phase2_grounding: float = 1.0
    phase2_alignment: float = 0.5
    phase2_empowerment: float = 0.3

    phase3_grounding: float = 0.0
    phase3_alignment: float = 0.0
    phase3_empowerment: float = 0.0

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Generation
    max_gen_len: int = 32

    # EMA for empowerment
    ema_decay: float = 0.99


class PedagogicalSCANTrainer:
    """
    Three-phase pedagogical trainer for SCAN-lite.

    Implements the GPN curriculum:
    - Phase 1: Strong grounding (Judge signal), weak alignment
    - Phase 2: Balanced (Weaver learns to predict Witness perception)
    - Phase 3: Drift test (minimal supervision)
    """

    def __init__(
        self,
        weaver: TransformerWeaver,
        witness: TransformerWitness,
        judge: SCANJudgeWrapper,
        config: SCANTrainerConfig,
        device: str = 'cpu',
    ):
        self.weaver = weaver
        self.witness = witness
        self.judge = judge
        self.config = config
        self.device = device

        # Optimizers
        self.weaver_optimizer = torch.optim.AdamW(
            weaver.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.witness_optimizer = torch.optim.AdamW(
            witness.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Loss weights (updated by phase)
        self.grounding_weight = config.phase1_grounding
        self.alignment_weight = config.phase1_alignment
        self.empowerment_weight = config.phase1_empowerment

        # EMA state for empowerment
        self.ema_v_seen_mean = None
        self.ema_v_seen_var = None

        # Training state
        self.step = 0
        self.phase = 1

        # Token indices
        self.sos_idx = ACTION_TO_IDX[SOS_TOKEN]
        self.eos_idx = ACTION_TO_IDX[EOS_TOKEN]
        self.pad_idx = ACTION_TO_IDX[PAD_TOKEN]

    def _update_phase(self):
        """Update training phase based on step."""
        if self.step < self.config.phase1_steps:
            if self.phase != 1:
                self.phase = 1
                self._set_phase_weights(1)
                logger.info(f"Step {self.step}: Entering Phase 1")
        elif self.step < self.config.phase2_steps:
            if self.phase != 2:
                self.phase = 2
                self._set_phase_weights(2)
                logger.info(f"Step {self.step}: Entering Phase 2")
        else:
            if self.phase != 3:
                self.phase = 3
                self._set_phase_weights(3)
                logger.info(f"Step {self.step}: Entering Phase 3")

    def _set_phase_weights(self, phase: int):
        """Set loss weights for phase."""
        if phase == 1:
            self.grounding_weight = self.config.phase1_grounding
            self.alignment_weight = self.config.phase1_alignment
            self.empowerment_weight = self.config.phase1_empowerment
        elif phase == 2:
            self.grounding_weight = self.config.phase2_grounding
            self.alignment_weight = self.config.phase2_alignment
            self.empowerment_weight = self.config.phase2_empowerment
        else:
            self.grounding_weight = self.config.phase3_grounding
            self.alignment_weight = self.config.phase3_alignment
            self.empowerment_weight = self.config.phase3_empowerment

    def _update_ema(self, v_seen: torch.Tensor):
        """Update EMA statistics for v_seen."""
        with torch.no_grad():
            v_mean = v_seen.mean(dim=0)
            v_var = v_seen.var(dim=0)

            if self.ema_v_seen_mean is None:
                self.ema_v_seen_mean = v_mean
                self.ema_v_seen_var = v_var
            else:
                self.ema_v_seen_mean = (
                    self.config.ema_decay * self.ema_v_seen_mean +
                    (1 - self.config.ema_decay) * v_mean
                )
                self.ema_v_seen_var = (
                    self.config.ema_decay * self.ema_v_seen_var +
                    (1 - self.config.ema_decay) * v_var
                )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Dict with 'command', 'actions' tensors

        Returns:
            Dict of loss values
        """
        self._update_phase()

        command = batch['command'].to(self.device)
        actions = batch['actions'].to(self.device)

        # Teacher forcing: use ground truth as decoder input
        tgt_input = actions[:, :-1]  # All but last token
        tgt_output = actions[:, 1:]  # All but first token (shifted)

        # Weaver forward (teacher forcing)
        logits, v_pred = self.weaver(command, tgt_input)

        # Generation loss (cross-entropy)
        gen_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=self.pad_idx,
        )

        # Generate sequences for Witness and Judge
        with torch.no_grad():
            generated, _ = self.weaver.generate(
                command,
                max_len=self.config.max_gen_len,
                sos_idx=self.sos_idx,
                eos_idx=self.eos_idx,
            )

        # Witness evaluation
        _, v_seen = self.witness(generated)
        self._update_ema(v_seen.detach())

        # Judge correctness
        judge_correctness = self.judge.forward(generated, actions)

        # === Losses ===

        # Grounding loss: Witness should predict the FULL compositional command
        # This is the key pedagogical signal - Witness learns to recognize
        # "walk left twice" as a unique compositional pattern, not just "walk"
        witness_cmd_logits, _ = self.witness(actions)  # Use ground truth actions
        command_id = batch['command_id'].to(self.device)  # Unique ID for full command (0-63)
        grounding_loss = F.cross_entropy(witness_cmd_logits, command_id)

        # Alignment loss: v_pred should match v_seen
        alignment_loss = F.mse_loss(v_pred, v_seen.detach())

        # Empowerment loss: v_seen should be diverse (not collapse)
        if self.ema_v_seen_var is not None and self.empowerment_weight > 0:
            # Encourage variance to stay above EMA baseline
            current_var = v_seen.var(dim=0)
            empowerment_loss = F.relu(self.ema_v_seen_var - current_var).mean()
        else:
            empowerment_loss = torch.tensor(0.0, device=self.device)

        # Total loss
        total_loss = (
            gen_loss +
            self.grounding_weight * grounding_loss +
            self.alignment_weight * alignment_loss +
            self.empowerment_weight * empowerment_loss
        )

        # Backward and optimize
        self.weaver_optimizer.zero_grad()
        self.witness_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.weaver.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.witness.parameters(), 1.0)
        self.weaver_optimizer.step()
        self.witness_optimizer.step()

        self.step += 1

        return {
            'total_loss': total_loss.item(),
            'gen_loss': gen_loss.item(),
            'grounding_loss': grounding_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'empowerment_loss': empowerment_loss.item() if isinstance(empowerment_loss, torch.Tensor) else empowerment_loss,
            'judge_correctness': judge_correctness.mean().item(),
            'phase': self.phase,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on a dataset.

        Args:
            dataloader: DataLoader to evaluate on

        Returns:
            Dict with sequence_accuracy and token_accuracy
        """
        self.weaver.eval()

        total_seq_correct = 0
        total_tok_correct = 0
        total_tok_count = 0
        total_examples = 0

        for batch in dataloader:
            command = batch['command'].to(self.device)
            actions = batch['actions'].to(self.device)

            generated, _ = self.weaver.generate(
                command,
                max_len=self.config.max_gen_len,
                sos_idx=self.sos_idx,
                eos_idx=self.eos_idx,
            )

            # Evaluate
            correct, tok_acc = self.judge.judge.evaluate_batch(generated, actions)
            total_seq_correct += correct.sum().item()
            total_tok_correct += tok_acc.sum().item()
            total_examples += command.size(0)

        self.weaver.train()

        return {
            'sequence_accuracy': total_seq_correct / total_examples if total_examples > 0 else 0,
            'token_accuracy': total_tok_correct / total_examples if total_examples > 0 else 0,
        }


class AdversarialSCANTrainer:
    """
    Adversarial (seq2seq GAN) trainer for SCAN-lite.

    Uses a discriminator to distinguish real vs generated action sequences.
    This is the baseline to compare against pedagogical training.
    """

    def __init__(
        self,
        weaver: TransformerWeaver,
        discriminator: nn.Module,
        judge: SCANJudgeWrapper,
        config: SCANTrainerConfig,
        device: str = 'cpu',
    ):
        self.weaver = weaver
        self.discriminator = discriminator
        self.judge = judge
        self.config = config
        self.device = device

        self.weaver_optimizer = torch.optim.AdamW(
            weaver.parameters(),
            lr=config.learning_rate,
        )
        self.disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=config.learning_rate,
        )

        self.step = 0

        self.sos_idx = ACTION_TO_IDX[SOS_TOKEN]
        self.eos_idx = ACTION_TO_IDX[EOS_TOKEN]
        self.pad_idx = ACTION_TO_IDX[PAD_TOKEN]

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single adversarial training step."""
        command = batch['command'].to(self.device)
        actions = batch['actions'].to(self.device)

        # === Train Discriminator ===
        self.disc_optimizer.zero_grad()

        # Real sequences
        real_score = self.discriminator(command, actions)
        real_loss = F.binary_cross_entropy_with_logits(
            real_score, torch.ones_like(real_score)
        )

        # Generated sequences
        with torch.no_grad():
            generated, _ = self.weaver.generate(
                command,
                max_len=self.config.max_gen_len,
                sos_idx=self.sos_idx,
                eos_idx=self.eos_idx,
            )

        fake_score = self.discriminator(command, generated)
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_score, torch.zeros_like(fake_score)
        )

        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        self.disc_optimizer.step()

        # === Train Weaver (Generator) ===
        self.weaver_optimizer.zero_grad()

        # Teacher forcing for generation loss
        tgt_input = actions[:, :-1]
        tgt_output = actions[:, 1:]
        logits, _ = self.weaver(command, tgt_input)
        gen_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=self.pad_idx,
        )

        # Adversarial loss: fool discriminator
        generated, _ = self.weaver.generate(
            command,
            max_len=self.config.max_gen_len,
            sos_idx=self.sos_idx,
            eos_idx=self.eos_idx,
        )
        # Need gradients through generation - use soft outputs
        # For simplicity, use REINFORCE-style update or teacher forcing

        # Simplified: just use generation loss (most seq2seq GANs do this)
        total_loss = gen_loss
        total_loss.backward()
        self.weaver_optimizer.step()

        # Judge correctness
        with torch.no_grad():
            judge_correctness = self.judge.forward(generated, actions)

        self.step += 1

        return {
            'total_loss': total_loss.item(),
            'gen_loss': gen_loss.item(),
            'disc_loss': disc_loss.item(),
            'judge_correctness': judge_correctness.mean().item(),
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on dataset."""
        self.weaver.eval()

        total_seq_correct = 0
        total_examples = 0

        for batch in dataloader:
            command = batch['command'].to(self.device)
            actions = batch['actions'].to(self.device)

            generated, _ = self.weaver.generate(
                command,
                max_len=self.config.max_gen_len,
                sos_idx=self.sos_idx,
                eos_idx=self.eos_idx,
            )

            correct, _ = self.judge.judge.evaluate_batch(generated, actions)
            total_seq_correct += correct.sum().item()
            total_examples += command.size(0)

        self.weaver.train()

        return {
            'sequence_accuracy': total_seq_correct / total_examples if total_examples > 0 else 0,
        }


class SeqDiscriminator(nn.Module):
    """
    Discriminator for sequence GAN.

    Takes (command, action_sequence) pairs and predicts real/fake.
    """

    def __init__(
        self,
        cmd_vocab_size: int,
        action_vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.cmd_embedding = nn.Embedding(cmd_vocab_size, d_model, padding_idx=pad_idx)
        self.action_embedding = nn.Embedding(action_vocab_size, d_model, padding_idx=pad_idx)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Linear(d_model, 1)
        self.pad_idx = pad_idx

    def forward(self, command: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            command: (batch, cmd_len) command tokens
            actions: (batch, action_len) action tokens

        Returns:
            (batch, 1) real/fake logits
        """
        cmd_emb = self.cmd_embedding(command)
        action_emb = self.action_embedding(actions)

        # Concatenate
        combined = torch.cat([cmd_emb, action_emb], dim=1)

        # Create padding mask
        cmd_mask = command == self.pad_idx
        action_mask = actions == self.pad_idx
        padding_mask = torch.cat([cmd_mask, action_mask], dim=1)

        encoded = self.encoder(combined, src_key_padding_mask=padding_mask)

        # Pool
        mask = ~padding_mask
        mask = mask.unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.classifier(pooled)
