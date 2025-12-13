"""
Staged Curriculum Trainer for seq2seq compositional generalization.

The complete pedagogical system:
1. Curriculum structure: What order to present material (stages)
2. Mastery gating: When to advance (accuracy thresholds)
3. Staged perception: Witness develops perception alongside curriculum

Stages:
  Stage 1: Primitives only (walk, run, jump, look)
  Stage 2: Action + modifier (walk left, jump around, etc.)
  Stage 3: Full composition (walk left twice, jump around thrice, etc.)

The hypothesis: Witness's perceptual capacity is the bottleneck for what it can teach.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.models.transformer_weaver import TransformerWeaver
from src.models.staged_witness import StagedTransformerWitness
from src.models.scan_judge import SCANJudgeWrapper
from src.data.scan_lite import (
    ACTION_TO_IDX, COMMAND_TO_IDX,
    SOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    SCANLiteDataset, get_compositional_split,
    NUM_ACTIONS, NUM_MODIFIERS, NUM_COUNTS,
)

logger = logging.getLogger(__name__)


@dataclass
class StagedCurriculumConfig:
    """Configuration for staged curriculum training."""
    # Mastery thresholds for stage advancement
    stage1_mastery_threshold: float = 0.85  # Action accuracy to advance to stage 2
    stage2_mastery_threshold: float = 0.80  # Action + modifier accuracy to advance to stage 3

    # Minimum steps per stage (don't advance too early even if threshold met)
    min_steps_per_stage: int = 200

    # Maximum steps per stage (advance even if threshold not met)
    max_steps_per_stage: int = 1500

    # Total training steps
    total_steps: int = 3000

    # Loss weights (constant across stages - the curriculum IS the structure)
    grounding_weight: float = 1.0
    alignment_weight: float = 0.3
    empowerment_weight: float = 0.1

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Generation
    max_gen_len: int = 32

    # EMA for empowerment
    ema_decay: float = 0.99

    # Evaluation frequency for mastery check
    eval_every: int = 50


def filter_dataset_by_stage(
    examples: List,
    stage: int,
) -> List:
    """
    Filter examples based on curriculum stage.

    Stage 1: Primitives only (no modifier, no count)
    Stage 2: Action + optional modifier (no count, or no modifier)
    Stage 3: All examples
    """
    if stage == 1:
        # Primitives only: no modifier AND no count
        return [ex for ex in examples if ex.modifier is None and ex.count is None]
    elif stage == 2:
        # Single composition: has modifier XOR count, but not both
        # Also include primitives for continued practice
        return [ex for ex in examples if ex.count is None]  # No count restriction
    else:
        # Stage 3: all examples
        return examples


class StagedCurriculumTrainer:
    """
    Complete pedagogical system for seq2seq with staged curriculum.

    The three components work together:
    1. Curriculum (filter_dataset_by_stage): Controls what examples are presented
    2. Mastery gating (_check_mastery): Controls when to advance
    3. Staged perception (StagedTransformerWitness): Witness perception develops with curriculum
    """

    def __init__(
        self,
        weaver: TransformerWeaver,
        witness: StagedTransformerWitness,
        judge: SCANJudgeWrapper,
        train_examples: List,
        config: StagedCurriculumConfig,
        device: str = 'cpu',
    ):
        self.weaver = weaver
        self.witness = witness
        self.judge = judge
        self.train_examples = train_examples
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

        # EMA state
        self.ema_v_seen_mean = None
        self.ema_v_seen_var = None

        # Training state
        self.step = 0
        self.stage = 1
        self.steps_in_current_stage = 0

        # Stage-specific dataloaders (created lazily)
        self._stage_dataloaders = {}
        self._stage_datasets = {}

        # Token indices
        self.sos_idx = ACTION_TO_IDX[SOS_TOKEN]
        self.eos_idx = ACTION_TO_IDX[EOS_TOKEN]
        self.pad_idx = ACTION_TO_IDX[PAD_TOKEN]

        # Metrics history for mastery checking
        self.stage_metrics = {1: [], 2: [], 3: []}

        # Initialize stage 1
        self._setup_stage(1)

    def _setup_stage(self, stage: int):
        """Set up dataloader and Witness for a stage."""
        self.stage = stage
        self.steps_in_current_stage = 0

        # Filter training data for this stage
        filtered = filter_dataset_by_stage(self.train_examples, stage)
        logger.info(f"Stage {stage}: {len(filtered)} training examples")

        # Create dataset and dataloader
        dataset = SCANLiteDataset(filtered)
        self._stage_datasets[stage] = dataset
        self._stage_dataloaders[stage] = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
        )
        self._current_loader_iter = iter(self._stage_dataloaders[stage])

        # Set Witness perception stage
        self.witness.set_stage(stage)
        logger.info(f"Witness perception heads: {self.witness.get_active_heads()}")

    def _get_batch(self) -> Dict[str, torch.Tensor]:
        """Get next batch, cycling through dataloader."""
        try:
            return next(self._current_loader_iter)
        except StopIteration:
            self._current_loader_iter = iter(self._stage_dataloaders[self.stage])
            return next(self._current_loader_iter)

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

    def _compute_grounding_loss(
        self,
        witness_output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute grounding loss based on active perception heads.

        This is the key pedagogical mechanism: Witness is grounded on what it
        can currently perceive, which develops with the curriculum.
        """
        losses = []

        # Stage 1+: Action perception
        action_logits = witness_output['action_logits']
        action_target = batch['action_idx'].to(self.device)
        losses.append(F.cross_entropy(action_logits, action_target))

        # Stage 2+: Modifier perception
        if 'modifier_logits' in witness_output:
            modifier_logits = witness_output['modifier_logits']
            modifier_target = batch['modifier_idx'].to(self.device)
            losses.append(F.cross_entropy(modifier_logits, modifier_target))

        # Stage 3: Count perception
        if 'count_logits' in witness_output:
            count_logits = witness_output['count_logits']
            count_target = batch['count_idx'].to(self.device)
            losses.append(F.cross_entropy(count_logits, count_target))

        return sum(losses) / len(losses)

    def _compute_perception_accuracy(
        self,
        witness_output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute accuracy for each active perception head."""
        accuracies = {}

        with torch.no_grad():
            # Action accuracy
            action_pred = witness_output['action_logits'].argmax(dim=-1)
            action_target = batch['action_idx'].to(self.device)
            accuracies['action_acc'] = (action_pred == action_target).float().mean().item()

            # Modifier accuracy (stage 2+)
            if 'modifier_logits' in witness_output:
                modifier_pred = witness_output['modifier_logits'].argmax(dim=-1)
                modifier_target = batch['modifier_idx'].to(self.device)
                accuracies['modifier_acc'] = (modifier_pred == modifier_target).float().mean().item()

            # Count accuracy (stage 3)
            if 'count_logits' in witness_output:
                count_pred = witness_output['count_logits'].argmax(dim=-1)
                count_target = batch['count_idx'].to(self.device)
                accuracies['count_acc'] = (count_pred == count_target).float().mean().item()

        return accuracies

    def _check_mastery(self) -> bool:
        """
        Check if current stage mastery threshold is met.

        Returns True if should advance to next stage.
        """
        if self.stage >= 3:
            return False  # Already at final stage

        if self.steps_in_current_stage < self.config.min_steps_per_stage:
            return False  # Too early

        if self.steps_in_current_stage >= self.config.max_steps_per_stage:
            logger.info(f"Stage {self.stage}: Max steps reached, advancing regardless of mastery")
            return True

        # Check recent metrics
        recent_metrics = self.stage_metrics[self.stage][-10:]  # Last 10 evaluations
        if len(recent_metrics) < 5:
            return False  # Not enough data

        # Compute average accuracy for mastery check
        if self.stage == 1:
            avg_acc = sum(m.get('action_acc', 0) for m in recent_metrics) / len(recent_metrics)
            threshold = self.config.stage1_mastery_threshold
        else:  # stage 2
            # Both action and modifier must be high
            avg_action = sum(m.get('action_acc', 0) for m in recent_metrics) / len(recent_metrics)
            avg_modifier = sum(m.get('modifier_acc', 0) for m in recent_metrics) / len(recent_metrics)
            avg_acc = min(avg_action, avg_modifier)  # Both must meet threshold
            threshold = self.config.stage2_mastery_threshold

        if avg_acc >= threshold:
            logger.info(f"Stage {self.stage}: Mastery achieved ({avg_acc:.3f} >= {threshold})")
            return True

        return False

    def train_step(self) -> Dict[str, float]:
        """Single training step with staged curriculum."""
        batch = self._get_batch()

        command = batch['command'].to(self.device)
        actions = batch['actions'].to(self.device)

        # Teacher forcing
        tgt_input = actions[:, :-1]
        tgt_output = actions[:, 1:]

        # Weaver forward
        logits, v_pred = self.weaver(command, tgt_input)

        # Generation loss
        gen_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=self.pad_idx,
        )

        # Generate for Witness and Judge
        with torch.no_grad():
            generated, _ = self.weaver.generate(
                command,
                max_len=self.config.max_gen_len,
                sos_idx=self.sos_idx,
                eos_idx=self.eos_idx,
            )

        # Witness evaluation (with staged perception)
        witness_output = self.witness(generated)
        v_seen = witness_output['v_seen']
        self._update_ema(v_seen.detach())

        # Judge correctness
        judge_correctness = self.judge.forward(generated, actions)

        # === Losses ===

        # Grounding loss: Witness learns to perceive compositional structure
        # Uses ground truth actions so Witness sees correct patterns
        witness_gt_output = self.witness(actions)
        grounding_loss = self._compute_grounding_loss(witness_gt_output, batch)

        # Alignment loss: v_pred should match v_seen
        alignment_loss = F.mse_loss(v_pred, v_seen.detach())

        # Empowerment loss
        if self.ema_v_seen_var is not None and self.config.empowerment_weight > 0:
            current_var = v_seen.var(dim=0)
            empowerment_loss = F.relu(self.ema_v_seen_var - current_var).mean()
        else:
            empowerment_loss = torch.tensor(0.0, device=self.device)

        # Total loss
        total_loss = (
            gen_loss +
            self.config.grounding_weight * grounding_loss +
            self.config.alignment_weight * alignment_loss +
            self.config.empowerment_weight * empowerment_loss
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
        self.steps_in_current_stage += 1

        # Compute perception accuracies
        perception_acc = self._compute_perception_accuracy(witness_gt_output, batch)

        # Store metrics for mastery checking
        if self.step % self.config.eval_every == 0:
            self.stage_metrics[self.stage].append(perception_acc)

            # Check for stage advancement
            if self._check_mastery():
                self._setup_stage(self.stage + 1)

        return {
            'total_loss': total_loss.item(),
            'gen_loss': gen_loss.item(),
            'grounding_loss': grounding_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'empowerment_loss': empowerment_loss.item() if isinstance(empowerment_loss, torch.Tensor) else empowerment_loss,
            'judge_correctness': judge_correctness.mean().item(),
            'stage': self.stage,
            **perception_acc,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on a dataset (typically held-out test set)."""
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


class Stage1OnlyTrainer(StagedCurriculumTrainer):
    """
    Control condition: Stage 1 perception only (like Phase-1-only in MNIST).

    Witness only ever learns to perceive actions (primitives).
    Curriculum progresses but perception stays at stage 1.

    Prediction: Should scaffold primitives but fail on composition
    because Witness can't perceive compositional structure.
    """

    def _setup_stage(self, stage: int):
        """Override: Keep Witness at stage 1 even as curriculum advances."""
        self.stage = stage
        self.steps_in_current_stage = 0

        # Filter training data for curriculum stage
        filtered = filter_dataset_by_stage(self.train_examples, stage)
        logger.info(f"Stage {stage} (Stage1Only): {len(filtered)} training examples")

        # Create dataset and dataloader
        dataset = SCANLiteDataset(filtered)
        self._stage_datasets[stage] = dataset
        self._stage_dataloaders[stage] = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
        )
        self._current_loader_iter = iter(self._stage_dataloaders[stage])

        # KEY DIFFERENCE: Witness stays at stage 1
        self.witness.set_stage(1)
        logger.info(f"Witness perception LOCKED at stage 1: {self.witness.get_active_heads()}")


class FinalStageFromStartTrainer(StagedCurriculumTrainer):
    """
    Control condition: Final stage perception from the start.

    Witness tries to perceive full compositional structure immediately,
    without the staged development.

    Prediction: Should fail (parallel to Phase-1-only catastrophic failure)
    because trying to perceive everything at once prevents learning any of it well.
    """

    def _setup_stage(self, stage: int):
        """Override: Start Witness at stage 3 immediately."""
        self.stage = stage
        self.steps_in_current_stage = 0

        # Use ALL training data from the start
        filtered = self.train_examples  # No filtering
        logger.info(f"Stage {stage} (FinalFromStart): {len(filtered)} training examples (all)")

        # Create dataset and dataloader
        dataset = SCANLiteDataset(filtered)
        self._stage_datasets[stage] = dataset
        self._stage_dataloaders[stage] = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
        )
        self._current_loader_iter = iter(self._stage_dataloaders[stage])

        # KEY DIFFERENCE: Witness starts at stage 3 (full perception)
        self.witness.set_stage(3)
        logger.info(f"Witness perception at FULL from start: {self.witness.get_active_heads()}")

    def _check_mastery(self) -> bool:
        """No stage advancement - already at max."""
        return False
