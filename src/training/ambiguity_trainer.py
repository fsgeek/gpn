"""
Trainer for ambiguous examples with min-to-any loss.

For unambiguous examples: standard cross-entropy
For ambiguous examples: loss = min(CE(output, valid_1), CE(output, valid_2), ...)

This trains the model to produce ANY valid interpretation, not necessarily all of them.
If appropriate uncertainty emerges (model produces different valid outputs across samples),
that's interesting. If it collapses to one interpretation, that's also informative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.data.scan_lite import (
    SCANExample, SCANLiteDataset, get_ambiguous_examples,
    get_compositional_split, get_command_vocab_size, get_action_vocab_size,
    tokenize_actions, detokenize_actions, ACTION_TO_IDX, PAD_TOKEN,
    NUM_ACTIONS, NUM_MODIFIERS, NUM_COUNTS,
)
from src.models.transformer_weaver import create_transformer_weaver
from src.models.staged_witness import create_staged_witness
from src.models.scan_judge import create_scan_judge


@dataclass
class AmbiguityTrainingConfig:
    """Configuration for ambiguity-aware training."""
    total_steps: int = 1000
    batch_size: int = 16
    lr: float = 1e-4
    eval_every: int = 100
    # Loss weights
    generation_weight: float = 1.0
    grounding_weight: float = 0.5


def collate_ambiguous(batch: List[Dict]) -> Dict:
    """Custom collate function for variable-sized valid_outputs."""
    # Stack standard tensors
    commands = torch.stack([b['command'] for b in batch])
    actions = torch.stack([b['actions'] for b in batch])
    num_valid = torch.tensor([b['num_valid'] for b in batch])
    is_ambiguous = torch.tensor([b['is_ambiguous'] for b in batch])
    example_idx = torch.tensor([b['example_idx'] for b in batch])

    # Pad valid_outputs to max size in batch
    max_valid = max(b['valid_outputs'].size(0) for b in batch)
    seq_len = batch[0]['valid_outputs'].size(1)

    valid_outputs_padded = []
    for b in batch:
        vo = b['valid_outputs']
        if vo.size(0) < max_valid:
            # Pad with copies of first valid output (will be masked by num_valid)
            padding = vo[0:1].repeat(max_valid - vo.size(0), 1)
            vo = torch.cat([vo, padding], dim=0)
        valid_outputs_padded.append(vo)

    valid_outputs = torch.stack(valid_outputs_padded)

    return {
        'command': commands,
        'actions': actions,
        'valid_outputs': valid_outputs,
        'num_valid': num_valid,
        'is_ambiguous': is_ambiguous,
        'example_idx': example_idx,
    }


class AmbiguousDataset(Dataset):
    """Dataset that includes ambiguous examples with valid_outputs."""

    def __init__(
        self,
        examples: List[SCANExample],
        max_cmd_len: int = 10,  # Slightly longer for compound commands
        max_action_len: int = 32,
    ):
        self.examples = examples
        self.max_cmd_len = max_cmd_len
        self.max_action_len = max_action_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]

        # Tokenize command
        from src.data.scan_lite import tokenize_command, COMMAND_TO_IDX, PAD_TOKEN as CMD_PAD
        cmd_tokens = tokenize_command(ex.command)
        cmd_padded = cmd_tokens[:self.max_cmd_len]
        cmd_padded = cmd_padded + [COMMAND_TO_IDX[CMD_PAD]] * (self.max_cmd_len - len(cmd_padded))

        # Tokenize primary action sequence
        action_tokens = tokenize_actions(ex.action_sequence)
        action_padded = action_tokens[:self.max_action_len]
        action_padded = action_padded + [ACTION_TO_IDX[PAD_TOKEN]] * (self.max_action_len - len(action_padded))

        # Tokenize all valid outputs if ambiguous
        if ex.is_ambiguous and ex.valid_outputs:
            valid_tokenized = []
            for valid in ex.valid_outputs:
                tokens = tokenize_actions(valid)
                padded = tokens[:self.max_action_len]
                padded = padded + [ACTION_TO_IDX[PAD_TOKEN]] * (self.max_action_len - len(padded))
                valid_tokenized.append(padded)
            # Stack into tensor
            valid_outputs = torch.tensor(valid_tokenized, dtype=torch.long)
            num_valid = len(ex.valid_outputs)
        else:
            # Single valid output
            valid_outputs = torch.tensor([action_padded], dtype=torch.long)
            num_valid = 1

        return {
            'command': torch.tensor(cmd_padded, dtype=torch.long),
            'actions': torch.tensor(action_padded, dtype=torch.long),
            'valid_outputs': valid_outputs,  # (num_valid, seq_len)
            'num_valid': num_valid,
            'is_ambiguous': ex.is_ambiguous,
            'example_idx': idx,
        }


def compute_min_ce_loss(
    logits: torch.Tensor,
    valid_outputs: torch.Tensor,
    num_valid: int,
    pad_idx: int = 0,
) -> torch.Tensor:
    """
    Compute minimum cross-entropy loss over all valid outputs.

    Args:
        logits: (seq_len, vocab_size) model output logits
        valid_outputs: (num_valid, seq_len) valid target sequences
        num_valid: number of valid outputs
        pad_idx: padding index to ignore

    Returns:
        Minimum CE loss across all valid interpretations
    """
    losses = []

    for i in range(num_valid):
        target = valid_outputs[i]  # (seq_len,)

        # Compute per-token CE loss
        ce = F.cross_entropy(
            logits,
            target,
            ignore_index=pad_idx,
            reduction='mean',
        )
        losses.append(ce)

    # Return minimum loss
    return torch.min(torch.stack(losses))


class AmbiguityTrainer:
    """
    Trainer that handles ambiguous examples with min-to-any loss.

    Uses full-perception Witness (stage 3 from start) since that worked best.
    """

    def __init__(
        self,
        weaver: nn.Module,
        witness: nn.Module,
        judge,
        train_examples: List[SCANExample],
        config: AmbiguityTrainingConfig,
        device: str = 'cpu',
    ):
        self.weaver = weaver.to(device)
        self.witness = witness.to(device)
        self.judge = judge
        self.config = config
        self.device = device

        # Ensure witness is at stage 3 (full perception)
        if hasattr(self.witness, 'set_stage'):
            self.witness.set_stage(3)

        # Dataset and loader
        self.train_examples = train_examples
        self.dataset = AmbiguousDataset(train_examples)
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_ambiguous,
        )
        self.loader_iter = iter(self.loader)

        # Optimizers
        self.weaver_optim = torch.optim.Adam(self.weaver.parameters(), lr=config.lr)
        self.witness_optim = torch.optim.Adam(self.witness.parameters(), lr=config.lr)

        # Tracking
        self.step = 0
        self.pad_idx = ACTION_TO_IDX[PAD_TOKEN]

    def _get_batch(self) -> Dict:
        """Get next batch, cycling through data."""
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.loader)
            batch = next(self.loader_iter)
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def train_step(self) -> Dict[str, float]:
        """Execute one training step."""
        self.weaver.train()
        self.witness.train()

        batch = self._get_batch()
        commands = batch['command']
        actions = batch['actions']
        valid_outputs = batch['valid_outputs']  # List of (num_valid, seq_len) per example
        num_valid = batch['num_valid']  # Tensor of counts
        is_ambiguous = batch['is_ambiguous']

        batch_size = commands.size(0)

        # Teacher forcing: Weaver generates from (command, actions[:-1]) -> actions[1:]
        tgt_input = actions[:, :-1]  # Remove last token
        tgt_output = actions[:, 1:]  # Shift by 1

        # Forward pass
        logits, v_pred = self.weaver(commands, tgt_input)

        # Compute generation loss with min-to-any for ambiguous examples
        gen_loss = torch.tensor(0.0, device=self.device)

        for i in range(batch_size):
            example_logits = logits[i]  # (seq_len-1, vocab_size)
            example_valid = valid_outputs[i]  # (num_valid, seq_len)
            example_num_valid = num_valid[i].item()

            # Shift valid outputs for teacher forcing comparison
            valid_shifted = example_valid[:, 1:]  # (num_valid, seq_len-1)

            if example_num_valid > 1:
                # Ambiguous: min-to-any loss
                loss_i = compute_min_ce_loss(
                    example_logits,
                    valid_shifted[:example_num_valid],
                    example_num_valid,
                    self.pad_idx,
                )
            else:
                # Unambiguous: standard CE
                loss_i = F.cross_entropy(
                    example_logits,
                    valid_shifted[0],
                    ignore_index=self.pad_idx,
                    reduction='mean',
                )

            gen_loss = gen_loss + loss_i

        gen_loss = gen_loss / batch_size

        # Witness grounding loss (predict command structure from actions)
        witness_out = self.witness(actions)
        # For now, simple grounding on action component
        action_logits = witness_out['action_logits']
        # We'd need action labels here - skip for now, focus on generation

        # Total loss
        total_loss = self.config.generation_weight * gen_loss

        # Backward
        self.weaver_optim.zero_grad()
        self.witness_optim.zero_grad()
        total_loss.backward()
        self.weaver_optim.step()
        self.witness_optim.step()

        self.step += 1

        return {
            'loss': total_loss.item(),
            'gen_loss': gen_loss.item(),
            'step': self.step,
        }

    @torch.no_grad()
    def evaluate(
        self,
        examples: List[SCANExample],
        num_samples: int = 1,
    ) -> Dict[str, float]:
        """
        Evaluate on examples, optionally with multiple samples for diversity measurement.

        Args:
            examples: Examples to evaluate
            num_samples: Number of samples per example (for diversity measurement)

        Returns:
            Metrics dict including accuracy and diversity measures
        """
        self.weaver.eval()

        dataset = AmbiguousDataset(examples)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        correct = 0
        total = 0
        interpretations_produced = {i: [] for i in range(len(examples))}

        for batch in loader:
            commands = batch['command'].to(self.device)
            valid_outputs = batch['valid_outputs'][0]  # (num_valid, seq_len)
            num_valid = batch['num_valid'][0].item()
            example_idx = batch['example_idx'][0].item()
            ex = examples[example_idx]

            for sample_idx in range(num_samples):
                # Generate autoregressively
                generated = self._generate(commands)
                gen_str = detokenize_actions(generated[0].tolist())

                # Check against valid outputs
                if ex.is_ambiguous and ex.valid_outputs:
                    is_correct, _, matched = self.judge.judge.evaluate_against_set(
                        gen_str, ex.valid_outputs
                    )
                    interpretations_produced[example_idx].append(matched)
                else:
                    is_correct, _ = self.judge.judge.evaluate_sequence(
                        gen_str, ex.action_sequence
                    )
                    interpretations_produced[example_idx].append(0 if is_correct else -1)

                if is_correct:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0

        # Compute diversity metrics for ambiguous examples
        diversity_metrics = self._compute_diversity(examples, interpretations_produced)

        return {
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            **diversity_metrics,
        }

    def _generate(
        self,
        commands: torch.Tensor,
        max_len: int = 32,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate action sequence autoregressively.

        Args:
            commands: Input commands
            max_len: Maximum sequence length
            temperature: Sampling temperature. 0.0 = greedy (argmax), >0 = sample
        """
        batch_size = commands.size(0)
        device = commands.device

        # Start with SOS token
        sos_idx = ACTION_TO_IDX['<SOS>']
        eos_idx = ACTION_TO_IDX['<EOS>']

        generated = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits, _ = self.weaver(commands, generated)
            next_token_logits = logits[:, -1, :]

            if temperature > 0:
                # Temperature sampling
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS generated (for batch size 1)
            if next_token[0, 0].item() == eos_idx:
                break

        return generated

    def _compute_diversity(
        self,
        examples: List[SCANExample],
        interpretations: Dict[int, List[int]],
    ) -> Dict[str, float]:
        """
        Compute diversity metrics for ambiguous examples.

        Returns:
            Dict with diversity metrics
        """
        ambiguous_coverage = []
        ambiguous_entropy = []

        for i, ex in enumerate(examples):
            if not ex.is_ambiguous or not ex.valid_outputs:
                continue

            interps = interpretations[i]
            num_valid = len(ex.valid_outputs)

            if not interps:
                continue

            # Count valid interpretations produced
            valid_interps = [x for x in interps if x >= 0]
            unique_valid = set(valid_interps)

            # Coverage: fraction of valid interpretations produced
            coverage = len(unique_valid) / num_valid
            ambiguous_coverage.append(coverage)

            # Entropy of distribution over interpretations
            if len(valid_interps) > 1:
                counts = [valid_interps.count(j) for j in range(num_valid)]
                total = sum(counts)
                if total > 0:
                    probs = [c / total for c in counts if c > 0]
                    entropy = -sum(p * np.log(p + 1e-10) for p in probs)
                    max_entropy = np.log(num_valid)
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    ambiguous_entropy.append(normalized_entropy)

        return {
            'mean_coverage': np.mean(ambiguous_coverage) if ambiguous_coverage else 0.0,
            'mean_entropy': np.mean(ambiguous_entropy) if ambiguous_entropy else 0.0,
            'num_ambiguous': len(ambiguous_coverage),
        }


def create_ambiguity_trainer(
    train_examples: List[SCANExample],
    config: Optional[AmbiguityTrainingConfig] = None,
    device: str = 'cpu',
) -> AmbiguityTrainer:
    """Factory function for ambiguity-aware trainer."""
    if config is None:
        config = AmbiguityTrainingConfig()

    cmd_vocab = get_command_vocab_size()
    action_vocab = get_action_vocab_size()

    weaver = create_transformer_weaver(
        src_vocab_size=cmd_vocab,
        tgt_vocab_size=action_vocab,
        device=device,
    )

    witness = create_staged_witness(
        action_vocab_size=action_vocab,
        num_actions=NUM_ACTIONS,
        num_modifiers=NUM_MODIFIERS,
        num_counts=NUM_COUNTS,
        device=device,
    )

    judge = create_scan_judge()

    return AmbiguityTrainer(
        weaver=weaver,
        witness=witness,
        judge=judge,
        train_examples=train_examples,
        config=config,
        device=device,
    )


if __name__ == '__main__':
    print("Ambiguity Trainer Test")
    print("=" * 50)

    # Get regular examples + ambiguous
    train_examples, _ = get_compositional_split()
    ambiguous = get_ambiguous_examples()

    print(f"Training examples: {len(train_examples)}")
    print(f"Ambiguous examples: {len(ambiguous)}")

    # Combine for training
    all_examples = train_examples + ambiguous
    print(f"Total: {len(all_examples)}")

    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = AmbiguityTrainingConfig(total_steps=50, batch_size=8)
    trainer = create_ambiguity_trainer(all_examples, config, device)

    print("\nTraining for 50 steps...")
    for step in range(50):
        metrics = trainer.train_step()
        if step % 10 == 0:
            print(f"  Step {step}: loss={metrics['loss']:.4f}")

    print("\nEvaluating on ambiguous examples (10 samples each)...")
    eval_metrics = trainer.evaluate(ambiguous, num_samples=10)
    print(f"  Accuracy: {eval_metrics['accuracy']:.3f}")
    print(f"  Mean coverage: {eval_metrics['mean_coverage']:.3f}")
    print(f"  Mean entropy: {eval_metrics['mean_entropy']:.3f}")
