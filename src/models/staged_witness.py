"""
Staged Transformer Witness for progressive compositional perception.

The key insight: Witness's perceptual capacity is the bottleneck for what it can teach.
This Witness develops perception through stages:
  Stage 1: Perceive ACTION only (4 classes)
  Stage 2: Perceive ACTION + MODIFIER (4 + 4 classes)
  Stage 3: Perceive ACTION + MODIFIER + COUNT (4 + 4 + 4 classes)

Each stage adds a new perception head. The curriculum and Witness perception co-develop.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


class StagedTransformerWitness(nn.Module):
    """
    Transformer encoder with staged perception heads.

    Perception develops progressively:
    - Stage 1: action_head only (primitives)
    - Stage 2: action_head + modifier_head (single composition)
    - Stage 3: action_head + modifier_head + count_head (full composition)
    """

    def __init__(
        self,
        action_vocab_size: int,  # Input vocabulary (output actions: WALK, RUN, etc.)
        num_actions: int = 4,     # walk, run, jump, look
        num_modifiers: int = 4,   # None, left, right, around
        num_counts: int = 4,      # None, once, twice, thrice
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 64,
        v_seen_dim: int = 16,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx
        self.num_actions = num_actions
        self.num_modifiers = num_modifiers
        self.num_counts = num_counts

        # Embedding for action sequences
        self.embedding = nn.Embedding(action_vocab_size, d_model, padding_idx=pad_idx)

        # Positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder (shared backbone)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Staged perception heads
        # Stage 1: Action perception
        self.action_head = nn.Linear(d_model, num_actions)

        # Stage 2: Modifier perception (added in stage 2)
        self.modifier_head = nn.Linear(d_model, num_modifiers)

        # Stage 3: Count perception (added in stage 3)
        self.count_head = nn.Linear(d_model, num_counts)

        # v_seen head (always active - represents overall perceived quality)
        self.v_seen_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, v_seen_dim),
        )

        # Current stage (1, 2, or 3)
        self._current_stage = 1

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        return seq == self.pad_idx

    @property
    def current_stage(self) -> int:
        return self._current_stage

    def set_stage(self, stage: int):
        """Set the current perception stage (1, 2, or 3)."""
        if stage not in [1, 2, 3]:
            raise ValueError(f"Stage must be 1, 2, or 3, got {stage}")
        self._current_stage = stage

    def advance_stage(self) -> int:
        """Advance to the next stage if not already at stage 3."""
        if self._current_stage < 3:
            self._current_stage += 1
        return self._current_stage

    def forward(
        self,
        action_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with staged perception.

        Args:
            action_seq: (batch, seq_len) action token indices

        Returns:
            Dictionary containing:
                - action_logits: (batch, num_actions) - always present
                - modifier_logits: (batch, num_modifiers) - stage 2+
                - count_logits: (batch, num_counts) - stage 3 only
                - v_seen: (batch, v_seen_dim) - always present
                - stage: current stage number
        """
        # Embed and add positional encoding
        x = self.embedding(action_seq) * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)

        # Create padding mask
        padding_mask = self._create_padding_mask(action_seq)

        # Encode
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        # Pool over non-padded positions
        mask = ~padding_mask
        mask = mask.unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Build output based on current stage
        output = {
            'stage': self._current_stage,
            'v_seen': self.v_seen_head(pooled),
        }

        # Stage 1+: Action perception (always active)
        output['action_logits'] = self.action_head(pooled)

        # Stage 2+: Modifier perception
        if self._current_stage >= 2:
            output['modifier_logits'] = self.modifier_head(pooled)

        # Stage 3: Count perception
        if self._current_stage >= 3:
            output['count_logits'] = self.count_head(pooled)

        return output

    def get_active_heads(self) -> list:
        """Return list of currently active perception heads."""
        heads = ['action']
        if self._current_stage >= 2:
            heads.append('modifier')
        if self._current_stage >= 3:
            heads.append('count')
        return heads


def create_staged_witness(
    action_vocab_size: int,
    num_actions: int = 4,
    num_modifiers: int = 4,
    num_counts: int = 4,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    v_seen_dim: int = 16,
    device: str = 'cpu',
) -> StagedTransformerWitness:
    """Factory function for StagedTransformerWitness."""
    model = StagedTransformerWitness(
        action_vocab_size=action_vocab_size,
        num_actions=num_actions,
        num_modifiers=num_modifiers,
        num_counts=num_counts,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        v_seen_dim=v_seen_dim,
    )
    return model.to(device)


if __name__ == '__main__':
    from src.data.scan_lite import get_action_vocab_size, NUM_ACTIONS, NUM_MODIFIERS, NUM_COUNTS

    print("StagedTransformerWitness Test")
    print("=" * 50)

    action_vocab = get_action_vocab_size()

    model = create_staged_witness(
        action_vocab,
        num_actions=NUM_ACTIONS,
        num_modifiers=NUM_MODIFIERS,
        num_counts=NUM_COUNTS,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass at each stage
    batch_size = 4
    seq_len = 20
    actions = torch.randint(0, action_vocab, (batch_size, seq_len))

    for stage in [1, 2, 3]:
        model.set_stage(stage)
        output = model(actions)
        print(f"\nStage {stage}:")
        print(f"  Active heads: {model.get_active_heads()}")
        print(f"  action_logits: {output['action_logits'].shape}")
        if 'modifier_logits' in output:
            print(f"  modifier_logits: {output['modifier_logits'].shape}")
        if 'count_logits' in output:
            print(f"  count_logits: {output['count_logits'].shape}")
        print(f"  v_seen: {output['v_seen'].shape}")
