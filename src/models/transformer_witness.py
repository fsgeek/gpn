"""
Transformer-based Witness for SCAN-lite sequence evaluation.

Encoder-only architecture that takes action sequences and predicts:
1. The original command (reverse mapping)
2. v_seen (perceived quality/understanding)
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class TransformerWitness(nn.Module):
    """
    Transformer encoder for evaluating generated action sequences.

    Takes action tokens, outputs command prediction and v_seen.
    """

    def __init__(
        self,
        action_vocab_size: int,
        command_vocab_size: int,
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

        # Embedding
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

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Command prediction head - predicts unique command ID (0-63)
        # This captures the FULL compositional structure, not just action type
        self.num_commands = command_vocab_size  # Will be 64 for full command IDs
        self.command_head = nn.Linear(d_model, command_vocab_size)

        # v_seen head
        self.v_seen_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, v_seen_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        return seq == self.pad_idx

    def forward(
        self,
        action_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            action_seq: (batch, seq_len) action token indices

        Returns:
            command_logits: (batch, command_vocab_size) predicted command
            v_seen: (batch, v_seen_dim) perceived quality
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

        # Output heads
        command_logits = self.command_head(pooled)
        v_seen = self.v_seen_head(pooled)

        return command_logits, v_seen


def create_transformer_witness(
    action_vocab_size: int,
    command_vocab_size: int,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    v_seen_dim: int = 16,
    device: str = 'cpu',
) -> TransformerWitness:
    """Factory function for TransformerWitness."""
    model = TransformerWitness(
        action_vocab_size=action_vocab_size,
        command_vocab_size=command_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        v_seen_dim=v_seen_dim,
    )
    return model.to(device)


if __name__ == '__main__':
    from src.data.scan_lite import get_command_vocab_size, get_action_vocab_size

    print("TransformerWitness Test")
    print("=" * 50)

    action_vocab = get_action_vocab_size()
    command_vocab = get_command_vocab_size()

    model = create_transformer_witness(action_vocab, command_vocab)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    seq_len = 20

    actions = torch.randint(0, action_vocab, (batch_size, seq_len))

    command_logits, v_seen = model(actions)
    print(f"Command logits shape: {command_logits.shape}")  # Should be (4, command_vocab)
    print(f"v_seen shape: {v_seen.shape}")  # Should be (4, 16)
