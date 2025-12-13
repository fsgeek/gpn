"""
Transformer-based Weaver for SCAN-lite sequence generation.

Encoder-decoder architecture that takes command tokens and generates action sequences.
Includes v_pred head for pedagogical training (predicting Witness perception).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerWeaver(nn.Module):
    """
    Transformer encoder-decoder for sequence generation.

    Takes command tokens, generates action sequence tokens.
    Outputs v_pred for pedagogical training.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 64,
        v_pred_dim: int = 16,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # v_pred head (for pedagogical training)
        self.v_pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, v_pred_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def _create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Create padding mask (True where padded)."""
        return seq == self.pad_idx

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode source sequence.

        Args:
            src: (batch, src_len) source token indices
            src_mask: optional padding mask

        Returns:
            (batch, src_len, d_model) encoder output
        """
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)

        src_key_padding_mask = self._create_padding_mask(src) if src_mask is None else src_mask
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode target sequence.

        Args:
            tgt: (batch, tgt_len) target token indices
            memory: (batch, src_len, d_model) encoder output
            tgt_mask: causal mask
            memory_key_padding_mask: source padding mask

        Returns:
            (batch, tgt_len, d_model) decoder output
        """
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)

        tgt_len = tgt.size(1)
        causal_mask = self._generate_square_subsequent_mask(tgt_len, tgt.device)
        tgt_key_padding_mask = self._create_padding_mask(tgt)

        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return output

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            src: (batch, src_len) command tokens
            tgt: (batch, tgt_len) action tokens (teacher forcing)

        Returns:
            logits: (batch, tgt_len, vocab_size) output logits
            v_pred: (batch, v_pred_dim) predicted value
        """
        src_padding_mask = self._create_padding_mask(src)
        memory = self.encode(src, src_padding_mask)
        decoder_output = self.decode(tgt, memory, memory_key_padding_mask=src_padding_mask)

        # Output logits
        logits = self.output_proj(decoder_output)

        # v_pred from pooled encoder output (mean over non-padded positions)
        mask = ~src_padding_mask  # True where NOT padded
        mask = mask.unsqueeze(-1).float()  # (batch, src_len, 1)
        pooled = (memory * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (batch, d_model)
        v_pred = self.v_pred_head(pooled)

        return logits, v_pred

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 32,
        sos_idx: int = 1,
        eos_idx: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive generation.

        Args:
            src: (batch, src_len) command tokens
            max_len: maximum output length
            sos_idx: start of sequence token index
            eos_idx: end of sequence token index

        Returns:
            output: (batch, gen_len) generated token indices
            v_pred: (batch, v_pred_dim) predicted value
        """
        batch_size = src.size(0)
        device = src.device

        src_padding_mask = self._create_padding_mask(src)
        memory = self.encode(src, src_padding_mask)

        # Initialize with SOS
        output = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            decoder_output = self.decode(output, memory, memory_key_padding_mask=src_padding_mask)
            logits = self.output_proj(decoder_output[:, -1, :])  # Last position
            next_token = logits.argmax(dim=-1, keepdim=True)  # (batch, 1)

            output = torch.cat([output, next_token], dim=1)

            # Check for EOS
            finished = finished | (next_token.squeeze(-1) == eos_idx)
            if finished.all():
                break

        # Compute v_pred
        mask = ~src_padding_mask
        mask = mask.unsqueeze(-1).float()
        pooled = (memory * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        v_pred = self.v_pred_head(pooled)

        return output, v_pred


def create_transformer_weaver(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 128,
    n_heads: int = 4,
    n_encoder_layers: int = 3,
    n_decoder_layers: int = 3,
    v_pred_dim: int = 16,
    device: str = 'cpu',
) -> TransformerWeaver:
    """Factory function for TransformerWeaver."""
    model = TransformerWeaver(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        v_pred_dim=v_pred_dim,
    )
    return model.to(device)


if __name__ == '__main__':
    # Test the model
    from src.data.scan_lite import get_command_vocab_size, get_action_vocab_size

    print("TransformerWeaver Test")
    print("=" * 50)

    src_vocab = get_command_vocab_size()
    tgt_vocab = get_action_vocab_size()

    model = create_transformer_weaver(src_vocab, tgt_vocab)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    src_len = 5
    tgt_len = 10

    src = torch.randint(0, src_vocab, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))

    logits, v_pred = model(src, tgt)
    print(f"Logits shape: {logits.shape}")  # Should be (4, 10, tgt_vocab)
    print(f"v_pred shape: {v_pred.shape}")  # Should be (4, 16)

    # Test generation
    output, v_pred_gen = model.generate(src, max_len=15)
    print(f"Generated shape: {output.shape}")
    print(f"v_pred (gen) shape: {v_pred_gen.shape}")
