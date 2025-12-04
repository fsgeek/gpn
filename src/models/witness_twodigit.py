"""
Two-Digit Witness: Classifier for 2-digit MNIST with value estimation.

Evaluates whether generated 2-digit images are recognizable and produces
v_seen for alignment training.

Two modes:
1. full_number: Single 100-class classification
2. per_position: Two 10-class heads for tens/ones digits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TwoDigitWitness(nn.Module):
    """
    Classifier with value estimation for 2-digit MNIST images (28x56).

    Mirrors TwoDigitJudge architecture but adds v_seen output.
    """

    def __init__(
        self,
        hidden_dims: list[int] = [64, 128, 256],
        v_seen_dim: int = 16,
        v_seen_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.v_seen_dim = v_seen_dim

        # Convolutional layers
        # Input: [B, 1, 28, 56]
        self.conv1 = nn.Conv2d(1, hidden_dims[0], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dims[0])
        # After pool: [B, 64, 14, 28]

        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dims[1])
        # After pool: [B, 128, 7, 14]

        self.conv3 = nn.Conv2d(hidden_dims[1], hidden_dims[2], 3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dims[2])
        # After pool: [B, 256, 3, 7]

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        # Calculate flattened size: 256 * 3 * 7 = 5376
        flat_size = hidden_dims[2] * 3 * 7

        # Shared feature layer
        self.fc_shared = nn.Linear(flat_size, 256)

        # Full number head (100 classes)
        self.fc_full = nn.Linear(256, 100)

        # Per-position heads (10 classes each)
        self.fc_tens = nn.Linear(256, 10)
        self.fc_ones = nn.Linear(256, 10)

        # Value estimation head (v_seen)
        self.v_seen_head = nn.Sequential(
            nn.Linear(256, v_seen_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(v_seen_hidden, v_seen_dim),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared features from input."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B, 64, 14, 28]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 128, 7, 14]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [B, 256, 3, 7]
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc_shared(x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "full",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images [B, 1, 28, 56]
            mode: "full" for 100-class output, "per_position" for two 10-class outputs

        Returns:
            If mode="full": (logits [B, 100], v_seen [B, v_seen_dim])
            If mode="per_position": ((tens [B,10], ones [B,10]), v_seen [B, v_seen_dim])
        """
        features = self.extract_features(x)
        v_seen = self.v_seen_head(features)

        if mode == "full":
            logits = self.fc_full(features)
            return logits, v_seen
        elif mode == "per_position":
            tens_logits = self.fc_tens(features)
            ones_logits = self.fc_ones(features)
            return (tens_logits, ones_logits), v_seen
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Classify without v_seen."""
        logits, _ = self.forward(x, mode="full")
        return logits

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


def create_twodigit_witness(
    v_seen_dim: int = 16,
    dropout: float = 0.3,
    device: Optional[torch.device] = None,
) -> TwoDigitWitness:
    """
    Create a TwoDigitWitness.

    Args:
        v_seen_dim: Value estimation output dimension
        dropout: Dropout rate
        device: Device to place model on

    Returns:
        Initialized TwoDigitWitness
    """
    witness = TwoDigitWitness(v_seen_dim=v_seen_dim, dropout=dropout)

    if device is not None:
        witness = witness.to(device)

    return witness


if __name__ == "__main__":
    # Quick test
    print("Testing TwoDigitWitness...")

    device = torch.device("cpu")
    witness = create_twodigit_witness(device=device)

    # Test input
    images = torch.randn(4, 1, 28, 56)

    # Full mode
    logits, v_seen = witness(images, mode="full")
    print(f"Full mode - logits: {logits.shape}, v_seen: {v_seen.shape}")

    # Per-position mode
    (tens, ones), v_seen = witness(images, mode="per_position")
    print(f"Per-position mode - tens: {tens.shape}, ones: {ones.shape}, v_seen: {v_seen.shape}")

    print("TwoDigitWitness test passed!")
