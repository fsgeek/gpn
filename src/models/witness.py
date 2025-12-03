"""
Witness (Classifier) network for GPN-1.

The Witness classifies images and produces v_seen (perceived value) that
the Weaver tries to predict. This creates a feedback loop for cooperative
learning.

Per contracts/models.md: WitnessInterface with forward(images) -> (logits, v_seen).

Exports:
    - Witness: Classifier with value estimation
    - create_witness: Factory function
"""

from typing import Optional

import torch
import torch.nn as nn


class Witness(nn.Module):
    """
    Classifier with value estimation (v_seen).

    The Witness takes images and produces classification logits along with
    v_seen (the value it "sees" in the image). The Weaver's v_pred is trained
    to match v_seen, creating an alignment signal.

    Implements WitnessInterface per contracts/models.md:
    - forward(images: Tensor[B,1,28,28]) -> (logits: Tensor[B,10], v_seen: Tensor[B, v_dim])

    Attributes:
        num_classes: Number of output classes (10 for MNIST)
        v_seen_dim: Dimension of value estimation output
    """

    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 28,
        num_classes: int = 10,
        hidden_dims: Optional[list[int]] = None,
        use_batch_norm: bool = True,
        v_seen_hidden: int = 128,
        v_seen_dim: int = 16,
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize Witness network.

        Args:
            image_channels: Input image channels (1 for MNIST)
            image_size: Input image size (28 for MNIST)
            num_classes: Number of output classes
            hidden_dims: Encoder hidden dimensions
            use_batch_norm: Whether to use batch normalization
            v_seen_hidden: Hidden dimension for value estimation head
            v_seen_dim: Output dimension for value estimation
            dropout: Dropout rate
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 256]  # Three blocks: 28->14->7->3 (rounded)

        self.image_channels = image_channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.v_seen_dim = v_seen_dim

        # Build encoder blocks
        blocks: list[nn.Module] = []
        in_channels = image_channels

        for out_channels in hidden_dims:
            blocks.append(
                self._make_block(in_channels, out_channels, use_batch_norm, dropout)
            )
            in_channels = out_channels

        self.encoder = nn.Sequential(*blocks)

        # Calculate feature size after encoding
        # Each block reduces spatial dimensions by 2 (stride=2)
        # 28 -> 14 -> 7 -> 3 (for 3 blocks)
        num_downsamples = len(hidden_dims)
        final_size = image_size
        for _ in range(num_downsamples):
            final_size = (final_size + 1) // 2  # ceil division for stride 2
        if final_size < 1:
            final_size = 1
        feature_dim = hidden_dims[-1] * final_size * final_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dims[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes),
        )

        # Value estimation head (v_seen)
        self.v_seen_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, v_seen_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(v_seen_hidden, v_seen_dim),
        )

    def _make_block(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool,
        dropout: float,
    ) -> nn.Module:
        """Create an encoder block with downsampling."""
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.extend([
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        ])

        return nn.Sequential(*layers)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Classify images and estimate value.

        Args:
            images: Input images [B, 1, 28, 28]

        Returns:
            Tuple of:
                - logits: Classification logits [B, 10]
                - v_seen: Value estimation [B, v_seen_dim]
        """
        # Encode images
        features = self.encoder(images)

        # Classify
        logits = self.classifier(features)

        # Estimate value
        v_seen = self.v_seen_head(features)

        return logits, v_seen

    def classify(self, images: torch.Tensor) -> torch.Tensor:
        """
        Classify images only (without v_seen).

        Args:
            images: Input images [B, 1, 28, 28]

        Returns:
            Classification logits [B, 10]
        """
        logits, _ = self.forward(images)
        return logits

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions.

        Args:
            images: Input images [B, 1, 28, 28]

        Returns:
            Predicted class indices [B]
        """
        logits = self.classify(images)
        return logits.argmax(dim=1)

    def accuracy(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Compute classification accuracy.

        Args:
            images: Input images [B, 1, 28, 28]
            labels: Ground truth labels [B]

        Returns:
            Accuracy as float (0.0 to 1.0)
        """
        with torch.no_grad():
            predictions = self.predict(images)
            correct = (predictions == labels).float().mean()
            return correct.item()


def create_witness(
    num_classes: int = 10,
    hidden_dims: Optional[list[int]] = None,
    v_seen_dim: int = 16,
    dropout: float = 0.3,
    device: Optional[torch.device] = None,
) -> Witness:
    """
    Factory function to create a Witness.

    Args:
        num_classes: Number of output classes
        hidden_dims: Encoder hidden dimensions
        v_seen_dim: Value estimation output dimension
        dropout: Dropout rate
        device: Device to place model on

    Returns:
        Initialized Witness

    Example:
        >>> witness = create_witness(num_classes=10, v_seen_dim=16)
        >>> images = torch.randn(32, 1, 28, 28)
        >>> logits, v_seen = witness(images)
    """
    witness = Witness(
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        v_seen_dim=v_seen_dim,
        dropout=dropout,
    )

    if device is not None:
        witness = witness.to(device)

    return witness
