"""
Weaver (Generator) network for GPN-1.

The Weaver generates images from latent codes and produces v_pred (costly signal)
that estimates the value the Witness will perceive.

Per contracts/models.md: WeaverInterface with forward(z, labels) -> (images, v_pred).

Exports:
    - Weaver: Generator with value prediction
    - create_weaver: Factory function
"""

from typing import Optional

import torch
import torch.nn as nn


class Weaver(nn.Module):
    """
    Generator with costly signaling (v_pred).

    The Weaver takes latent noise and class labels, generating images along with
    a value prediction (v_pred) that estimates what the Witness will perceive.
    This creates a cooperative signaling dynamic where the Weaver learns to
    predict how its outputs will be received.

    Implements WeaverInterface per contracts/models.md:
    - forward(z: Tensor[B, latent_dim], labels: Tensor[B]) -> (images: Tensor[B,1,28,28], v_pred: Tensor[B, v_dim])

    Attributes:
        latent_dim: Dimension of input noise vector
        num_classes: Number of output classes (10 for MNIST)
        v_pred_dim: Dimension of value prediction output
    """

    def __init__(
        self,
        latent_dim: int = 64,
        num_classes: int = 10,
        image_channels: int = 1,
        image_size: int = 28,
        hidden_dims: Optional[list[int]] = None,
        use_batch_norm: bool = True,
        v_pred_hidden: int = 128,
        v_pred_dim: int = 16,
    ) -> None:
        """
        Initialize Weaver network.

        Args:
            latent_dim: Dimension of latent noise vector
            num_classes: Number of classes for conditioning
            image_channels: Output image channels (1 for MNIST)
            image_size: Output image size (28 for MNIST)
            hidden_dims: Generator hidden dimensions
            use_batch_norm: Whether to use batch normalization
            v_pred_hidden: Hidden dimension for value prediction head
            v_pred_dim: Output dimension for value prediction
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 512]

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_channels = image_channels
        self.image_size = image_size
        self.v_pred_dim = v_pred_dim

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        # Calculate initial feature map size
        # We'll use a 7x7 initial feature map for 28x28 output (7 * 2^2 = 28)
        self.init_size = image_size // 4  # 7 for 28x28
        init_channels = hidden_dims[-1]

        # Project latent + label to initial feature map
        self.fc = nn.Linear(latent_dim * 2, init_channels * self.init_size * self.init_size)

        # Build generator blocks (upsampling)
        # Need 2 upsampling blocks: 7 -> 14 -> 28
        blocks: list[nn.Module] = []
        in_channels = init_channels

        # decoder_dims determines number of upsample blocks
        decoder_dims = [256, 128]  # Two blocks for 2x upsampling each

        for out_channels in decoder_dims:
            blocks.append(self._make_block(in_channels, out_channels, use_batch_norm))
            in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)

        # Final output layer
        self.output = nn.Sequential(
            nn.Conv2d(in_channels, image_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Value prediction head (v_pred)
        # Takes features from the generator's intermediate representation
        self.v_pred_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, v_pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(v_pred_hidden, v_pred_dim),
        )

        # Store intermediate features for v_pred
        self._features: Optional[torch.Tensor] = None

    def _make_block(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool,
    ) -> nn.Module:
        """Create a generator block with upsampling."""
        layers: list[nn.Module] = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate images and value predictions.

        Args:
            z: Latent noise vectors [B, latent_dim]
            labels: Class labels [B]

        Returns:
            Tuple of:
                - images: Generated images [B, 1, 28, 28]
                - v_pred: Value predictions [B, v_pred_dim]
        """
        batch_size = z.size(0)

        # Embed labels and concatenate with noise
        label_emb = self.label_embedding(labels)  # [B, latent_dim]
        x = torch.cat([z, label_emb], dim=1)  # [B, latent_dim * 2]

        # Project to initial feature map
        x = self.fc(x)
        x = x.view(batch_size, -1, self.init_size, self.init_size)

        # Generate through blocks
        x = self.blocks(x)

        # Store features for v_pred before final output
        self._features = x

        # Generate image
        images = self.output(x)

        # Compute value prediction from features
        v_pred = self.v_pred_head(self._features)

        return images, v_pred

    def generate(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate images only (without v_pred).

        Args:
            z: Latent noise vectors [B, latent_dim]
            labels: Class labels [B]

        Returns:
            Generated images [B, 1, 28, 28]
        """
        images, _ = self.forward(z, labels)
        return images


def create_weaver(
    latent_dim: int = 64,
    num_classes: int = 10,
    hidden_dims: Optional[list[int]] = None,
    v_pred_dim: int = 16,
    device: Optional[torch.device] = None,
) -> Weaver:
    """
    Factory function to create a Weaver.

    Args:
        latent_dim: Dimension of latent noise vector
        num_classes: Number of classes
        hidden_dims: Generator hidden dimensions
        v_pred_dim: Value prediction output dimension
        device: Device to place model on

    Returns:
        Initialized Weaver

    Example:
        >>> weaver = create_weaver(latent_dim=64, v_pred_dim=16)
        >>> z = torch.randn(32, 64)
        >>> labels = torch.randint(0, 10, (32,))
        >>> images, v_pred = weaver(z, labels)
    """
    weaver = Weaver(
        latent_dim=latent_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        v_pred_dim=v_pred_dim,
    )

    if device is not None:
        weaver = weaver.to(device)

    return weaver
