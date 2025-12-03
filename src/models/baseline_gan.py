"""
Baseline GAN models for comparison with GPN.

Implements standard adversarial GAN for convergence speed comparison.

Exports:
    - Generator: Standard conditional GAN generator
    - Discriminator: Standard conditional GAN discriminator
    - create_baseline_gan: Factory function
"""

from typing import Optional

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Standard conditional GAN generator for MNIST.

    Generates images from latent noise and class labels.
    Uses transposed convolutions for upsampling.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        num_classes: int = 10,
        image_channels: int = 1,
        image_size: int = 28,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initialize Generator.

        Args:
            latent_dim: Dimension of latent noise vector
            num_classes: Number of classes for conditioning
            image_channels: Output image channels
            image_size: Output image size
            hidden_dim: Base hidden dimension
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_size = image_size

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        # Initial projection
        self.init_size = image_size // 4  # 7 for 28x28
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim * self.init_size * self.init_size),
            nn.BatchNorm1d(hidden_dim * self.init_size * self.init_size),
            nn.ReLU(inplace=True),
        )

        # Upsampling blocks
        self.conv_blocks = nn.Sequential(
            # 7x7 -> 14x14
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            # 14x14 -> 28x28
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            # Final output
            nn.Conv2d(hidden_dim // 4, image_channels, 3, padding=1),
            nn.Tanh(),
        )

        self._hidden_dim = hidden_dim

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate images.

        Args:
            z: Latent noise [B, latent_dim]
            labels: Class labels [B]

        Returns:
            Generated images [B, 1, 28, 28]
        """
        batch_size = z.size(0)

        # Embed labels and concatenate
        label_emb = self.label_embedding(labels)
        x = torch.cat([z, label_emb], dim=1)

        # Project
        x = self.fc(x)
        x = x.view(batch_size, self._hidden_dim, self.init_size, self.init_size)

        # Generate
        return self.conv_blocks(x)


class Discriminator(nn.Module):
    """
    Standard conditional GAN discriminator for MNIST.

    Classifies images as real or fake, conditioned on class labels.
    """

    def __init__(
        self,
        num_classes: int = 10,
        image_channels: int = 1,
        image_size: int = 28,
        hidden_dim: int = 64,
    ) -> None:
        """
        Initialize Discriminator.

        Args:
            num_classes: Number of classes
            image_channels: Input image channels
            image_size: Input image size
            hidden_dim: Base hidden dimension
        """
        super().__init__()

        self.num_classes = num_classes
        self.image_size = image_size

        # Label embedding (projected to image size)
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)

        # Discriminator blocks
        self.conv_blocks = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(image_channels + 1, hidden_dim, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # 14x14 -> 7x7
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # 7x7 -> 3x3
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        # Output layer
        final_size = (image_size + 7) // 8  # After 3 stride-2 convs
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 4 * final_size * final_size, 1),
        )

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discriminate images.

        Args:
            images: Input images [B, 1, 28, 28]
            labels: Class labels [B]

        Returns:
            Discrimination scores [B, 1]
        """
        batch_size = images.size(0)

        # Embed labels as image channel
        label_emb = self.label_embedding(labels)
        label_emb = label_emb.view(batch_size, 1, self.image_size, self.image_size)

        # Concatenate with images
        x = torch.cat([images, label_emb], dim=1)

        # Discriminate
        x = self.conv_blocks(x)
        return self.fc(x)


def create_baseline_gan(
    latent_dim: int = 64,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
) -> tuple[Generator, Discriminator]:
    """
    Create baseline GAN generator and discriminator.

    Args:
        latent_dim: Latent dimension
        num_classes: Number of classes
        device: Device to place models on

    Returns:
        Tuple of (Generator, Discriminator)
    """
    generator = Generator(latent_dim=latent_dim, num_classes=num_classes)
    discriminator = Discriminator(num_classes=num_classes)

    if device is not None:
        generator = generator.to(device)
        discriminator = discriminator.to(device)

    return generator, discriminator
