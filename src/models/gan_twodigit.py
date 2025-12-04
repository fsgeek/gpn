"""
Two-digit GAN models for GPN-2 baseline comparison.

Tests whether from-scratch adversarial training can learn
compositional 2-digit generation without curriculum.
"""

from typing import Optional
import torch
import torch.nn as nn


class TwoDigitGenerator(nn.Module):
    """
    GAN generator for 2-digit MNIST (28x56 images).

    Direct generation without curriculum - the from-scratch ablation baseline.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 100,  # 0-99
        image_channels: int = 1,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initialize generator.

        Args:
            latent_dim: Dimension of latent noise
            num_classes: Number of classes (100 for 0-99)
            image_channels: Output channels
            hidden_dim: Base hidden dimension
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        # Initial projection to 7x14 (will upsample to 28x56)
        self.init_h = 7
        self.init_w = 14
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim * self.init_h * self.init_w),
            nn.BatchNorm1d(hidden_dim * self.init_h * self.init_w),
            nn.ReLU(inplace=True),
        )

        # Upsampling: 7x14 -> 14x28 -> 28x56
        self.conv_blocks = nn.Sequential(
            # 7x14 -> 14x28
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            # 14x28 -> 28x56
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            # Final
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
        Generate 2-digit images.

        Args:
            z: Latent noise [B, latent_dim]
            labels: Class labels [B] in range 0-99

        Returns:
            Generated images [B, 1, 28, 56]
        """
        batch_size = z.size(0)

        # Embed and concatenate
        label_emb = self.label_embedding(labels)
        x = torch.cat([z, label_emb], dim=1)

        # Project
        x = self.fc(x)
        x = x.view(batch_size, self._hidden_dim, self.init_h, self.init_w)

        # Generate
        return self.conv_blocks(x)


class TwoDigitDiscriminator(nn.Module):
    """
    GAN discriminator for 2-digit MNIST (28x56 images).

    Classifies images as real or fake, conditioned on 2-digit labels.
    """

    def __init__(
        self,
        num_classes: int = 100,
        image_channels: int = 1,
        hidden_dim: int = 64,
    ) -> None:
        """
        Initialize discriminator.

        Args:
            num_classes: Number of classes (100)
            image_channels: Input channels
            hidden_dim: Base hidden dimension
        """
        super().__init__()

        self.num_classes = num_classes

        # Label embedding (project to image size)
        self.label_embedding = nn.Embedding(num_classes, 28 * 56)

        # Discriminator blocks (28x56 -> 14x28 -> 7x14 -> 3x7)
        self.conv_blocks = nn.Sequential(
            # 28x56 -> 14x28
            nn.Conv2d(image_channels + 1, hidden_dim, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # 14x28 -> 7x14
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # 7x14 -> 3x7
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        # Output layer
        # After 3 stride-2 convs: 28->14->7->4, 56->28->14->7
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 4 * 4 * 7, 1),
        )

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discriminate images.

        Args:
            images: Input images [B, 1, 28, 56]
            labels: Class labels [B] in range 0-99

        Returns:
            Discrimination scores [B, 1]
        """
        batch_size = images.size(0)

        # Embed labels as image channel
        label_emb = self.label_embedding(labels)
        label_emb = label_emb.view(batch_size, 1, 28, 56)

        # Concatenate
        x = torch.cat([images, label_emb], dim=1)

        # Discriminate
        x = self.conv_blocks(x)
        return self.fc(x)


def create_twodigit_gan(
    latent_dim: int = 128,
    device: Optional[torch.device] = None,
) -> tuple[TwoDigitGenerator, TwoDigitDiscriminator]:
    """
    Create 2-digit GAN models.

    Args:
        latent_dim: Latent dimension
        device: Device to place models

    Returns:
        Tuple of (Generator, Discriminator)
    """
    generator = TwoDigitGenerator(latent_dim=latent_dim)
    discriminator = TwoDigitDiscriminator()

    if device is not None:
        generator = generator.to(device)
        discriminator = discriminator.to(device)

    return generator, discriminator
