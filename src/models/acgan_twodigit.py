"""
AC-GAN (Auxiliary Classifier GAN) for 2-digit MNIST.

Tests hybrid adversarial + pedagogical approach:
- Adversarial objective: real vs fake
- Pedagogical objective: class prediction

This combines GAN's bootstrapping ability with pedagogical class guidance.
"""

from typing import Optional
import torch
import torch.nn as nn


class ACGANGenerator(nn.Module):
    """
    AC-GAN generator for 2-digit MNIST.

    Same as standard GAN generator - the difference is in training,
    not architecture.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 100,
        image_channels: int = 1,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        # Initial projection to 7x14
        self.init_h = 7
        self.init_w = 14
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim * self.init_h * self.init_w),
            nn.BatchNorm1d(hidden_dim * self.init_h * self.init_w),
            nn.ReLU(inplace=True),
        )

        # Upsampling: 7x14 -> 14x28 -> 28x56
        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, image_channels, 3, padding=1),
            nn.Tanh(),
        )

        self._hidden_dim = hidden_dim

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = z.size(0)

        label_emb = self.label_embedding(labels)
        x = torch.cat([z, label_emb], dim=1)

        x = self.fc(x)
        x = x.view(batch_size, self._hidden_dim, self.init_h, self.init_w)

        return self.conv_blocks(x)


class ACGANDiscriminator(nn.Module):
    """
    AC-GAN discriminator for 2-digit MNIST.

    Outputs TWO things:
    1. Real/fake discrimination (adversarial)
    2. Class prediction (pedagogical)

    This is the key: forcing the discriminator to be class-aware
    makes the generator produce class-discriminative features.
    """

    def __init__(
        self,
        num_classes: int = 100,
        image_channels: int = 1,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Convolutional feature extraction
        self.conv_blocks = nn.Sequential(
            # 28x56 -> 14x28
            nn.Conv2d(image_channels, hidden_dim, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # 14x28 -> 7x14
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # 7x14 -> 4x7
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        # Shared features
        self.flatten = nn.Flatten()

        # Two heads
        self.adversarial_head = nn.Linear(hidden_dim * 4 * 4 * 7, 1)  # Real/fake
        self.class_head = nn.Linear(hidden_dim * 4 * 4 * 7, num_classes)  # Class prediction

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Input images [B, 1, 28, 56]

        Returns:
            Tuple of (adversarial_logits [B, 1], class_logits [B, num_classes])
        """
        features = self.conv_blocks(images)
        features = self.flatten(features)

        adversarial = self.adversarial_head(features)
        classes = self.class_head(features)

        return adversarial, classes


def create_acgan_twodigit(
    latent_dim: int = 128,
    device: Optional[torch.device] = None,
) -> tuple[ACGANGenerator, ACGANDiscriminator]:
    """
    Create AC-GAN models for 2-digit MNIST.

    Args:
        latent_dim: Latent dimension
        device: Device to place models

    Returns:
        Tuple of (Generator, Discriminator)
    """
    generator = ACGANGenerator(latent_dim=latent_dim)
    discriminator = ACGANDiscriminator()

    if device is not None:
        generator = generator.to(device)
        discriminator = discriminator.to(device)

    return generator, discriminator
