"""
AC-GAN (Auxiliary Classifier GAN) for single-digit MNIST.

Tests hybrid adversarial + pedagogical approach on single digits:
- Adversarial objective: real vs fake
- Pedagogical objective: class prediction

This is the curriculum foundation for AC-GAN 2-digit composition.
"""

from typing import Optional
import torch
import torch.nn as nn


class ACGANGenerator(nn.Module):
    """
    AC-GAN generator for single-digit MNIST.

    Same as standard GAN generator - the difference is in training,
    not architecture.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 10,
        image_channels: int = 1,
        image_size: int = 28,
        hidden_dim: int = 256,
    ) -> None:
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

        # Upsampling blocks: 7x7 -> 14x14 -> 28x28
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
        x = x.view(batch_size, self._hidden_dim, self.init_size, self.init_size)

        return self.conv_blocks(x)


class ACGANDiscriminator(nn.Module):
    """
    AC-GAN discriminator for single-digit MNIST.

    Outputs TWO things:
    1. Real/fake discrimination (adversarial)
    2. Class prediction (pedagogical)

    This is the key: forcing the discriminator to be class-aware
    makes the generator produce class-discriminative features.
    """

    def __init__(
        self,
        num_classes: int = 10,
        image_channels: int = 1,
        image_size: int = 28,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Convolutional feature extraction: 28x28 -> 14x14 -> 7x7 -> 4x4
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        # Shared features
        self.flatten = nn.Flatten()

        # Two heads
        self.adversarial_head = nn.Linear(hidden_dim * 4 * 4 * 4, 1)  # Real/fake
        self.class_head = nn.Linear(hidden_dim * 4 * 4 * 4, num_classes)  # Class prediction

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Input images [B, 1, 28, 28]

        Returns:
            Tuple of (adversarial_logits [B, 1], class_logits [B, num_classes])
        """
        features = self.conv_blocks(images)
        features = self.flatten(features)

        adversarial = self.adversarial_head(features)
        classes = self.class_head(features)

        return adversarial, classes


def create_acgan(
    latent_dim: int = 128,
    device: Optional[torch.device] = None,
) -> tuple[ACGANGenerator, ACGANDiscriminator]:
    """
    Create AC-GAN models for single-digit MNIST.

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
