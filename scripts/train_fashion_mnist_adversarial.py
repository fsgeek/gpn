"""
Quick Fashion-MNIST adversarial training for generalization test.

Standard conditional GAN training on Fashion-MNIST.
Tests if adversarial topology signature replicates.
"""

import argparse
import logging
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baseline_gan import Generator, Discriminator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_fashion_mnist_loader(batch_size: int = 64, train: bool = True) -> DataLoader:
    """Get Fashion-MNIST data loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ])

    # Try with existing data first, fall back to download if needed
    try:
        dataset = datasets.FashionMNIST(
            root="data",
            train=train,
            download=False,
            transform=transform,
        )
    except RuntimeError:
        # If that fails, try downloading
        dataset = datasets.FashionMNIST(
            root="data",
            train=train,
            download=True,
            transform=transform,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


def train_adversarial(
    steps: int = 10000,
    batch_size: int = 64,
    latent_dim: int = 64,
    device: torch.device = torch.device('cpu'),
    checkpoint_path: Path = Path('checkpoints/fashion_mnist_adversarial.pt'),
):
    """
    Train standard conditional GAN on Fashion-MNIST.
    """
    logger.info("Starting Fashion-MNIST adversarial training")
    logger.info(f"Total steps: {steps}, Device: {device}")

    # Data
    train_loader = get_fashion_mnist_loader(batch_size=batch_size, train=True)

    # Models
    generator = Generator(latent_dim=latent_dim, num_classes=10)
    discriminator = Discriminator(num_classes=10)

    generator.to(device)
    discriminator.to(device)

    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    generator.train()
    discriminator.train()

    step = 0
    data_iter = iter(train_loader)

    while step < steps:
        # Get batch
        try:
            real_images, real_labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            real_images, real_labels = next(data_iter)

        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        batch_size_actual = real_images.size(0)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        d_optimizer.zero_grad()

        # Real images
        real_validity = discriminator(real_images, real_labels)
        d_real_loss = criterion(real_validity, torch.ones_like(real_validity))

        # Fake images
        z = torch.randn(batch_size_actual, latent_dim, device=device)
        fake_labels = torch.randint(0, 10, (batch_size_actual,), device=device)
        fake_images = generator(z, fake_labels)

        fake_validity = discriminator(fake_images.detach(), fake_labels)
        d_fake_loss = criterion(fake_validity, torch.zeros_like(fake_validity))

        # Combined D loss
        d_loss = d_real_loss + d_fake_loss

        d_loss.backward()
        d_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------

        g_optimizer.zero_grad()

        # Generate new batch
        z = torch.randn(batch_size_actual, latent_dim, device=device)
        fake_labels = torch.randint(0, 10, (batch_size_actual,), device=device)
        fake_images = generator(z, fake_labels)

        validity = discriminator(fake_images, fake_labels)

        g_total_loss = criterion(validity, torch.ones_like(validity))

        g_total_loss.backward()
        g_optimizer.step()

        step += 1

        if step % 500 == 0:
            logger.info(
                f"Step {step}/{steps} | "
                f"D Loss: {d_loss.item():.4f} | "
                f"G Loss: {g_total_loss.item():.4f}"
            )

    # Save checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'models': {
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
        },
        'step': step,
    }, checkpoint_path)

    logger.info(f"Saved checkpoint to {checkpoint_path}")

    return generator, discriminator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/fashion_mnist_adversarial.pt'))
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])

    args = parser.parse_args()

    device = torch.device(args.device)

    train_adversarial(
        steps=args.steps,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        device=device,
        checkpoint_path=args.checkpoint,
    )


if __name__ == '__main__':
    main()
