"""
Quick Fashion-MNIST pedagogical training for generalization test.

Minimal script to train pedagogical model on Fashion-MNIST using existing architecture.
Tests if topology differences replicate in non-MNIST domain.
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

from src.models.weaver import Weaver
from src.models.witness import Witness
from src.models.judge import Judge

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


def train_judge(device: torch.device, epochs: int = 5) -> Judge:
    """Train frozen Judge classifier on Fashion-MNIST."""
    logger.info("Training Judge classifier on Fashion-MNIST...")

    judge = Judge()
    judge.to(device)

    train_loader = get_fashion_mnist_loader(batch_size=128, train=True)
    test_loader = get_fashion_mnist_loader(batch_size=128, train=False)

    optimizer = torch.optim.Adam(judge.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        judge.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = judge(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate
        judge.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = judge(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        logger.info(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy:.2f}%")

    return judge


def train_pedagogical(
    steps: int = 10000,
    batch_size: int = 64,
    latent_dim: int = 64,
    device: torch.device = torch.device('cpu'),
    checkpoint_path: Path = Path('checkpoints/fashion_mnist_pedagogical.pt'),
):
    """
    Train pedagogical model on Fashion-MNIST.

    Simplified three-phase training focused on getting a working model quickly.
    """
    logger.info("Starting Fashion-MNIST pedagogical training")
    logger.info(f"Total steps: {steps}, Device: {device}")

    # Data
    train_loader = get_fashion_mnist_loader(batch_size=batch_size, train=True)

    # Train Judge first (frozen oracle)
    judge = train_judge(device, epochs=5)
    judge.eval()
    for param in judge.parameters():
        param.requires_grad = False

    # Models
    weaver = Weaver(latent_dim=latent_dim, num_classes=10)
    witness = Witness(num_classes=10)

    weaver.to(device)
    witness.to(device)

    # Optimizers
    weaver_optimizer = torch.optim.Adam(weaver.parameters(), lr=0.0002, betas=(0.5, 0.999))
    witness_optimizer = torch.optim.Adam(witness.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    weaver.train()
    witness.train()

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

        # Generate fake images
        z = torch.randn(batch_size_actual, latent_dim, device=device)
        fake_labels = torch.randint(0, 10, (batch_size_actual,), device=device)

        fake_images, v_pred = weaver(z, fake_labels)

        # Get Witness perception (returns class_logits, v_seen)
        witness_class_logits, v_seen = witness(fake_images)

        # Get Judge grounding
        with torch.no_grad():
            judge_logits = judge(fake_images)

        # Losses
        # 1. Alignment: Weaver predicts what Witness sees
        alignment_loss = nn.functional.mse_loss(v_pred, v_seen.detach())

        # 2. Grounding: Match Judge's perception (phase-dependent)
        if step < steps // 3:
            # Phase 1: Heavy grounding
            grounding_weight = 1.0
        elif step < 2 * steps // 3:
            # Phase 2: Balanced
            grounding_weight = 0.5
        else:
            # Phase 3: Minimal grounding
            grounding_weight = 0.1

        judge_target = judge_logits.argmax(dim=1)
        grounding_loss = nn.functional.cross_entropy(witness_class_logits, judge_target)

        # Update Weaver
        weaver_loss = alignment_loss + grounding_weight * grounding_loss.detach()
        weaver_optimizer.zero_grad()
        weaver_loss.backward()
        weaver_optimizer.step()

        # Update Witness (separate forward passes to avoid graph issues)
        # 1. Grounding on fake images
        witness_class_logits_2, _ = witness(fake_images.detach())
        grounding_loss_2 = nn.functional.cross_entropy(witness_class_logits_2, judge_target.detach())

        # 2. Quality on real images
        witness_real_class_logits, _ = witness(real_images)
        witness_quality_loss = nn.functional.cross_entropy(witness_real_class_logits, real_labels)

        witness_loss = grounding_loss_2 + witness_quality_loss
        witness_optimizer.zero_grad()
        witness_loss.backward()
        witness_optimizer.step()

        step += 1

        if step % 500 == 0:
            logger.info(
                f"Step {step}/{steps} | "
                f"Weaver: {weaver_loss.item():.4f} | "
                f"Witness: {witness_loss.item():.4f} | "
                f"Alignment: {alignment_loss.item():.4f}"
            )

    # Save checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'models': {
            'weaver': weaver.state_dict(),
            'witness': witness.state_dict(),
            'judge': judge.state_dict(),
        },
        'step': step,
    }, checkpoint_path)

    logger.info(f"Saved checkpoint to {checkpoint_path}")

    return weaver, witness, judge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/fashion_mnist_pedagogical.pt'))
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])

    args = parser.parse_args()

    device = torch.device(args.device)

    train_pedagogical(
        steps=args.steps,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        device=device,
        checkpoint_path=args.checkpoint,
    )


if __name__ == '__main__':
    main()
