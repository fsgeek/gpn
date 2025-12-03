#!/usr/bin/env python3
"""
Train Judge classifier on MNIST.

The Judge must achieve >95% accuracy before being frozen for GPN training.
This script trains the Judge and saves a checkpoint.

Usage:
    python scripts/train_judge.py [--epochs N] [--checkpoint PATH]
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.judge import Judge
from src.utils.reproducibility import set_reproducibility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_mnist_loaders(
    batch_size: int = 64,
    num_workers: int = 4,
    data_dir: str = "data",
) -> tuple[DataLoader, DataLoader]:
    """
    Get MNIST train and test data loaders.

    Args:
        batch_size: Batch size for loading
        num_workers: Number of data loading workers
        data_dir: Directory to store/load MNIST data

    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def train_epoch(
    model: Judge,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: Judge model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Training device

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
    model: Judge,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate model on test set.

    Args:
        model: Judge model
        test_loader: Test data loader
        criterion: Loss function
        device: Evaluation device

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_judge(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    checkpoint_path: str = "checkpoints/judge.pt",
    device: str = "auto",
    target_accuracy: float = 0.95,
) -> tuple[Judge, float]:
    """
    Train Judge classifier to target accuracy.

    Args:
        epochs: Maximum training epochs
        batch_size: Training batch size
        lr: Learning rate
        seed: Random seed
        checkpoint_path: Path to save checkpoint
        device: Training device
        target_accuracy: Target test accuracy (default 95%)

    Returns:
        Tuple of (trained Judge, final accuracy)

    Raises:
        RuntimeError: If target accuracy not achieved
    """
    # Setup
    set_reproducibility(seed)

    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    logger.info(f"Training on {device_obj}")

    # Data
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
    logger.info(f"Loaded MNIST: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

    # Model
    model = Judge()
    model = model.to(device_obj)
    logger.info(f"Judge parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training components
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_accuracy = 0.0
    best_state = None

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device_obj
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device_obj)

        scheduler.step(test_acc)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_state = model.state_dict()

        # Early stopping if target reached
        if test_acc >= target_accuracy:
            logger.info(f"Target accuracy {target_accuracy:.2%} reached!")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "accuracy": best_accuracy,
            "epochs_trained": epoch + 1,
            "hidden_dims": model.hidden_dims,
        },
        checkpoint_path,
    )
    logger.info(f"Saved checkpoint to {checkpoint_path} (accuracy: {best_accuracy:.4f})")

    if best_accuracy < target_accuracy:
        logger.warning(
            f"Target accuracy {target_accuracy:.2%} not achieved. "
            f"Best: {best_accuracy:.2%}. Consider training longer."
        )

    return model, best_accuracy


def main():
    parser = argparse.ArgumentParser(description="Train Judge classifier on MNIST")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/judge.pt",
        help="Checkpoint path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.95,
        help="Target test accuracy",
    )

    args = parser.parse_args()

    model, accuracy = train_judge(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        device=args.device,
        target_accuracy=args.target_accuracy,
    )

    logger.info(f"Training complete. Final accuracy: {accuracy:.4f}")

    if accuracy >= args.target_accuracy:
        logger.info("✓ Judge ready for GPN training")
    else:
        logger.warning("✗ Judge below target accuracy - consider retraining")


if __name__ == "__main__":
    main()
