"""
Two-Digit Judge: Classifier for 2-digit MNIST numbers.

Evaluates whether generated 2-digit images are recognizable.
Can operate in two modes:
1. Full-number classification (100 classes)
2. Per-position classification (two 10-class heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from pathlib import Path


class TwoDigitJudge(nn.Module):
    """
    Classifier for 2-digit MNIST images (28x56).

    Two evaluation modes:
    - full_number: Single 100-class output for the complete number
    - per_position: Two 10-class outputs for tens and ones digits

    Architecture: Simple CNN adapted for wider input.
    """

    def __init__(
        self,
        hidden_dims: list[int] = [64, 128, 256],
        dropout: float = 0.3,
    ):
        super().__init__()

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
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images [B, 1, 28, 56]
            mode: "full" for 100-class output, "per_position" for two 10-class outputs

        Returns:
            If mode="full": logits [B, 100]
            If mode="per_position": (tens_logits [B, 10], ones_logits [B, 10])
        """
        features = self.extract_features(x)

        if mode == "full":
            return self.fc_full(features)
        elif mode == "per_position":
            return self.fc_tens(features), self.fc_ones(features)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


def create_twodigit_judge(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    freeze: bool = True,
) -> TwoDigitJudge:
    """
    Create a TwoDigitJudge, optionally loading from checkpoint.

    Args:
        checkpoint_path: Path to pretrained weights
        device: Device to load model to
        freeze: Whether to freeze weights after loading

    Returns:
        Initialized TwoDigitJudge
    """
    device = device or torch.device("cpu")
    model = TwoDigitJudge()

    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)

    if freeze:
        model.freeze()

    return model


def train_twodigit_judge(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.001,
    save_path: str = "checkpoints/judge_twodigit.pt",
    device: Optional[torch.device] = None,
) -> TwoDigitJudge:
    """
    Train a TwoDigitJudge on 2-digit MNIST.

    Uses both full-number loss and per-position loss for robust learning.
    """
    import torch.optim as optim
    from src.data.multidigit import TwoDigitMNIST
    from torch.utils.data import DataLoader

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training TwoDigitJudge on {device}")

    # Data
    train_dataset = TwoDigitMNIST(train=True, num_samples=50000)
    test_dataset = TwoDigitMNIST(train=False, num_samples=10000, seed=123)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = TwoDigitJudge().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_full = 0
        correct_tens = 0
        correct_ones = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            tens = labels // 10
            ones = labels % 10

            optimizer.zero_grad()

            # Full number prediction
            logits_full = model(images, mode="full")
            loss_full = F.cross_entropy(logits_full, labels)

            # Per-position prediction
            logits_tens, logits_ones = model(images, mode="per_position")
            loss_tens = F.cross_entropy(logits_tens, tens)
            loss_ones = F.cross_entropy(logits_ones, ones)

            # Combined loss
            loss = loss_full + 0.5 * (loss_tens + loss_ones)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct_full += (logits_full.argmax(dim=1) == labels).sum().item()
            correct_tens += (logits_tens.argmax(dim=1) == tens).sum().item()
            correct_ones += (logits_ones.argmax(dim=1) == ones).sum().item()
            total += labels.size(0)

        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images, mode="full")
                test_correct += (logits.argmax(dim=1) == labels).sum().item()
                test_total += labels.size(0)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {total_loss/len(train_loader):.4f} | "
            f"Train Full: {100*correct_full/total:.1f}% | "
            f"Tens: {100*correct_tens/total:.1f}% | "
            f"Ones: {100*correct_ones/total:.1f}% | "
            f"Test: {100*test_correct/test_total:.1f}%"
        )

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "epochs": epochs,
        "test_accuracy": test_correct / test_total,
    }, save_path)
    print(f"Saved to {save_path}")

    return model


if __name__ == "__main__":
    train_twodigit_judge(epochs=10)
