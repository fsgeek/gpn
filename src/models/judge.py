"""
Judge network for GPN-1.

The Judge is a frozen external classifier that provides grounding signal.
Per contracts/models.md: JudgeInterface with forward(images) -> class_logits.

The Judge is trained to high accuracy (>95%) on MNIST, then frozen during
GPN training. It serves as the external reference point that keeps the
Weaver/Witness relationship grounded in actual digit classification.

Exports:
    - Judge: Frozen MNIST classifier
    - create_judge: Factory function
"""

from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn


class Judge(nn.Module):
    """
    Frozen MNIST classifier for grounding signal.

    The Judge provides external classification that the Weaver must satisfy.
    It is trained separately and frozen during GPN training.

    Implements JudgeInterface per contracts/models.md:
    - forward(images: Tensor[B, 1, 28, 28]) -> Tensor[B, 10] (class logits)
    - freeze(): Disable gradient computation
    - accuracy(images, labels) -> float: Compute classification accuracy

    Attributes:
        hidden_dims: Hidden layer dimensions
        is_frozen: Whether parameters are frozen
    """

    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 28,
        num_classes: int = 10,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.25,
    ) -> None:
        """
        Initialize Judge network.

        Args:
            image_channels: Number of input channels (1 for MNIST)
            image_size: Input image size (28 for MNIST)
            num_classes: Number of output classes (10 for MNIST)
            hidden_dims: Hidden layer dimensions (default: [512, 256])
            dropout: Dropout rate for regularization
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.image_channels = image_channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self._is_frozen = False

        # Feature extractor (CNN)
        self.features = nn.Sequential(
            # Conv block 1: 28x28 -> 14x14
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            # Conv block 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )

        # Calculate flattened size
        feature_size = 64 * (image_size // 4) ** 2  # After two 2x2 max pools

        # Classifier head
        layers: list[nn.Module] = [nn.Flatten()]
        prev_dim = feature_size

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Classify images.

        Args:
            images: Input images [B, 1, 28, 28]

        Returns:
            Class logits [B, 10]
        """
        features = self.features(images)
        logits = self.classifier(features)
        return logits

    def freeze(self) -> None:
        """
        Freeze all parameters (disable gradients).

        Call this after training to prevent updates during GPN training.
        """
        self._is_frozen = True
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze parameters (enable gradients)."""
        self._is_frozen = False
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    @property
    def is_frozen(self) -> bool:
        """Check if Judge is frozen."""
        return self._is_frozen

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
            logits = self.forward(images)
            predictions = logits.argmax(dim=1)
            correct = (predictions == labels).float().mean()
            return correct.item()

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions.

        Args:
            images: Input images [B, 1, 28, 28]

        Returns:
            Predicted class indices [B]
        """
        with torch.no_grad():
            logits = self.forward(images)
            return logits.argmax(dim=1)


def create_judge(
    hidden_dims: Optional[list[int]] = None,
    checkpoint_path: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
    freeze: bool = True,
) -> Judge:
    """
    Factory function to create and optionally load a Judge.

    Args:
        hidden_dims: Hidden layer dimensions
        checkpoint_path: Path to load weights from
        device: Device to place model on
        freeze: Whether to freeze after loading

    Returns:
        Initialized (and optionally frozen) Judge

    Example:
        >>> judge = create_judge(
        ...     checkpoint_path="checkpoints/judge.pt",
        ...     freeze=True,
        ... )
    """
    judge = Judge(hidden_dims=hidden_dims)

    if device is not None:
        judge = judge.to(device)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            state_dict = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
            # Handle both direct state_dict and wrapped checkpoint
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            judge.load_state_dict(state_dict)

    if freeze:
        judge.freeze()

    return judge
