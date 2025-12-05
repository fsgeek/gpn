"""
Linear probe models for testing compositional structure in representations.

Three probes test progressively stronger compositionality:
1. Digit Identity: Can we recover which digit? (baseline - both should pass)
2. Spatial Position: Can we recover left/right placement? (tests spatial encoding)
3. Stroke Structure: Can we decode edge maps? (tests part-whole decomposition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitIdentityProbe(nn.Module):
    """
    Linear probe for digit identity classification.

    Tests: Can we linearly decode which digit (0-9) from intermediate features?
    Both pedagogical and adversarial should pass this (they both classify well).
    """

    def __init__(self, input_dim: int, num_classes: int = 10):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Flattened intermediate features [B, D]

        Returns:
            Class logits [B, num_classes]
        """
        if len(features.shape) > 2:
            features = features.flatten(1)
        return self.classifier(features)


class SpatialPositionProbe(nn.Module):
    """
    Linear probe for spatial position encoding in 2-digit compositions.

    Tests: Given features from a 2-digit image, can we identify which digit
    is on the left vs right? This requires spatially-structured representations.

    Hypothesis:
    - Pedagogical: PASS (learns reusable spatial structure)
    - Adversarial: FAIL (holistic, no spatial decomposition)
    """

    def __init__(self, feature_channels: int, num_digits: int = 10):
        """
        Args:
            feature_channels: Number of channels in feature map
            num_digits: Number of digit classes (10 for MNIST)
        """
        super().__init__()

        # Separate classifiers for left and right positions
        # Take features from left/right halves of 2-digit feature map
        self.left_classifier = nn.Linear(feature_channels, num_digits)
        self.right_classifier = nn.Linear(feature_channels, num_digits)

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Spatial feature maps [B, C, H, W] where W = 2*H (2-digit)

        Returns:
            Tuple of (left_logits [B, 10], right_logits [B, 10])
        """
        B, C, H, W = features.shape

        # Split into left and right halves
        mid = W // 2
        left_features = features[:, :, :, :mid]  # [B, C, H, W/2]
        right_features = features[:, :, :, mid:]  # [B, C, H, W/2]

        # Global average pooling over spatial dimensions
        left_pooled = left_features.mean(dim=[2, 3])  # [B, C]
        right_pooled = right_features.mean(dim=[2, 3])  # [B, C]

        # Classify each position
        left_logits = self.left_classifier(left_pooled)
        right_logits = self.right_classifier(right_pooled)

        return left_logits, right_logits


class StrokeStructureProbe(nn.Module):
    """
    Convolutional probe for stroke-level structure.

    Tests: Can we decode edge maps from intermediate features?
    This tests whether the generator learns decomposable visual primitives
    (strokes, edges) vs holistic patterns.

    Hypothesis:
    - Pedagogical: PASS (learns reusable stroke features)
    - Adversarial: FAIL (holistic patterns, no part-whole decomposition)
    """

    def __init__(self, input_channels: int):
        """
        Args:
            input_channels: Number of channels in input feature maps
        """
        super().__init__()

        # Simple decoder: feature channels -> edge map
        # Keep it linear (1x1 conv) to test if structure exists in features
        self.decoder = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),  # Edge probability
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Intermediate feature maps [B, C, H, W]

        Returns:
            Predicted edge maps [B, 1, H, W]
        """
        return self.decoder(features)


def train_digit_identity_probe(
    probe: DigitIdentityProbe,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 0.001,
) -> float:
    """
    Train digit identity probe and return final accuracy.

    Args:
        probe: Probe model
        features: Extracted features [N, D]
        labels: Ground truth labels [N]
        device: Device to train on
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate

    Returns:
        Final test accuracy
    """
    probe = probe.to(device)
    features = features.to(device)
    labels = labels.to(device)

    # Split train/test
    n_train = int(0.8 * len(features))
    train_features, test_features = features[:n_train], features[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    probe.train()
    for epoch in range(epochs):
        for i in range(0, len(train_features), batch_size):
            batch_features = train_features[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            logits = probe(batch_features)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    probe.eval()
    with torch.no_grad():
        test_logits = probe(test_features)
        predictions = test_logits.argmax(dim=1)
        accuracy = (predictions == test_labels).float().mean().item()

    return accuracy
