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


def train_spatial_position_probe(
    probe: SpatialPositionProbe,
    features: torch.Tensor,
    left_labels: torch.Tensor,
    right_labels: torch.Tensor,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.001,
) -> tuple[float, float]:
    """
    Train spatial position probe and return accuracies for both positions.

    Args:
        probe: Probe model
        features: Extracted spatial feature maps [N, C, H, W]
        left_labels: Ground truth labels for left digit [N]
        right_labels: Ground truth labels for right digit [N]
        device: Device to train on
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate

    Returns:
        Tuple of (left_accuracy, right_accuracy)
    """
    probe = probe.to(device)
    features = features.to(device)
    left_labels = left_labels.to(device)
    right_labels = right_labels.to(device)

    # Split train/test
    n_train = int(0.8 * len(features))
    train_features = features[:n_train]
    test_features = features[n_train:]
    train_left_labels = left_labels[:n_train]
    test_left_labels = left_labels[n_train:]
    train_right_labels = right_labels[:n_train]
    test_right_labels = right_labels[n_train:]

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    probe.train()
    for epoch in range(epochs):
        for i in range(0, len(train_features), batch_size):
            batch_features = train_features[i:i + batch_size]
            batch_left = train_left_labels[i:i + batch_size]
            batch_right = train_right_labels[i:i + batch_size]

            left_logits, right_logits = probe(batch_features)

            # Combined loss
            loss = criterion(left_logits, batch_left) + criterion(right_logits, batch_right)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    probe.eval()
    with torch.no_grad():
        test_left_logits, test_right_logits = probe(test_features)

        left_predictions = test_left_logits.argmax(dim=1)
        right_predictions = test_right_logits.argmax(dim=1)

        left_accuracy = (left_predictions == test_left_labels).float().mean().item()
        right_accuracy = (right_predictions == test_right_labels).float().mean().item()

    return left_accuracy, right_accuracy


def extract_edge_maps(
    images: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    Extract edge maps from images using Sobel filters.

    This creates ground truth targets for the stroke structure probe.

    Args:
        images: Input images [B, C, H, W]
        threshold: Threshold for edge detection

    Returns:
        Binary edge maps [B, 1, H, W]
    """
    # Sobel filters
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32).view(1, 1, 3, 3)

    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=torch.float32).view(1, 1, 3, 3)

    sobel_x = sobel_x.to(images.device)
    sobel_y = sobel_y.to(images.device)

    # Convert to grayscale if needed
    if images.shape[1] > 1:
        images = images.mean(dim=1, keepdim=True)

    # Apply Sobel filters
    edges_x = F.conv2d(images, sobel_x, padding=1)
    edges_y = F.conv2d(images, sobel_y, padding=1)

    # Magnitude
    edges = torch.sqrt(edges_x**2 + edges_y**2)

    # Normalize to [0, 1]
    edges = edges / (edges.max() + 1e-8)

    # Threshold
    edges = (edges > threshold).float()

    return edges


def train_stroke_structure_probe(
    probe: StrokeStructureProbe,
    features: torch.Tensor,
    images: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.001,
    edge_threshold: float = 0.1,
) -> tuple[float, float]:
    """
    Train stroke structure probe to decode edge maps from features.

    Args:
        probe: Probe model
        features: Extracted feature maps [N, C, H, W]
        images: Original images to extract edges from [N, C, H_img, W_img]
        device: Device to train on
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        edge_threshold: Threshold for edge extraction

    Returns:
        Tuple of (reconstruction_loss, edge_iou)
    """
    probe = probe.to(device)
    features = features.to(device)
    images = images.to(device)

    # Extract ground truth edge maps
    with torch.no_grad():
        edge_maps = extract_edge_maps(images, threshold=edge_threshold)

        # Resize edge maps to match feature map resolution
        _, _, H, W = features.shape
        if edge_maps.shape[-2:] != (H, W):
            edge_maps = F.interpolate(
                edge_maps,
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            )

    edge_maps = edge_maps.to(device)

    # Split train/test
    n_train = int(0.8 * len(features))
    train_features = features[:n_train]
    test_features = features[n_train:]
    train_edges = edge_maps[:n_train]
    test_edges = edge_maps[n_train:]

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Training loop
    probe.train()
    for epoch in range(epochs):
        for i in range(0, len(train_features), batch_size):
            batch_features = train_features[i:i + batch_size]
            batch_edges = train_edges[i:i + batch_size]

            predicted_edges = probe(batch_features)
            loss = criterion(predicted_edges, batch_edges)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    probe.eval()
    with torch.no_grad():
        test_predictions = probe(test_features)
        test_loss = criterion(test_predictions, test_edges).item()

        # Compute IoU (Intersection over Union)
        pred_binary = (test_predictions > 0.5).float()
        true_binary = test_edges

        intersection = (pred_binary * true_binary).sum()
        union = (pred_binary + true_binary).clamp(0, 1).sum()

        iou = (intersection / (union + 1e-8)).item()

    return test_loss, iou


def compare_probe_results(
    gpn_results: dict[str, float],
    gan_results: dict[str, float],
    probe_name: str,
) -> str:
    """
    Generate formatted comparison of probe results between GPN and GAN.

    Args:
        gpn_results: Dictionary of GPN probe metrics
        gan_results: Dictionary of GAN probe metrics
        probe_name: Name of the probe being compared

    Returns:
        Formatted comparison string
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"PROBE COMPARISON: {probe_name}")
    lines.append("=" * 80)
    lines.append("")

    # Compare each metric
    for metric_name in gpn_results.keys():
        gpn_val = gpn_results[metric_name]
        gan_val = gan_results[metric_name]
        delta = gpn_val - gan_val
        better = "GPN" if delta > 0 else "GAN"

        lines.append(f"{metric_name}:")
        lines.append(f"  GPN: {gpn_val:.4f}")
        lines.append(f"  GAN: {gan_val:.4f}")
        lines.append(f"  Δ (GPN - GAN): {delta:+.4f} ({better} better)")
        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)
