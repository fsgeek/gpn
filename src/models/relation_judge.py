"""
Relation Judge: Validates X > Y constraint in generated images

Architecture:
- Input: [X][>][Y] image (1 x 28 x 84)
- Extract X and Y digits from left and right thirds
- Classify each digit (0-9)
- Output: P(X > Y)

This provides pedagogical signal for relational generation:
The generator learns to satisfy "X > Y" by maximizing judge accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationJudge(nn.Module):
    """
    Judge that validates relational constraints in generated images.

    Takes [X][>][Y] images and outputs P(valid relation).
    """

    def __init__(self):
        super().__init__()

        # Digit classifier (shared for X and Y)
        self.digit_classifier = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),  # 10 digit classes
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validate relational images.

        Args:
            images: [X][>][Y] images [B, 1, 28, 84]

        Returns:
            x_logits: Digit classification for left position [B, 10]
            y_logits: Digit classification for right position [B, 10]
            valid_prob: P(X > Y) based on predicted digits [B]
        """
        B = images.size(0)

        # Extract left and right digit regions (ignore middle ">" symbol)
        x_img = images[:, :, :, :28]  # Left third [B, 1, 28, 28]
        y_img = images[:, :, :, 56:]  # Right third [B, 1, 28, 28]

        # Classify digits
        x_logits = self.digit_classifier(x_img)  # [B, 10]
        y_logits = self.digit_classifier(y_img)  # [B, 10]

        # Compute P(X > Y) from predicted distributions
        # P(X > Y) = sum over all (i, j) where i > j of P(X=i) * P(Y=j)
        x_probs = F.softmax(x_logits, dim=1)  # [B, 10]
        y_probs = F.softmax(y_logits, dim=1)  # [B, 10]

        # Create mask for i > j
        i_vals = torch.arange(10, device=images.device).view(10, 1)  # [10, 1]
        j_vals = torch.arange(10, device=images.device).view(1, 10)  # [1, 10]
        valid_mask = (i_vals > j_vals).float()  # [10, 10]

        # P(X > Y) = sum_{i,j: i>j} P(X=i) P(Y=j)
        # = sum_i sum_j P(X=i) P(Y=j) * mask[i,j]
        joint_probs = x_probs.unsqueeze(2) * y_probs.unsqueeze(1)  # [B, 10, 10]
        valid_prob = (joint_probs * valid_mask).sum(dim=[1, 2])  # [B]

        return x_logits, y_logits, valid_prob

    def compute_accuracy(
        self,
        images: torch.Tensor,
        x_labels: torch.Tensor,
        y_labels: torch.Tensor,
    ) -> dict[str, float]:
        """
        Compute accuracy metrics.

        Args:
            images: [X][>][Y] images [B, 1, 28, 84]
            x_labels: Ground truth X digits [B]
            y_labels: Ground truth Y digits [B]

        Returns:
            Dictionary of metrics
        """
        with torch.no_grad():
            x_logits, y_logits, valid_prob = self(images)

            # Digit classification accuracy
            x_pred = x_logits.argmax(dim=1)
            y_pred = y_logits.argmax(dim=1)

            x_acc = (x_pred == x_labels).float().mean().item()
            y_acc = (y_pred == y_labels).float().mean().item()

            # Relation validity accuracy
            # Ground truth: X > Y
            valid_gt = (x_labels > y_labels).float()
            valid_pred = (valid_prob > 0.5).float()
            relation_acc = (valid_pred == valid_gt).float().mean().item()

            # Average validity probability for valid relations
            avg_valid_prob = valid_prob.mean().item()

        return {
            "x_accuracy": x_acc,
            "y_accuracy": y_acc,
            "relation_accuracy": relation_acc,
            "avg_valid_prob": avg_valid_prob,
        }


def create_relation_judge(
    checkpoint_path: str | None = None,
    device: torch.device | None = None,
    freeze: bool = True,
) -> RelationJudge:
    """
    Create RelationJudge, optionally loading from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint (if resuming)
        device: Device to load to
        freeze: Whether to freeze parameters

    Returns:
        RelationJudge model
    """
    device = device or torch.device("cpu")
    judge = RelationJudge().to(device)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "models" in checkpoint and "judge" in checkpoint["models"]:
            judge.load_state_dict(checkpoint["models"]["judge"])
        elif "model" in checkpoint:
            # Checkpoint from train_relation_judge wraps state_dict in "model" key
            judge.load_state_dict(checkpoint["model"])
        else:
            judge.load_state_dict(checkpoint)

    if freeze:
        for param in judge.parameters():
            param.requires_grad = False
        judge.eval()

    return judge
