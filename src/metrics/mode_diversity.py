"""
Mode diversity metrics for GPN-1.

Detects mode collapse and measures coverage of digit classes.

Exports:
    - ModeDiversity: Mode collapse detection and coverage metrics
    - ModeDiversityResult: Result dataclass
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ModeDiversityResult:
    """Result of mode diversity evaluation."""

    mode_coverage: float  # Fraction of classes covered (0-1)
    entropy: float  # Distribution entropy (higher = more diverse)
    class_counts: list[int]  # Count per class
    missing_classes: list[int]  # Classes not generated
    is_collapsed: bool  # True if mode collapse detected
    dominant_class: Optional[int]  # Most frequent class if collapsed


class ModeDiversity:
    """
    Mode collapse detection and coverage metrics.

    Uses the Judge classifier to determine what class each generated
    image belongs to, then measures:
    - Mode coverage: What fraction of classes are represented
    - Entropy: How uniform is the distribution across classes
    - Collapse detection: Is generation dominated by one class

    Collapse is detected when:
    - Fewer than min_classes are represented, OR
    - One class has > max_dominance fraction of samples
    """

    def __init__(
        self,
        judge: nn.Module,
        num_classes: int = 10,
        min_classes: int = 8,
        max_dominance: float = 0.5,
    ) -> None:
        """
        Initialize mode diversity metrics.

        Args:
            judge: Frozen Judge classifier
            num_classes: Total number of classes (10 for MNIST)
            min_classes: Minimum classes required to avoid collapse
            max_dominance: Maximum fraction for single class
        """
        self.judge = judge
        self.num_classes = num_classes
        self.min_classes = min_classes
        self.max_dominance = max_dominance

    @torch.no_grad()
    def evaluate(
        self,
        images: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> ModeDiversityResult:
        """
        Evaluate mode diversity of generated images.

        Args:
            images: Generated images [B, 1, 28, 28]
            target_labels: Optional target labels (for conditional generation)

        Returns:
            ModeDiversityResult with diversity metrics
        """
        # Classify images with Judge
        logits = self.judge(images)
        predictions = logits.argmax(dim=1)

        # Count predictions per class
        class_counts = torch.zeros(self.num_classes, dtype=torch.long, device=images.device)
        for pred in predictions:
            class_counts[pred] += 1

        class_counts_list = class_counts.tolist()
        total = len(predictions)

        # Find missing classes
        missing_classes = [i for i, count in enumerate(class_counts_list) if count == 0]
        covered_classes = self.num_classes - len(missing_classes)
        mode_coverage = covered_classes / self.num_classes

        # Calculate entropy
        probs = class_counts.float() / total
        probs = probs[probs > 0]  # Remove zeros for log
        entropy = -(probs * probs.log()).sum().item()
        max_entropy = torch.tensor(self.num_classes).float().log().item()
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Check for collapse
        max_count = class_counts.max().item()
        dominance = max_count / total if total > 0 else 0
        dominant_class = class_counts.argmax().item() if dominance > self.max_dominance else None

        is_collapsed = (
            covered_classes < self.min_classes or
            dominance > self.max_dominance
        )

        return ModeDiversityResult(
            mode_coverage=mode_coverage,
            entropy=normalized_entropy,
            class_counts=class_counts_list,
            missing_classes=missing_classes,
            is_collapsed=is_collapsed,
            dominant_class=dominant_class,
        )

    def evaluate_conditional(
        self,
        images: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> ModeDiversityResult:
        """
        Evaluate mode diversity for conditional generation.

        Checks if generated images match their target labels.

        Args:
            images: Generated images [B, 1, 28, 28]
            target_labels: Target labels used for generation [B]

        Returns:
            ModeDiversityResult with conditional metrics
        """
        result = self.evaluate(images)

        # Also check conditional accuracy
        logits = self.judge(images)
        predictions = logits.argmax(dim=1)
        conditional_accuracy = (predictions == target_labels).float().mean().item()

        # Adjust collapse detection for conditional case
        # If conditional accuracy is low, generation isn't following labels
        if conditional_accuracy < 0.5:
            result.is_collapsed = True

        return result
