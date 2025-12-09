"""
Base classes for epistemic honesty instrumentation.

Defines abstract interface for measuring epistemic states during GPN training:
- Honest uncertainty ("I don't know")
- Collusion/gaming (both wrong but agreeing)
- Genuine synchronization (both right and agreeing)

Design Philosophy:
- Simple → Complex progression
- Support sparse tensor sampling (resistance to gaming)
- Minimal computational overhead
- Clear interpretability
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class EpistemicState:
    """
    Snapshot of epistemic metrics at a training step.

    Used for tracking, logging, and analysis across different approaches.
    """
    step: int
    approach: str  # 'simple_2d', 'bayesian', 'neutrosophic'
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate required fields."""
        if not self.approach:
            raise ValueError("approach must be specified")
        if self.step < 0:
            raise ValueError("step must be non-negative")


class EpistemicMetric(ABC):
    """
    Abstract base class for epistemic honesty metrics.

    All approaches (Simple 2D, Bayesian, Neutrosophic) inherit from this.
    """

    def __init__(
        self,
        name: str,
        random_sampling: bool = False,
        sample_rate: float = 0.5,
        device: torch.device | None = None,
    ):
        """
        Initialize epistemic metric.

        Args:
            name: Identifier for this approach
            random_sampling: Enable sparse tensor sampling (test for gaming)
            sample_rate: Fraction of metrics to compute when sampling (0-1)
            device: Device for tensor operations
        """
        self.name = name
        self.random_sampling = random_sampling
        self.sample_rate = sample_rate
        self.device = device or torch.device('cpu')

        # History tracking for improvement metrics
        self.history: list[EpistemicState] = []

    @abstractmethod
    def compute(
        self,
        step: int,
        v_pred: torch.Tensor,
        v_seen: torch.Tensor,
        judge_logits: torch.Tensor,
        witness_logits: torch.Tensor,
        labels: torch.Tensor,
        fake_images: torch.Tensor,
        **kwargs,
    ) -> EpistemicState:
        """
        Compute epistemic metrics for current training step.

        Args:
            step: Current training step
            v_pred: Weaver's value predictions [B]
            v_seen: Witness's value predictions [B]
            judge_logits: Judge classifier outputs [B, num_classes]
            witness_logits: Witness classifier outputs [B, num_classes]
            labels: Ground truth labels [B]
            fake_images: Generated images [B, C, H, W]
            **kwargs: Additional approach-specific arguments

        Returns:
            EpistemicState containing computed metrics
        """
        pass

    def _should_compute_metric(self, metric_name: str) -> bool:
        """
        Determine if metric should be computed (for sparse sampling).

        Args:
            metric_name: Name of metric to check

        Returns:
            True if metric should be computed
        """
        if not self.random_sampling:
            return True

        # Deterministic sampling based on metric name and step
        # Ensures reproducibility across runs
        seed = hash(metric_name) % 2**32
        torch.manual_seed(seed + len(self.history))
        return torch.rand(1).item() < self.sample_rate

    def update_history(self, state: EpistemicState):
        """
        Add state to history for improvement tracking.

        Args:
            state: Current epistemic state
        """
        self.history.append(state)

    def get_recent_improvement(
        self,
        metric_key: str,
        window: int = 100,
    ) -> float:
        """
        Compute recent improvement in a specific metric.

        Args:
            metric_key: Key of metric to track
            window: Number of steps to look back

        Returns:
            Improvement (positive = getting better)
        """
        if len(self.history) < 2:
            return 0.0

        recent = [
            s.metrics.get(metric_key, 0.0)
            for s in self.history[-window:]
            if metric_key in s.metrics
        ]

        if len(recent) < 2:
            return 0.0

        # Simple linear trend
        return recent[-1] - recent[0]

    def compute_alignment(
        self,
        v_pred: torch.Tensor,
        v_seen: torch.Tensor,
    ) -> float:
        """
        Compute alignment between Weaver and Witness predictions.

        Standard metric used by all approaches.

        Args:
            v_pred: Weaver's value predictions [B]
            v_seen: Witness's value predictions [B]

        Returns:
            Alignment score (1 - normalized MSE), range [0, 1]
        """
        if not self._should_compute_metric('alignment'):
            return 0.0

        mse = torch.nn.functional.mse_loss(v_pred, v_seen).item()
        # Normalize: assume value range is roughly [0, 1]
        # alignment = 1 means perfect match, 0 means maximum divergence
        alignment = max(0.0, 1.0 - mse)
        return alignment

    def compute_correctness(
        self,
        judge_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Compute Judge classification accuracy.

        Standard metric used by all approaches.

        Args:
            judge_logits: Judge classifier outputs [B, num_classes]
            labels: Ground truth labels [B]

        Returns:
            Accuracy [0, 1]
        """
        if not self._should_compute_metric('correctness'):
            return 0.0

        preds = judge_logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()
        return accuracy

    def compute_disagreement(
        self,
        judge_logits: torch.Tensor,
        witness_logits: torch.Tensor,
    ) -> float:
        """
        Compute disagreement between Judge and Witness predictions.

        Used for detecting indeterminacy (honest uncertainty).

        Args:
            judge_logits: Judge classifier outputs [B, num_classes]
            witness_logits: Witness classifier outputs [B, num_classes]

        Returns:
            Disagreement rate [0, 1]
        """
        if not self._should_compute_metric('disagreement'):
            return 0.0

        judge_preds = judge_logits.argmax(dim=1)
        witness_preds = witness_logits.argmax(dim=1)
        disagreement = (judge_preds != witness_preds).float().mean().item()
        return disagreement

    def reset_history(self):
        """Clear history (useful for new training runs)."""
        self.history = []
