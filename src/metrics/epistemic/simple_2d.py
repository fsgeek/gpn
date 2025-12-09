"""
Simple 2D Epistemic Metric (Approach A)

Philosophy: Simplest possible decomposition - two orthogonal axes

Metrics:
- X-axis: Alignment = 1 - MSE(v_pred, v_seen)
- Y-axis: Correctness = Judge accuracy

State Space Quadrants:
- Top-right: Genuine synchronization (high alignment, high correctness)
- Top-left: Collusion (high alignment, low correctness - both wrong but agreeing)
- Bottom-right: Healthy learning (low alignment, high correctness - Judge correct, Weaver catching up)
- Bottom-left: Chaos/early training (low alignment, low correctness)

Pros:
- Extremely interpretable
- Minimal computational overhead (~0.1-0.5ms)
- Clear actionable signal

Cons:
- Cannot distinguish uncertainty (I) from falsity (F)
- No explicit mode collapse detection
"""

import torch

from src.metrics.epistemic.base import EpistemicMetric, EpistemicState


class Simple2DMetric(EpistemicMetric):
    """
    Simple 2D metric: Alignment + Correctness

    Tracks epistemic state using two primary dimensions.
    """

    def __init__(
        self,
        random_sampling: bool = False,
        sample_rate: float = 0.5,
        device: torch.device | None = None,
    ):
        """
        Initialize Simple 2D metric.

        Args:
            random_sampling: Enable sparse tensor sampling
            sample_rate: Fraction of metrics to compute when sampling
            device: Device for tensor operations
        """
        super().__init__(
            name='simple_2d',
            random_sampling=random_sampling,
            sample_rate=sample_rate,
            device=device,
        )

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
        Compute Simple 2D metrics.

        Args:
            step: Current training step
            v_pred: Weaver's value predictions [B]
            v_seen: Witness's value predictions [B]
            judge_logits: Judge classifier outputs [B, num_classes]
            witness_logits: Witness classifier outputs [B, num_classes]
            labels: Ground truth labels [B]
            fake_images: Generated images [B, C, H, W]

        Returns:
            EpistemicState with alignment and correctness
        """
        # Primary metrics
        alignment = self.compute_alignment(v_pred, v_seen)
        correctness = self.compute_correctness(judge_logits, labels)

        # Secondary metrics (for deeper analysis)
        disagreement = self.compute_disagreement(judge_logits, witness_logits)
        witness_correctness = self._compute_witness_correctness(witness_logits, labels)

        # Improvement tracking (if history available)
        alignment_improvement = self.get_recent_improvement('alignment', window=100)
        correctness_improvement = self.get_recent_improvement('correctness', window=100)

        # Quadrant classification
        quadrant = self._classify_quadrant(alignment, correctness)

        # Velocity (rate of change)
        velocity = self._compute_velocity(
            alignment, correctness,
            alignment_improvement, correctness_improvement
        )

        metrics = {
            # Primary dimensions
            'alignment': alignment,
            'correctness': correctness,

            # Secondary signals
            'disagreement': disagreement,
            'witness_correctness': witness_correctness,

            # Improvement tracking
            'alignment_improvement': alignment_improvement,
            'correctness_improvement': correctness_improvement,

            # Derived interpretations
            'quadrant': quadrant,
            'velocity': velocity,
        }

        metadata = {
            'quadrant_name': self._quadrant_name(quadrant),
            'interpretation': self._interpret_state(alignment, correctness, velocity),
        }

        state = EpistemicState(
            step=step,
            approach=self.name,
            metrics=metrics,
            metadata=metadata,
        )

        self.update_history(state)
        return state

    def _compute_witness_correctness(
        self,
        witness_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Compute Witness classification accuracy.

        Args:
            witness_logits: Witness classifier outputs [B, num_classes]
            labels: Ground truth labels [B]

        Returns:
            Accuracy [0, 1]
        """
        if not self._should_compute_metric('witness_correctness'):
            return 0.0

        preds = witness_logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()
        return accuracy

    def _classify_quadrant(
        self,
        alignment: float,
        correctness: float,
        threshold: float = 0.5,
    ) -> int:
        """
        Classify state into one of four quadrants.

        Args:
            alignment: Alignment score [0, 1]
            correctness: Correctness score [0, 1]
            threshold: Boundary between high/low

        Returns:
            Quadrant number (0-3):
                0: Bottom-left (chaos/early training)
                1: Bottom-right (healthy learning)
                2: Top-left (collusion)
                3: Top-right (genuine synchronization)
        """
        high_align = alignment > threshold
        high_correct = correctness > threshold

        if not high_align and not high_correct:
            return 0  # Bottom-left
        elif not high_align and high_correct:
            return 1  # Bottom-right
        elif high_align and not high_correct:
            return 2  # Top-left (DANGER: collusion)
        else:
            return 3  # Top-right

    def _quadrant_name(self, quadrant: int) -> str:
        """Get human-readable quadrant name."""
        names = {
            0: 'chaos/early_training',
            1: 'healthy_learning',
            2: 'collusion',
            3: 'genuine_synchronization',
        }
        return names.get(quadrant, 'unknown')

    def _compute_velocity(
        self,
        alignment: float,
        correctness: float,
        alignment_improvement: float,
        correctness_improvement: float,
    ) -> float:
        """
        Compute velocity in 2D state space.

        Fast convergence (high velocity) can indicate gaming.

        Args:
            alignment: Current alignment
            correctness: Current correctness
            alignment_improvement: Recent change in alignment
            correctness_improvement: Recent change in correctness

        Returns:
            Velocity magnitude (Euclidean distance of improvement vector)
        """
        velocity = (alignment_improvement**2 + correctness_improvement**2)**0.5
        return velocity

    def _interpret_state(
        self,
        alignment: float,
        correctness: float,
        velocity: float,
    ) -> str:
        """
        Generate human-readable interpretation of current state.

        Args:
            alignment: Alignment score
            correctness: Correctness score
            velocity: Rate of change

        Returns:
            Interpretation string
        """
        quadrant = self._classify_quadrant(alignment, correctness)

        if quadrant == 0:
            return "Early training: system still learning basics"
        elif quadrant == 1:
            return "Healthy learning: Judge working, Weaver catching up"
        elif quadrant == 2:
            if velocity > 0.5:
                return "WARNING: Rapid collusion - both wrong but agreeing quickly"
            else:
                return "WARNING: Collusion - both wrong but agreeing"
        else:  # quadrant == 3
            if velocity > 0.5:
                return "SUSPICIOUS: Very fast convergence - possible gaming"
            else:
                return "Good: Genuine synchronization"
