"""
Recovery tracking for intervention experiments.

Tracks metrics before/during/after intervention to quantify recovery success.
Provides statistical measures of recovery fraction and success rate.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RecoveryMetrics:
    """Metrics for a single recovery attempt."""
    intervention_step: int
    pre_metrics: Dict[str, float]
    post_metrics: Dict[str, float]
    recovery_trajectory: List[Dict[str, float]] = field(default_factory=list)

    # Baseline values for recovery computation
    baseline_mode_collapse: float = 0.30
    baseline_F: float = 0.41
    baseline_correctness: float = 0.21

    @property
    def recovery_achieved(self) -> bool:
        """Check if recovery target was achieved (50% gap closure)."""
        return self.recovery_fraction >= 0.5

    @property
    def recovery_fraction(self) -> float:
        """
        Fraction of pathology gap closed.

        Computed as: (pre - post) / (pre - baseline)
        Where:
            - pre: metric value at intervention
            - post: metric value after recovery window
            - baseline: healthy training value

        Returns:
            Float in [0, 1] indicating recovery fraction (1 = full recovery)
        """
        pre = self.pre_metrics.get('mode_collapse', 1.0)
        post = self.post_metrics.get('mode_collapse', 1.0)
        baseline = self.baseline_mode_collapse

        gap = pre - baseline
        if gap <= 0:
            return 1.0  # Was never pathological

        closure = pre - post
        return min(1.0, max(0.0, closure / gap))

    @property
    def recovery_time(self) -> Optional[int]:
        """Steps to achieve 50% recovery (None if not achieved)."""
        if not self.recovery_trajectory:
            return None

        pre = self.pre_metrics.get('mode_collapse', 1.0)
        baseline = self.baseline_mode_collapse
        target = pre - 0.5 * (pre - baseline)  # 50% closure target

        for entry in self.recovery_trajectory:
            if entry.get('mode_collapse', 1.0) <= target:
                return entry['step'] - self.intervention_step

        return None  # Never achieved 50% recovery

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'intervention_step': self.intervention_step,
            'pre_metrics': self.pre_metrics,
            'post_metrics': self.post_metrics,
            'recovery_fraction': self.recovery_fraction,
            'recovery_achieved': self.recovery_achieved,
            'recovery_time': self.recovery_time,
            'trajectory_length': len(self.recovery_trajectory),
        }


class RecoveryTracker:
    """
    Tracks recovery across intervention events.

    Usage:
        tracker = RecoveryTracker(recovery_window=500)

        # When intervention triggers:
        tracker.start_recovery(step, pre_metrics)

        # Each step during recovery:
        tracker.record_step(step, current_metrics)

        # Get summary:
        summary = tracker.get_summary()
    """

    def __init__(
        self,
        recovery_window: int = 500,
        baseline_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize recovery tracker.

        Args:
            recovery_window: Steps to track after intervention
            baseline_metrics: Expected healthy training values
        """
        self.recovery_window = recovery_window
        self.baseline_metrics = baseline_metrics or {
            'mode_collapse': 0.30,
            'F': 0.41,
            'correctness': 0.21,
        }

        self.recoveries: List[RecoveryMetrics] = []
        self.current_recovery: Optional[Dict[str, Any]] = None
        self.is_tracking: bool = False

    def start_recovery(self, step: int, pre_metrics: Dict[str, float]):
        """
        Start tracking a recovery attempt.

        Args:
            step: Step when intervention was applied
            pre_metrics: Metrics at intervention time
        """
        self.current_recovery = {
            'intervention_step': step,
            'pre_metrics': {k: float(v) for k, v in pre_metrics.items()
                          if isinstance(v, (int, float))},
            'trajectory': [],
        }
        self.is_tracking = True
        logger.info(f"Recovery tracking started at step {step}")

    def record_step(self, step: int, metrics: Dict[str, float]):
        """
        Record metrics during recovery window.

        Args:
            step: Current training step
            metrics: Current epistemic metrics
        """
        if not self.is_tracking or self.current_recovery is None:
            return

        # Extract numeric metrics
        numeric_metrics = {k: float(v) for k, v in metrics.items()
                         if isinstance(v, (int, float))}
        numeric_metrics['step'] = step

        self.current_recovery['trajectory'].append(numeric_metrics)

        # Check if recovery window complete
        intervention_step = self.current_recovery['intervention_step']
        if step >= intervention_step + self.recovery_window:
            self._finalize_recovery(numeric_metrics)

    def _finalize_recovery(self, final_metrics: Dict[str, float]):
        """Finalize recovery attempt and compute statistics."""
        if self.current_recovery is None:
            return

        recovery = RecoveryMetrics(
            intervention_step=self.current_recovery['intervention_step'],
            pre_metrics=self.current_recovery['pre_metrics'],
            post_metrics=final_metrics,
            recovery_trajectory=self.current_recovery['trajectory'],
            baseline_mode_collapse=self.baseline_metrics.get('mode_collapse', 0.30),
            baseline_F=self.baseline_metrics.get('F', 0.41),
            baseline_correctness=self.baseline_metrics.get('correctness', 0.21),
        )

        self.recoveries.append(recovery)
        self.current_recovery = None
        self.is_tracking = False

        logger.info(f"Recovery finalized: fraction={recovery.recovery_fraction:.3f}, "
                   f"achieved={recovery.recovery_achieved}")

    def force_finalize(self, final_metrics: Dict[str, float]):
        """Force finalization even if window not complete."""
        if self.is_tracking and self.current_recovery is not None:
            self._finalize_recovery(final_metrics)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics across all recovery attempts.

        Returns:
            Dict with recovery statistics
        """
        if not self.recoveries:
            return {
                'num_recoveries': 0,
                'success_rate': 0.0,
                'mean_recovery_fraction': 0.0,
                'std_recovery_fraction': 0.0,
            }

        fractions = [r.recovery_fraction for r in self.recoveries]
        successes = [r.recovery_achieved for r in self.recoveries]
        recovery_times = [r.recovery_time for r in self.recoveries if r.recovery_time is not None]

        return {
            'num_recoveries': len(self.recoveries),
            'success_rate': float(np.mean(successes)),
            'mean_recovery_fraction': float(np.mean(fractions)),
            'std_recovery_fraction': float(np.std(fractions)),
            'min_recovery_fraction': float(np.min(fractions)),
            'max_recovery_fraction': float(np.max(fractions)),
            'mean_recovery_time': float(np.mean(recovery_times)) if recovery_times else None,
            'all_fractions': fractions,
            'all_successes': successes,
            'recovery_details': [r.to_dict() for r in self.recoveries],
        }

    def reset(self):
        """Reset tracker for new experiment."""
        self.recoveries = []
        self.current_recovery = None
        self.is_tracking = False
