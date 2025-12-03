"""
Convergence metrics for GPN-1.

Compares convergence speed between GPN and baseline GAN.

Exports:
    - ConvergenceMetrics: Convergence speed tracking
    - ConvergenceComparison: Statistical comparison
    - StatisticalResult: Statistical test results
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class StatisticalResult:
    """Result of statistical significance test."""

    significant: bool  # Is difference significant?
    p_value: float  # P-value of test
    effect_size: float  # Cohen's d effect size
    confidence_interval: tuple[float, float]  # 95% CI


@dataclass
class ConvergencePoint:
    """A convergence milestone."""

    step: int
    metric_value: float
    threshold: float


@dataclass
class ConvergenceComparison:
    """Comparison of convergence between two training runs."""

    gpn_convergence_step: Optional[int]
    gan_convergence_step: Optional[int]
    speedup_ratio: Optional[float]  # GAN steps / GPN steps
    gpn_metrics: list[tuple[int, float]]
    gan_metrics: list[tuple[int, float]]
    statistical_test: Optional[StatisticalResult]


class ConvergenceMetrics:
    """
    Track and compare convergence speed.

    Convergence is defined as reaching a threshold quality metric
    (e.g., Judge accuracy > 0.9) and maintaining it.

    Compares GPN vs baseline GAN to measure speedup from
    cooperative vs adversarial training.
    """

    def __init__(
        self,
        convergence_threshold: float = 0.9,
        stability_window: int = 10,
        metric_name: str = "judge_accuracy",
    ) -> None:
        """
        Initialize convergence metrics.

        Args:
            convergence_threshold: Quality threshold for convergence
            stability_window: Steps to maintain threshold
            metric_name: Name of metric being tracked
        """
        self.convergence_threshold = convergence_threshold
        self.stability_window = stability_window
        self.metric_name = metric_name

        self._history: list[tuple[int, float]] = []
        self._convergence_step: Optional[int] = None
        self._above_threshold_count = 0

    def update(self, step: int, metric_value: float) -> bool:
        """
        Update with new metric value.

        Args:
            step: Training step
            metric_value: Current metric value

        Returns:
            True if convergence just detected, False otherwise
        """
        self._history.append((step, metric_value))

        # Check for convergence
        if self._convergence_step is None:
            if metric_value >= self.convergence_threshold:
                self._above_threshold_count += 1
                if self._above_threshold_count >= self.stability_window:
                    # Find first step that started the stable window
                    self._convergence_step = self._history[-self.stability_window][0]
                    return True
            else:
                self._above_threshold_count = 0

        return False

    @property
    def converged(self) -> bool:
        """Whether convergence has been detected."""
        return self._convergence_step is not None

    @property
    def convergence_step(self) -> Optional[int]:
        """Step at which convergence was detected."""
        return self._convergence_step

    @property
    def history(self) -> list[tuple[int, float]]:
        """Full history of (step, metric) pairs."""
        return self._history.copy()

    @property
    def best_value(self) -> float:
        """Best metric value achieved."""
        if not self._history:
            return 0.0
        return max(v for _, v in self._history)

    @property
    def latest_value(self) -> Optional[float]:
        """Most recent metric value."""
        if not self._history:
            return None
        return self._history[-1][1]

    def compare(
        self,
        other: "ConvergenceMetrics",
        run_statistical_test: bool = True,
    ) -> ConvergenceComparison:
        """
        Compare convergence with another run.

        Args:
            other: Another ConvergenceMetrics instance (e.g., GAN baseline)
            run_statistical_test: Whether to run statistical significance test

        Returns:
            ConvergenceComparison with speedup analysis
        """
        gpn_step = self._convergence_step
        gan_step = other._convergence_step

        # Calculate speedup
        if gpn_step is not None and gan_step is not None:
            speedup = gan_step / gpn_step if gpn_step > 0 else float('inf')
        else:
            speedup = None

        # Statistical test
        stat_result = None
        if run_statistical_test and len(self._history) > 10 and len(other._history) > 10:
            stat_result = self._statistical_test(other)

        return ConvergenceComparison(
            gpn_convergence_step=gpn_step,
            gan_convergence_step=gan_step,
            speedup_ratio=speedup,
            gpn_metrics=self._history,
            gan_metrics=other._history,
            statistical_test=stat_result,
        )

    def _statistical_test(self, other: "ConvergenceMetrics") -> StatisticalResult:
        """
        Run statistical significance test.

        Uses a simple t-test approximation for comparison.
        """
        # Get overlapping steps for comparison
        self_values = [v for _, v in self._history]
        other_values = [v for _, v in other._history]

        # Truncate to same length
        min_len = min(len(self_values), len(other_values))
        self_values = self_values[:min_len]
        other_values = other_values[:min_len]

        # Calculate means and stds
        self_mean = sum(self_values) / len(self_values)
        other_mean = sum(other_values) / len(other_values)

        self_var = sum((v - self_mean) ** 2 for v in self_values) / (len(self_values) - 1)
        other_var = sum((v - other_mean) ** 2 for v in other_values) / (len(other_values) - 1)

        self_std = math.sqrt(self_var) if self_var > 0 else 1e-8
        other_std = math.sqrt(other_var) if other_var > 0 else 1e-8

        # Pooled std for effect size
        pooled_std = math.sqrt((self_var + other_var) / 2)

        # Cohen's d effect size
        effect_size = (self_mean - other_mean) / pooled_std if pooled_std > 0 else 0

        # Simple t-statistic
        se = math.sqrt(self_var / len(self_values) + other_var / len(other_values))
        t_stat = (self_mean - other_mean) / se if se > 0 else 0

        # Approximate p-value (two-tailed)
        # Using normal approximation for large n
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

        # 95% CI for difference
        diff = self_mean - other_mean
        margin = 1.96 * se
        ci = (diff - margin, diff + margin)

        return StatisticalResult(
            significant=p_value < 0.05,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
        )

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximate normal CDF using error function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def state_dict(self) -> dict:
        """Export state for checkpointing."""
        return {
            "history": self._history,
            "convergence_step": self._convergence_step,
            "above_threshold_count": self._above_threshold_count,
            "threshold": self.convergence_threshold,
            "stability_window": self.stability_window,
            "metric_name": self.metric_name,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self._history = state["history"]
        self._convergence_step = state["convergence_step"]
        self._above_threshold_count = state["above_threshold_count"]
        self.convergence_threshold = state["threshold"]
        self.stability_window = state["stability_window"]
        self.metric_name = state["metric_name"]
