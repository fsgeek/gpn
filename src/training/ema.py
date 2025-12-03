"""
EMA (Exponential Moving Average) state tracking for GPN-1.

Tracks running statistics of v_seen for empowerment loss calculation.
Per contracts/training.md: EMAStateInterface with update on Witness forward only.

Exports:
    - EMAState: EMA tracking with stagnation detection (T027a)
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import logging

import torch

logger = logging.getLogger(__name__)


@dataclass
class StagnationState:
    """State for variance stagnation detection (T027a)."""

    variance_history: list[float] = field(default_factory=list)
    consecutive_stagnant: int = 0
    stagnation_detected: bool = False


class EMAState:
    """
    Exponential Moving Average state tracker.

    Tracks running mean and variance of v_seen values for use in
    empowerment loss calculation. Updates only occur on Witness
    forward passes (not Weaver-only passes).

    Implements EMAStateInterface per contracts/training.md:
    - update(values): Update EMA with new values
    - mean: Current EMA mean
    - variance: Current EMA variance
    - state_dict(): Export state for checkpointing
    - load_state_dict(state): Restore state

    Includes stagnation detection (T027a):
    - Detects when variance change < threshold for N consecutive steps
    - Logs warning with phase context when stagnation detected

    Attributes:
        decay: EMA decay factor (0.99 = slow adaptation)
        dim: Dimension of tracked values
        initialized: Whether EMA has been initialized with first batch
    """

    def __init__(
        self,
        dim: int = 16,
        decay: float = 0.99,
        variance_threshold: float = 1e-6,
        window_size: int = 100,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize EMA state tracker.

        Args:
            dim: Dimension of values to track
            decay: EMA decay factor (higher = slower adaptation)
            variance_threshold: Threshold for stagnation detection (T027a)
            window_size: Window for stagnation detection (T027a)
            device: Device for tensors
        """
        self.dim = dim
        self.decay = decay
        self.variance_threshold = variance_threshold
        self.window_size = window_size
        self.device = device or torch.device("cpu")

        # EMA statistics
        self._mean = torch.zeros(dim, device=self.device)
        self._variance = torch.ones(dim, device=self.device)
        self._initialized = False
        self._update_count = 0

        # Stagnation detection state (T027a)
        self._stagnation = StagnationState()
        self._current_phase: Optional[int] = None

    @property
    def mean(self) -> torch.Tensor:
        """Current EMA mean."""
        return self._mean

    @property
    def variance(self) -> torch.Tensor:
        """Current EMA variance."""
        return self._variance

    @property
    def initialized(self) -> bool:
        """Whether EMA has been initialized."""
        return self._initialized

    @property
    def update_count(self) -> int:
        """Number of updates performed."""
        return self._update_count

    def set_phase(self, phase: int) -> None:
        """Set current training phase for stagnation warnings."""
        self._current_phase = phase

    def update(self, values: torch.Tensor) -> None:
        """
        Update EMA with new values.

        Should only be called after Witness forward pass.

        Args:
            values: New v_seen values [B, dim]
        """
        if values.dim() != 2 or values.size(1) != self.dim:
            raise ValueError(f"Expected values of shape [B, {self.dim}], got {values.shape}")

        # Move to correct device
        values = values.to(self.device)

        # Compute batch statistics
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)

        if not self._initialized:
            # First batch - initialize directly
            self._mean = batch_mean.clone()
            self._variance = batch_var.clone() + 1e-8
            self._initialized = True
        else:
            # EMA update
            self._mean = self.decay * self._mean + (1 - self.decay) * batch_mean
            self._variance = self.decay * self._variance + (1 - self.decay) * batch_var

        self._update_count += 1

        # Stagnation detection (T027a)
        self._check_stagnation(batch_var)

    def _check_stagnation(self, batch_var: torch.Tensor) -> None:
        """
        Check for variance stagnation (T027a).

        Detects when variance change is below threshold for consecutive steps.
        """
        current_var = batch_var.mean().item()
        history = self._stagnation.variance_history

        if len(history) >= self.window_size:
            history.pop(0)
        history.append(current_var)

        if len(history) < 2:
            return

        # Check if variance change is below threshold
        var_change = abs(history[-1] - history[-2])

        if var_change < self.variance_threshold:
            self._stagnation.consecutive_stagnant += 1
        else:
            self._stagnation.consecutive_stagnant = 0
            self._stagnation.stagnation_detected = False

        # Trigger stagnation warning after window_size consecutive stagnant steps
        if self._stagnation.consecutive_stagnant >= self.window_size:
            if not self._stagnation.stagnation_detected:
                self._stagnation.stagnation_detected = True
                self._log_stagnation_warning()

    def _log_stagnation_warning(self) -> None:
        """Log stagnation warning with phase context."""
        phase_str = f"Phase {self._current_phase}" if self._current_phase else "Unknown phase"
        logger.warning(
            f"EMA stagnation detected ({phase_str}): "
            f"Variance change < {self.variance_threshold} for "
            f"{self._stagnation.consecutive_stagnant} consecutive steps. "
            f"Current variance: {self._stagnation.variance_history[-1]:.6f}"
        )

    @property
    def is_stagnant(self) -> bool:
        """Check if currently in stagnation state."""
        return self._stagnation.stagnation_detected

    def reset_stagnation(self) -> None:
        """Reset stagnation detection state."""
        self._stagnation = StagnationState()

    def state_dict(self) -> dict[str, Any]:
        """Export state for checkpointing."""
        return {
            "mean": self._mean.cpu(),
            "variance": self._variance.cpu(),
            "initialized": self._initialized,
            "update_count": self._update_count,
            "dim": self.dim,
            "decay": self.decay,
            "variance_threshold": self.variance_threshold,
            "window_size": self.window_size,
            "stagnation": {
                "variance_history": self._stagnation.variance_history,
                "consecutive_stagnant": self._stagnation.consecutive_stagnant,
                "stagnation_detected": self._stagnation.stagnation_detected,
            },
            "current_phase": self._current_phase,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self._mean = state["mean"].to(self.device)
        self._variance = state["variance"].to(self.device)
        self._initialized = state["initialized"]
        self._update_count = state["update_count"]
        self.dim = state["dim"]
        self.decay = state["decay"]
        self.variance_threshold = state.get("variance_threshold", 1e-6)
        self.window_size = state.get("window_size", 100)

        if "stagnation" in state:
            s = state["stagnation"]
            self._stagnation = StagnationState(
                variance_history=s["variance_history"],
                consecutive_stagnant=s["consecutive_stagnant"],
                stagnation_detected=s["stagnation_detected"],
            )

        self._current_phase = state.get("current_phase")

    def to(self, device: torch.device) -> "EMAState":
        """Move state to device."""
        self.device = device
        self._mean = self._mean.to(device)
        self._variance = self._variance.to(device)
        return self
