"""
Quality metrics for GPN-1.

Exports:
    - QualityMetrics: Judge accuracy and recognizability
    - CollusionDetector: Phase-aware cooperative collapse detection (T032a)
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """Result of quality evaluation."""

    judge_accuracy: float
    witness_accuracy: float
    agreement_rate: float  # How often Judge and Witness agree
    confidence_mean: float  # Mean classification confidence
    confidence_std: float  # Std of classification confidence


class QualityMetrics:
    """
    Quality metrics using Judge and Witness classifiers.

    Measures:
    - Judge accuracy: How well generated images are classified by frozen Judge
    - Witness accuracy: How well Witness classifies its own training data
    - Agreement: How often Judge and Witness agree on classification
    """

    def __init__(
        self,
        judge: nn.Module,
        witness: Optional[nn.Module] = None,
    ) -> None:
        """
        Initialize quality metrics.

        Args:
            judge: Frozen Judge classifier
            witness: Optional Witness classifier
        """
        self.judge = judge
        self.witness = witness

    @torch.no_grad()
    def evaluate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> QualityResult:
        """
        Evaluate quality of generated images.

        Args:
            images: Generated images [B, 1, 28, 28]
            labels: Target labels [B]

        Returns:
            QualityResult with metrics
        """
        # Judge evaluation
        judge_logits = self.judge(images)
        judge_preds = judge_logits.argmax(dim=1)
        judge_probs = torch.softmax(judge_logits, dim=1)
        judge_confidence = judge_probs.max(dim=1).values

        judge_accuracy = (judge_preds == labels).float().mean().item()

        # Witness evaluation (if available)
        if self.witness is not None:
            witness_logits, _ = self.witness(images)
            witness_preds = witness_logits.argmax(dim=1)
            witness_accuracy = (witness_preds == labels).float().mean().item()
            agreement_rate = (judge_preds == witness_preds).float().mean().item()
        else:
            witness_accuracy = 0.0
            agreement_rate = 0.0

        return QualityResult(
            judge_accuracy=judge_accuracy,
            witness_accuracy=witness_accuracy,
            agreement_rate=agreement_rate,
            confidence_mean=judge_confidence.mean().item(),
            confidence_std=judge_confidence.std().item(),
        )


@dataclass
class CollusionState:
    """State for collusion detection across phases."""

    alignment_history: list[float] = field(default_factory=list)
    quality_history: list[float] = field(default_factory=list)
    phase_warnings: dict[int, list[str]] = field(default_factory=dict)


class CollusionDetector:
    """
    Phase-aware collusion detection (T032a).

    Tracks alignment_loss vs quality_improvement ratio by phase:
    - Phase 1: Informational only (log but no warnings)
    - Phase 2: Warn if alignment drops while quality stagnates
    - Phase 3: Flag as diagnostic (cooperative collapse vs internalization)

    This helps distinguish:
    - Healthy learning: alignment drops AND quality improves
    - Cooperative collapse: alignment drops BUT quality stagnates
    - Internalization: quality maintained with minimal scaffolding
    """

    def __init__(
        self,
        alignment_drop_threshold: float = 0.1,
        quality_stagnation_threshold: float = 0.01,
        history_window: int = 50,
    ) -> None:
        """
        Initialize collusion detector.

        Args:
            alignment_drop_threshold: Threshold for significant alignment drop
            quality_stagnation_threshold: Threshold for quality stagnation
            history_window: Number of steps to track
        """
        self.alignment_drop_threshold = alignment_drop_threshold
        self.quality_stagnation_threshold = quality_stagnation_threshold
        self.history_window = history_window

        self._state = CollusionState()
        self._current_phase: int = 1

    def set_phase(self, phase: int) -> None:
        """Set current training phase."""
        self._current_phase = phase

    def update(
        self,
        alignment_loss: float,
        quality_metric: float,
    ) -> Optional[str]:
        """
        Update detector with new metrics.

        Args:
            alignment_loss: Current alignment loss
            quality_metric: Current quality metric (e.g., Judge accuracy)

        Returns:
            Warning message if collusion pattern detected, None otherwise
        """
        # Update history
        self._state.alignment_history.append(alignment_loss)
        self._state.quality_history.append(quality_metric)

        # Trim to window size
        if len(self._state.alignment_history) > self.history_window:
            self._state.alignment_history.pop(0)
            self._state.quality_history.pop(0)

        # Need enough history for analysis
        if len(self._state.alignment_history) < 10:
            return None

        # Analyze patterns
        return self._analyze_pattern()

    def _analyze_pattern(self) -> Optional[str]:
        """Analyze alignment vs quality pattern."""
        # Recent vs earlier comparison
        half = len(self._state.alignment_history) // 2
        early_alignment = sum(self._state.alignment_history[:half]) / half
        recent_alignment = sum(self._state.alignment_history[half:]) / (len(self._state.alignment_history) - half)

        early_quality = sum(self._state.quality_history[:half]) / half
        recent_quality = sum(self._state.quality_history[half:]) / (len(self._state.quality_history) - half)

        # Check for alignment drop
        alignment_drop = early_alignment - recent_alignment
        quality_change = recent_quality - early_quality

        # Generate message based on phase
        message = None

        if self._current_phase == 1:
            # Phase 1: Informational only
            if alignment_drop > self.alignment_drop_threshold:
                message = f"[INFO] Phase 1: Alignment dropping ({alignment_drop:.4f}), quality change: {quality_change:.4f}"
                logger.debug(message)

        elif self._current_phase == 2:
            # Phase 2: Warning mode
            if alignment_drop > self.alignment_drop_threshold and abs(quality_change) < self.quality_stagnation_threshold:
                message = (
                    f"[WARNING] Phase 2 collusion pattern: "
                    f"Alignment dropped {alignment_drop:.4f} but quality stagnant "
                    f"(change: {quality_change:.4f}). "
                    f"Possible cooperative collapse developing."
                )
                logger.warning(message)
                self._record_warning(2, message)

        elif self._current_phase == 3:
            # Phase 3: Diagnostic mode
            if abs(quality_change) < self.quality_stagnation_threshold:
                if recent_quality > 0.8:  # Quality still high
                    message = (
                        f"[DIAGNOSTIC] Phase 3: Quality maintained ({recent_quality:.4f}) "
                        f"without scaffolding. Suggests successful internalization."
                    )
                    logger.info(message)
                else:
                    message = (
                        f"[DIAGNOSTIC] Phase 3: Quality low ({recent_quality:.4f}) "
                        f"and stagnant. Suggests cooperative collapse or insufficient training."
                    )
                    logger.warning(message)
            else:
                if quality_change < 0:
                    message = (
                        f"[DIAGNOSTIC] Phase 3: Quality declining ({quality_change:.4f}). "
                        f"Scaffolding removal causing degradation."
                    )
                    logger.warning(message)

        return message

    def _record_warning(self, phase: int, message: str) -> None:
        """Record warning for phase."""
        if phase not in self._state.phase_warnings:
            self._state.phase_warnings[phase] = []
        self._state.phase_warnings[phase].append(message)

    def get_warnings(self, phase: Optional[int] = None) -> list[str]:
        """Get warnings for a phase (or all phases)."""
        if phase is not None:
            return self._state.phase_warnings.get(phase, [])
        return [w for warnings in self._state.phase_warnings.values() for w in warnings]

    def get_summary(self) -> dict:
        """Get summary of collusion detection state."""
        return {
            "current_phase": self._current_phase,
            "alignment_history_len": len(self._state.alignment_history),
            "quality_history_len": len(self._state.quality_history),
            "recent_alignment": self._state.alignment_history[-1] if self._state.alignment_history else None,
            "recent_quality": self._state.quality_history[-1] if self._state.quality_history else None,
            "warnings_by_phase": {k: len(v) for k, v in self._state.phase_warnings.items()},
        }

    def state_dict(self) -> dict:
        """Export state for checkpointing."""
        return {
            "alignment_history": self._state.alignment_history,
            "quality_history": self._state.quality_history,
            "phase_warnings": self._state.phase_warnings,
            "current_phase": self._current_phase,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self._state = CollusionState(
            alignment_history=state["alignment_history"],
            quality_history=state["quality_history"],
            phase_warnings=state["phase_warnings"],
        )
        self._current_phase = state["current_phase"]
