"""
Phase management for GPN-1 three-phase training.

Per spec: Phase 1 (Scaffolding), Phase 2 (Relationship), Phase 3 (Drift Test).

Exports:
    - PhaseManager: Phase transition and weight scheduling
"""

from dataclasses import dataclass
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhaseConfig:
    """Configuration for a training phase."""

    name: str
    start_step: int
    end_step: int
    grounding_weight: float
    alignment_weight: float
    empowerment_weight: float


class PhaseManager:
    """
    Manages three-phase GPN training curriculum.

    Phase 1 (Scaffolding, steps 0-5000):
        - Heavy grounding, light alignment, no empowerment
        - Train Witness to classify, Weaver to generate recognizable digits
        - Judge provides external reference

    Phase 2 (Relationship, steps 5000-10000):
        - Balanced grounding and alignment, introduce empowerment
        - Weaver learns to predict Witness's perception
        - Full cooperative loop active

    Phase 3 (Drift Test, steps 10000+):
        - All scaffolding removed (grounding=0, alignment=0, empowerment=0)
        - Observe if cooperative behavior persists
        - Diagnostic phase for internalization vs collapse

    Attributes:
        phase1_steps: End of Phase 1 (exclusive)
        phase2_steps: End of Phase 2 (exclusive)
        current_phase: Current training phase (1, 2, or 3)
    """

    def __init__(
        self,
        phase1_steps: int = 5000,
        phase2_steps: int = 10000,
        phase1_weights: Optional[tuple[float, float, float]] = None,
        phase2_weights: Optional[tuple[float, float, float]] = None,
        phase3_weights: Optional[tuple[float, float, float]] = None,
        on_phase_change: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Initialize phase manager.

        Args:
            phase1_steps: Step where Phase 1 ends
            phase2_steps: Step where Phase 2 ends
            phase1_weights: (grounding, alignment, empowerment) for Phase 1
            phase2_weights: (grounding, alignment, empowerment) for Phase 2
            phase3_weights: (grounding, alignment, empowerment) for Phase 3
            on_phase_change: Callback for phase transitions
        """
        self.phase1_steps = phase1_steps
        self.phase2_steps = phase2_steps
        self._current_phase = 1
        self._current_step = 0
        self._on_phase_change = on_phase_change

        # Default weights per spec
        p1 = phase1_weights or (1.0, 0.1, 0.0)
        p2 = phase2_weights or (1.0, 0.5, 0.3)
        p3 = phase3_weights or (0.0, 0.0, 0.0)

        self._phases = {
            1: PhaseConfig(
                name="Scaffolding",
                start_step=0,
                end_step=phase1_steps,
                grounding_weight=p1[0],
                alignment_weight=p1[1],
                empowerment_weight=p1[2],
            ),
            2: PhaseConfig(
                name="Relationship",
                start_step=phase1_steps,
                end_step=phase2_steps,
                grounding_weight=p2[0],
                alignment_weight=p2[1],
                empowerment_weight=p2[2],
            ),
            3: PhaseConfig(
                name="Drift Test",
                start_step=phase2_steps,
                end_step=float('inf'),
                grounding_weight=p3[0],
                alignment_weight=p3[1],
                empowerment_weight=p3[2],
            ),
        }

    @property
    def current_phase(self) -> int:
        """Current training phase."""
        return self._current_phase

    @property
    def current_step(self) -> int:
        """Current training step."""
        return self._current_step

    @property
    def phase_config(self) -> PhaseConfig:
        """Configuration for current phase."""
        return self._phases[self._current_phase]

    @property
    def phase_name(self) -> str:
        """Name of current phase."""
        return self._phases[self._current_phase].name

    def get_phase(self, step: int) -> int:
        """Get phase for a given step."""
        if step < self.phase1_steps:
            return 1
        elif step < self.phase2_steps:
            return 2
        else:
            return 3

    def get_weights(self, step: Optional[int] = None) -> tuple[float, float, float]:
        """
        Get loss weights for a given step (or current step).

        Args:
            step: Training step (uses current if None)

        Returns:
            Tuple of (grounding_weight, alignment_weight, empowerment_weight)
        """
        if step is None:
            step = self._current_step

        phase = self.get_phase(step)
        config = self._phases[phase]
        return (
            config.grounding_weight,
            config.alignment_weight,
            config.empowerment_weight,
        )

    def step(self, step: int) -> bool:
        """
        Update to a new step.

        Args:
            step: New training step

        Returns:
            True if phase changed, False otherwise
        """
        self._current_step = step
        new_phase = self.get_phase(step)

        if new_phase != self._current_phase:
            old_phase = self._current_phase
            self._current_phase = new_phase
            self._on_phase_transition(old_phase, new_phase, step)
            return True

        return False

    def _on_phase_transition(self, old_phase: int, new_phase: int, step: int) -> None:
        """Handle phase transition."""
        old_config = self._phases[old_phase]
        new_config = self._phases[new_phase]

        logger.info(
            f"Phase transition at step {step}: "
            f"{old_config.name} -> {new_config.name}"
        )
        logger.info(
            f"New weights: grounding={new_config.grounding_weight}, "
            f"alignment={new_config.alignment_weight}, "
            f"empowerment={new_config.empowerment_weight}"
        )

        if self._on_phase_change:
            self._on_phase_change(old_phase, new_phase)

    def progress_in_phase(self) -> float:
        """Get progress within current phase (0.0 to 1.0)."""
        config = self._phases[self._current_phase]
        if config.end_step == float('inf'):
            # Phase 3 has no defined end
            return 0.0

        phase_duration = config.end_step - config.start_step
        steps_in_phase = self._current_step - config.start_step
        return min(1.0, steps_in_phase / phase_duration)

    def steps_until_transition(self) -> Optional[int]:
        """Get steps until next phase transition (None for Phase 3)."""
        config = self._phases[self._current_phase]
        if config.end_step == float('inf'):
            return None
        return config.end_step - self._current_step

    def state_dict(self) -> dict:
        """Export state for checkpointing."""
        return {
            "current_phase": self._current_phase,
            "current_step": self._current_step,
            "phase1_steps": self.phase1_steps,
            "phase2_steps": self.phase2_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self._current_phase = state["current_phase"]
        self._current_step = state["current_step"]
        self.phase1_steps = state["phase1_steps"]
        self.phase2_steps = state["phase2_steps"]
