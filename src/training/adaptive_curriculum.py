"""
Adaptive curriculum modification functions for Phase 2 experiments.

These functions implement curriculum changes triggered by adaptive policies:
- Advance: Increase difficulty when mastery achieved
- Scaffold: Add support when student stuck
- Intervene: Modify training when pathology detected
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn


@dataclass
class CurriculumState:
    """Tracks current curriculum configuration."""
    difficulty_level: int = 0
    judge_noise_std: float = 0.0
    judge_signal_multiplier: float = 1.0
    feedback_frequency: float = 1.0  # 1.0 = every step, 0.5 = every other step
    scaffolding_active: bool = False
    interventions_count: int = 0


class NoisyJudgeWrapper(nn.Module):
    """
    Wraps Judge to add configurable noise for difficulty adjustment.

    Implements curriculum advancement by degrading Judge signal quality,
    forcing Weaver/Witness to learn more robust representations.
    """

    def __init__(self, judge: nn.Module, noise_std: float = 0.0):
        super().__init__()
        self.judge = judge
        self.noise_std = noise_std

    def forward(self, images):
        logits = self.judge(images)

        if self.noise_std > 0 and self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        return logits

    def set_noise(self, noise_std: float):
        """Adjust noise level."""
        self.noise_std = noise_std


class AmplifiedJudgeWrapper(nn.Module):
    """
    Wraps Judge to amplify signal for scaffolding.

    Implements scaffolding by strengthening Judge feedback when student stuck,
    providing clearer guidance to resolve uncertainty.
    """

    def __init__(self, judge: nn.Module, signal_multiplier: float = 1.0):
        super().__init__()
        self.judge = judge
        self.signal_multiplier = signal_multiplier

    def forward(self, images):
        logits = self.judge(images)

        if self.signal_multiplier != 1.0:
            # Amplify signal by scaling logits
            logits = logits * self.signal_multiplier

        return logits

    def set_multiplier(self, multiplier: float):
        """Adjust signal amplification."""
        self.signal_multiplier = multiplier


class CurriculumManager:
    """
    Manages curriculum state and applies modifications.

    Coordinates between adaptive policies and training components,
    implementing curriculum changes while maintaining training stability.
    """

    def __init__(
        self,
        initial_state: Optional[CurriculumState] = None,
        advancement_noise_increment: float = 0.05,
        max_noise_std: float = 0.3,
        scaffolding_multiplier: float = 1.5,
        max_signal_multiplier: float = 2.0
    ):
        """
        Args:
            initial_state: Starting curriculum configuration
            advancement_noise_increment: How much noise to add per advancement
            max_noise_std: Maximum noise level
            scaffolding_multiplier: Signal amplification for scaffolding
            max_signal_multiplier: Maximum signal amplification
        """
        self.state = initial_state or CurriculumState()
        self.advancement_noise_increment = advancement_noise_increment
        self.max_noise_std = max_noise_std
        self.scaffolding_multiplier = scaffolding_multiplier
        self.max_signal_multiplier = max_signal_multiplier

        self.advancement_history = []
        self.scaffolding_history = []
        self.intervention_history = []

    def advance_difficulty(self, step: int, reason: str) -> dict:
        """
        Increase task difficulty by adding noise to Judge.

        Args:
            step: Current training step
            reason: Why advancement triggered

        Returns:
            Dict of changes made
        """
        old_noise = self.state.judge_noise_std
        self.state.judge_noise_std = min(
            self.state.judge_noise_std + self.advancement_noise_increment,
            self.max_noise_std
        )
        self.state.difficulty_level += 1

        changes = {
            'action': 'advance',
            'step': step,
            'reason': reason,
            'difficulty_level': self.state.difficulty_level,
            'judge_noise_std': self.state.judge_noise_std,
            'delta_noise': self.state.judge_noise_std - old_noise
        }

        self.advancement_history.append(changes)
        return changes

    def add_scaffolding(self, step: int, reason: str) -> dict:
        """
        Add scaffolding by amplifying Judge signal.

        Args:
            step: Current training step
            reason: Why scaffolding triggered

        Returns:
            Dict of changes made
        """
        old_multiplier = self.state.judge_signal_multiplier
        self.state.judge_signal_multiplier = min(
            self.scaffolding_multiplier,
            self.max_signal_multiplier
        )
        self.state.scaffolding_active = True

        changes = {
            'action': 'scaffold',
            'step': step,
            'reason': reason,
            'signal_multiplier': self.state.judge_signal_multiplier,
            'delta_multiplier': self.state.judge_signal_multiplier - old_multiplier
        }

        self.scaffolding_history.append(changes)
        return changes

    def remove_scaffolding(self, step: int, reason: str = "Uncertainty resolved") -> dict:
        """
        Remove scaffolding when no longer needed.

        Args:
            step: Current training step
            reason: Why scaffolding removed

        Returns:
            Dict of changes made
        """
        old_multiplier = self.state.judge_signal_multiplier
        self.state.judge_signal_multiplier = 1.0
        self.state.scaffolding_active = False

        changes = {
            'action': 'remove_scaffold',
            'step': step,
            'reason': reason,
            'signal_multiplier': self.state.judge_signal_multiplier,
            'delta_multiplier': self.state.judge_signal_multiplier - old_multiplier
        }

        return changes

    def intervene(self, step: int, reason: str, reset_weaver: bool = False) -> dict:
        """
        Intervene when pathology detected.

        Args:
            step: Current training step
            reason: Why intervention triggered
            reset_weaver: Whether to reset Weaver weights

        Returns:
            Dict of changes made
        """
        self.state.interventions_count += 1

        changes = {
            'action': 'intervene',
            'step': step,
            'reason': reason,
            'reset_weaver': reset_weaver,
            'interventions_total': self.state.interventions_count
        }

        self.intervention_history.append(changes)
        return changes

    def get_state_dict(self) -> dict:
        """Export current state for logging/checkpointing."""
        return {
            'state': {
                'difficulty_level': self.state.difficulty_level,
                'judge_noise_std': self.state.judge_noise_std,
                'judge_signal_multiplier': self.state.judge_signal_multiplier,
                'feedback_frequency': self.state.feedback_frequency,
                'scaffolding_active': self.state.scaffolding_active,
                'interventions_count': self.state.interventions_count
            },
            'history': {
                'advancements': len(self.advancement_history),
                'scaffoldings': len(self.scaffolding_history),
                'interventions': len(self.intervention_history)
            }
        }

    def should_check_scaffolding_removal(self, dI_dt: float, threshold: float = -0.0005) -> bool:
        """
        Check if scaffolding should be removed.

        Scaffolding removed when uncertainty is resolving (∂I/∂t sufficiently negative).

        Args:
            dI_dt: Current indeterminacy rate of change
            threshold: Threshold for "resolving" (should be negative)

        Returns:
            True if scaffolding should be removed
        """
        return self.state.scaffolding_active and dI_dt < threshold
