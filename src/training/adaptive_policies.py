"""
Adaptive policies for curriculum modification based on temporal epistemic signals.

These policies implement mastery learning principles:
- Advance when ∂T/∂t → 0 with high T (mastered current level)
- Scaffold when ∂I/∂t not decreasing (uncertainty not resolving)
- Intervene when ∂F/∂t increasing (pathological patterns emerging)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PolicyState:
    """State tracked by adaptive policy."""
    step: int
    triggered: bool
    last_trigger_step: int
    trigger_count: int
    window_history: List[float]


@dataclass
class PolicyAction:
    """Action recommended by policy."""
    action_type: str  # 'advance', 'scaffold', 'intervene', 'none'
    reason: str
    triggered: bool
    metadata: Dict


class AdaptivePolicy(ABC):
    """Base class for adaptive curriculum policies."""

    def __init__(
        self,
        window_size: int = 50,
        cooldown_steps: int = 100,
        name: str = "base_policy"
    ):
        """
        Args:
            window_size: Number of steps for computing temporal derivatives
            cooldown_steps: Minimum steps between triggers
            name: Policy identifier
        """
        self.window_size = window_size
        self.cooldown_steps = cooldown_steps
        self.name = name
        self.state = PolicyState(
            step=0,
            triggered=False,
            last_trigger_step=-cooldown_steps,
            trigger_count=0,
            window_history=[]
        )

    @abstractmethod
    def evaluate(self, metrics: Dict) -> PolicyAction:
        """
        Evaluate whether to trigger action based on current metrics.

        Args:
            metrics: Current epistemic metrics from tracker

        Returns:
            PolicyAction indicating what to do
        """
        pass

    def can_trigger(self) -> bool:
        """Check if enough time has passed since last trigger."""
        return (self.state.step - self.state.last_trigger_step) >= self.cooldown_steps

    def record_trigger(self, action_type: str, reason: str):
        """Record that policy triggered an action."""
        self.state.triggered = True
        self.state.last_trigger_step = self.state.step
        self.state.trigger_count += 1

    def step_counter(self):
        """Increment internal step counter."""
        self.state.step += 1


class AdvancementPolicy(AdaptivePolicy):
    """
    Triggers advancement when ∂T/∂t → 0 with high T.

    Implements mastery learning: when Truth is high and no longer improving,
    student has mastered current level and is ready for harder material.
    """

    def __init__(
        self,
        mastery_threshold: float = 0.7,
        slope_threshold: float = 0.0001,
        min_steps: int = 200,
        **kwargs
    ):
        """
        Args:
            mastery_threshold: Minimum T value to consider mastery
            slope_threshold: Maximum |∂T/∂t| to consider "plateaued"
            min_steps: Minimum training steps before first advancement
        """
        super().__init__(name="advancement_policy", **kwargs)
        self.mastery_threshold = mastery_threshold
        self.slope_threshold = slope_threshold
        self.min_steps = min_steps

    def evaluate(self, metrics: Dict) -> PolicyAction:
        """Check if mastery achieved and plateau reached."""
        self.step_counter()

        # Extract relevant metrics
        T = metrics.get('neutrosophic', {}).get('avg_T', 0.0)
        dT_dt = metrics.get('neutrosophic', {}).get('avg_dT_dt', 0.0)

        # Check conditions
        conditions = {
            'enough_steps': self.state.step >= self.min_steps,
            'high_mastery': T >= self.mastery_threshold,
            'plateaued': abs(dT_dt) <= self.slope_threshold,
            'can_trigger': self.can_trigger()
        }

        all_met = all(conditions.values())

        if all_met:
            reason = (
                f"Mastery achieved: T={T:.3f} >= {self.mastery_threshold}, "
                f"|∂T/∂t|={abs(dT_dt):.6f} <= {self.slope_threshold}"
            )
            self.record_trigger('advance', reason)

            return PolicyAction(
                action_type='advance',
                reason=reason,
                triggered=True,
                metadata={'T': T, 'dT_dt': dT_dt, 'conditions': conditions}
            )

        return PolicyAction(
            action_type='none',
            reason='Conditions not met',
            triggered=False,
            metadata={'T': T, 'dT_dt': dT_dt, 'conditions': conditions}
        )


class ScaffoldingPolicy(AdaptivePolicy):
    """
    Triggers scaffolding when ∂I/∂t not decreasing.

    Implements support: when Indeterminacy is not resolving (flat or increasing),
    student is stuck and needs additional support/scaffolding.
    """

    def __init__(
        self,
        slope_threshold: float = -0.0001,  # Negative means decreasing
        patience_steps: int = 100,
        min_steps: int = 100,
        **kwargs
    ):
        """
        Args:
            slope_threshold: Minimum ∂I/∂t to consider "resolving" (should be negative)
            patience_steps: How many steps of non-resolution before triggering
            min_steps: Minimum training steps before first scaffolding
        """
        super().__init__(name="scaffolding_policy", **kwargs)
        self.slope_threshold = slope_threshold
        self.patience_steps = patience_steps
        self.min_steps = min_steps
        self.non_resolution_count = 0

    def evaluate(self, metrics: Dict) -> PolicyAction:
        """Check if uncertainty is resolving."""
        self.step_counter()

        # Extract relevant metrics
        I = metrics.get('neutrosophic', {}).get('avg_I', 0.0)
        dI_dt = metrics.get('neutrosophic', {}).get('avg_dI_dt', 0.0)

        # Check if uncertainty is resolving (∂I/∂t should be negative)
        is_resolving = dI_dt < self.slope_threshold

        if not is_resolving:
            self.non_resolution_count += 1
        else:
            self.non_resolution_count = 0  # Reset counter

        # Check conditions
        conditions = {
            'enough_steps': self.state.step >= self.min_steps,
            'not_resolving': self.non_resolution_count >= self.patience_steps,
            'can_trigger': self.can_trigger()
        }

        all_met = all(conditions.values())

        if all_met:
            reason = (
                f"Uncertainty not resolving: ∂I/∂t={dI_dt:.6f} >= {self.slope_threshold} "
                f"for {self.non_resolution_count} steps"
            )
            self.record_trigger('scaffold', reason)
            self.non_resolution_count = 0  # Reset after triggering

            return PolicyAction(
                action_type='scaffold',
                reason=reason,
                triggered=True,
                metadata={'I': I, 'dI_dt': dI_dt, 'conditions': conditions}
            )

        return PolicyAction(
            action_type='none',
            reason='Uncertainty resolving normally',
            triggered=False,
            metadata={'I': I, 'dI_dt': dI_dt, 'conditions': conditions, 'patience': self.non_resolution_count}
        )


class InterventionPolicy(AdaptivePolicy):
    """
    Triggers intervention when ∂F/∂t increasing.

    Implements correction: when Falsity is increasing, pathological patterns
    are emerging and intervention is needed.
    """

    def __init__(
        self,
        slope_threshold: float = 0.0001,  # Positive means increasing
        patience_steps: int = 50,
        min_steps: int = 100,
        **kwargs
    ):
        """
        Args:
            slope_threshold: Minimum ∂F/∂t to consider "pathological" (should be positive)
            patience_steps: How many steps of increasing F before triggering
            min_steps: Minimum training steps before first intervention
        """
        super().__init__(name="intervention_policy", **kwargs)
        self.slope_threshold = slope_threshold
        self.patience_steps = patience_steps
        self.min_steps = min_steps
        self.pathology_count = 0

    def evaluate(self, metrics: Dict) -> PolicyAction:
        """Check if falsity is increasing."""
        self.step_counter()

        # Extract relevant metrics
        F = metrics.get('neutrosophic', {}).get('avg_F', 0.0)
        dF_dt = metrics.get('neutrosophic', {}).get('avg_dF_dt', 0.0)

        # Check if falsity is increasing
        is_pathological = dF_dt > self.slope_threshold

        if is_pathological:
            self.pathology_count += 1
        else:
            self.pathology_count = 0  # Reset counter

        # Check conditions
        conditions = {
            'enough_steps': self.state.step >= self.min_steps,
            'pathological': self.pathology_count >= self.patience_steps,
            'can_trigger': self.can_trigger()
        }

        all_met = all(conditions.values())

        if all_met:
            reason = (
                f"Pathology detected: ∂F/∂t={dF_dt:.6f} > {self.slope_threshold} "
                f"for {self.pathology_count} steps"
            )
            self.record_trigger('intervene', reason)
            self.pathology_count = 0  # Reset after triggering

            return PolicyAction(
                action_type='intervene',
                reason=reason,
                triggered=True,
                metadata={'F': F, 'dF_dt': dF_dt, 'conditions': conditions}
            )

        return PolicyAction(
            action_type='none',
            reason='No pathology detected',
            triggered=False,
            metadata={'F': F, 'dF_dt': dF_dt, 'conditions': conditions, 'patience': self.pathology_count}
        )


class MultiSignalPolicy(AdaptivePolicy):
    """
    Combines multiple policies with priority ordering.

    Priority (highest to lowest):
    1. Intervention (stop pathology)
    2. Scaffolding (help when stuck)
    3. Advancement (progress when ready)
    """

    def __init__(
        self,
        advancement_config: Dict = None,
        scaffolding_config: Dict = None,
        intervention_config: Dict = None,
        **kwargs
    ):
        super().__init__(name="multi_signal_policy", **kwargs)

        # Initialize sub-policies
        self.advancement = AdvancementPolicy(**(advancement_config or {}))
        self.scaffolding = ScaffoldingPolicy(**(scaffolding_config or {}))
        self.intervention = InterventionPolicy(**(intervention_config or {}))

    def evaluate(self, metrics: Dict) -> PolicyAction:
        """Evaluate all policies in priority order."""
        self.step_counter()

        # Check in priority order
        intervention_action = self.intervention.evaluate(metrics)
        if intervention_action.triggered:
            return intervention_action

        scaffolding_action = self.scaffolding.evaluate(metrics)
        if scaffolding_action.triggered:
            return scaffolding_action

        advancement_action = self.advancement.evaluate(metrics)
        if advancement_action.triggered:
            return advancement_action

        # No policy triggered
        return PolicyAction(
            action_type='none',
            reason='All policies stable',
            triggered=False,
            metadata={
                'advancement': advancement_action.metadata,
                'scaffolding': scaffolding_action.metadata,
                'intervention': intervention_action.metadata
            }
        )
