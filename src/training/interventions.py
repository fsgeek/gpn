"""
Intervention mechanisms for pathology recovery.

Provides modular intervention actions that can be triggered by adaptive policies.
Each intervention type addresses a specific pathology pattern detected via
temporal epistemic derivatives.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class InterventionConfig:
    """Configuration for intervention actions."""
    # Empowerment boost (added to current weight)
    empowerment_boost: float = 0.5
    # Grounding boost (added to current weight)
    grounding_boost: float = 1.0
    # Whether to reset decoder layers
    reset_decoder: bool = False
    # Recovery tracking window (steps)
    recovery_window: int = 500


class InterventionAction:
    """
    Base class for intervention actions.

    Applies modifications to the training system when pathology is detected.
    Tracks pre-intervention state for analysis.
    """

    def __init__(self, config: InterventionConfig):
        self.config = config
        self.intervention_step: Optional[int] = None
        self.pre_intervention_state: Dict[str, Any] = {}

    def apply(
        self,
        step: int,
        trainer: Any,
        weaver: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """
        Apply intervention to training system.

        Args:
            step: Current training step
            trainer: GPNTrainerEpistemic instance (has loss weights)
            weaver: Optional Weaver model for weight modifications

        Returns:
            Dict describing changes made
        """
        self.intervention_step = step
        changes = {'step': step, 'action': 'base_intervention'}

        # Store pre-intervention state
        self.pre_intervention_state = {
            'grounding_weight': getattr(trainer, 'grounding_weight', 1.0),
            'alignment_weight': getattr(trainer, 'alignment_weight', 0.5),
            'empowerment_weight': getattr(trainer, 'empowerment_weight', 0.0),
        }

        # Boost empowerment
        if hasattr(trainer, 'empowerment_weight'):
            new_emp = trainer.empowerment_weight + self.config.empowerment_boost
            trainer.empowerment_weight = new_emp
            changes['empowerment_weight'] = new_emp
            logger.info(f"Intervention: empowerment_weight -> {new_emp}")

        # Boost grounding
        if hasattr(trainer, 'grounding_weight'):
            new_ground = trainer.grounding_weight + self.config.grounding_boost
            trainer.grounding_weight = new_ground
            changes['grounding_weight'] = new_ground
            logger.info(f"Intervention: grounding_weight -> {new_ground}")

        # Optional: Reset decoder layers
        if self.config.reset_decoder and weaver is not None:
            self._reset_decoder(weaver)
            changes['decoder_reset'] = True
            logger.info("Intervention: decoder layers reset")

        return changes

    def _reset_decoder(self, weaver: nn.Module):
        """Reset Weaver decoder layers to random initialization."""
        for name, module in weaver.named_modules():
            if 'decoder' in name.lower() or 'blocks' in name.lower():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()


class ModeCollapseIntervention(InterventionAction):
    """
    Specialized intervention for mode collapse recovery.

    Mode collapse signature:
    - High mode_collapse metric (~0.96 vs baseline ~0.30)
    - High F value
    - Low correctness
    - Sustained ∂F/∂t > 0

    Intervention strategy:
    1. Boost empowerment (force diversity in outputs)
    2. Boost grounding (strengthen Judge signal)
    3. Add noise to class embeddings (break collapsed pattern)
    """

    def __init__(
        self,
        empowerment_boost: float = 0.5,
        grounding_boost: float = 1.0,
        add_latent_noise: bool = True,
        latent_noise_std: float = 0.1,
    ):
        config = InterventionConfig(
            empowerment_boost=empowerment_boost,
            grounding_boost=grounding_boost,
            reset_decoder=False,
        )
        super().__init__(config)
        self.add_latent_noise = add_latent_noise
        self.latent_noise_std = latent_noise_std

    def apply(
        self,
        step: int,
        trainer: Any,
        weaver: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Apply mode collapse intervention."""
        changes = super().apply(step, trainer, weaver)
        changes['action'] = 'mode_collapse_intervention'

        # Add noise to class embeddings to break collapsed pattern
        if self.add_latent_noise and weaver is not None:
            if hasattr(weaver, 'class_embed'):
                with torch.no_grad():
                    noise = torch.randn_like(weaver.class_embed.weight) * self.latent_noise_std
                    weaver.class_embed.weight.add_(noise)
                changes['latent_noise_added'] = self.latent_noise_std
                logger.info(f"Intervention: added noise to class embeddings (std={self.latent_noise_std})")

        return changes


class CollusionIntervention(InterventionAction):
    """
    Specialized intervention for collusion recovery.

    Collusion signature:
    - High alignment (Weaver predicts Witness well)
    - Low correctness (but Judge disagrees)
    - High collusion metric

    Intervention strategy:
    1. Reduce alignment weight (break collusive feedback)
    2. Boost grounding weight (strengthen external signal)
    """

    def __init__(
        self,
        alignment_reduction: float = 0.3,
        grounding_boost: float = 1.5,
    ):
        config = InterventionConfig(
            empowerment_boost=0.0,
            grounding_boost=grounding_boost,
        )
        super().__init__(config)
        self.alignment_reduction = alignment_reduction

    def apply(
        self,
        step: int,
        trainer: Any,
        weaver: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Apply collusion intervention."""
        changes = super().apply(step, trainer, weaver)
        changes['action'] = 'collusion_intervention'

        # Reduce alignment weight
        if hasattr(trainer, 'alignment_weight'):
            new_align = max(0.1, trainer.alignment_weight - self.alignment_reduction)
            trainer.alignment_weight = new_align
            changes['alignment_weight'] = new_align
            logger.info(f"Intervention: alignment_weight -> {new_align}")

        return changes


class GamingIntervention(InterventionAction):
    """
    Specialized intervention for gaming/shortcut learning recovery.

    Gaming signature:
    - Suspiciously fast convergence (high T improvement rate)
    - High gaming metric
    - Fragile to perturbation

    Intervention strategy:
    1. Add noise to Judge signal (break shortcut reliance)
    2. Boost empowerment (force exploration)
    """

    def __init__(
        self,
        judge_noise_std: float = 0.2,
        empowerment_boost: float = 0.3,
    ):
        config = InterventionConfig(
            empowerment_boost=empowerment_boost,
            grounding_boost=0.0,
        )
        super().__init__(config)
        self.judge_noise_std = judge_noise_std

    def apply(
        self,
        step: int,
        trainer: Any,
        weaver: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Apply gaming intervention."""
        changes = super().apply(step, trainer, weaver)
        changes['action'] = 'gaming_intervention'

        # Add noise to Judge if wrapper exists
        if hasattr(trainer, 'judge') and hasattr(trainer.judge, 'set_noise'):
            trainer.judge.set_noise(self.judge_noise_std)
            changes['judge_noise_std'] = self.judge_noise_std
            logger.info(f"Intervention: judge_noise_std -> {self.judge_noise_std}")

        return changes


def get_intervention_for_pathology(pathology_type: str) -> InterventionAction:
    """
    Factory function to get appropriate intervention for detected pathology.

    Args:
        pathology_type: One of 'mode_collapse', 'collusion', 'gaming'

    Returns:
        Appropriate InterventionAction instance
    """
    interventions = {
        'mode_collapse': ModeCollapseIntervention,
        'collusion': CollusionIntervention,
        'gaming': GamingIntervention,
    }

    if pathology_type not in interventions:
        raise ValueError(f"Unknown pathology type: {pathology_type}. "
                        f"Available: {list(interventions.keys())}")

    return interventions[pathology_type]()
