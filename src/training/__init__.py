"""
Training infrastructure for GPN-1.

Exports:
    - GPNTrainer: Three-phase GPN training loop
    - GANTrainer: Baseline adversarial training
    - TrainingConfig: Configuration dataclass
    - PhaseManager: Phase transition and weight scheduling
    - EMAState: Exponential moving average tracking
    - Loss functions: grounding_loss, alignment_loss, empowerment_loss
"""


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies during setup."""
    if name == "TrainingConfig":
        from src.training.config import TrainingConfig
        return TrainingConfig
    elif name == "PhaseManager":
        from src.training.curriculum import PhaseManager
        return PhaseManager
    elif name == "EMAState":
        from src.training.ema import EMAState
        return EMAState
    elif name in ("grounding_loss", "alignment_loss", "empowerment_loss"):
        from src.training import losses
        return getattr(losses, name)
    elif name == "GPNTrainer":
        from src.training.gpn_trainer import GPNTrainer
        return GPNTrainer
    elif name == "GANTrainer":
        from src.training.gan_trainer import GANTrainer
        return GANTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TrainingConfig",
    "PhaseManager",
    "EMAState",
    "grounding_loss",
    "alignment_loss",
    "empowerment_loss",
    "GPNTrainer",
    "GANTrainer",
]
