"""
Model definitions for GPN-1.

Exports:
    - Weaver: Generator with costly signaling (v_pred)
    - Witness: Classifier with attribute estimation (v_seen)
    - Judge: Frozen external classifier for grounding
    - create_weaver, create_witness, create_judge: Factory functions
    - Generator, Discriminator: Baseline GAN models
"""


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies during setup."""
    if name in ("Judge", "create_judge"):
        from src.models.judge import Judge, create_judge
        return Judge if name == "Judge" else create_judge
    elif name in ("Weaver", "create_weaver"):
        from src.models.weaver import Weaver, create_weaver
        return Weaver if name == "Weaver" else create_weaver
    elif name in ("Witness", "create_witness"):
        from src.models.witness import Witness, create_witness
        return Witness if name == "Witness" else create_witness
    elif name in ("Generator", "Discriminator", "create_baseline_gan"):
        from src.models.baseline_gan import Generator, Discriminator, create_baseline_gan
        if name == "Generator":
            return Generator
        elif name == "Discriminator":
            return Discriminator
        else:
            return create_baseline_gan
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Weaver",
    "Witness",
    "Judge",
    "Generator",
    "Discriminator",
    "create_weaver",
    "create_witness",
    "create_judge",
    "create_baseline_gan",
]
