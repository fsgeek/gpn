"""
Checkpoint save/restore utilities for GPN-1.

Provides full training state persistence including RNG state for exact reproducibility.
Per contracts/training.md: checkpoints must include complete state for identical resume.

Exports:
    - save_checkpoint: Save complete training state
    - load_checkpoint: Restore training state from file
"""

from pathlib import Path
from typing import Any, Optional
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.reproducibility import RNGState, get_rng_state, set_rng_state

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    step: int,
    phase: int,
    models: dict[str, nn.Module],
    optimizers: dict[str, optim.Optimizer],
    rng_state: Optional[RNGState] = None,
    ema_state: Optional[dict[str, Any]] = None,
    metrics: Optional[dict[str, Any]] = None,
    config: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Save complete training state to checkpoint file.

    Args:
        path: Path to save checkpoint (directory or file)
        step: Current training step
        phase: Current training phase (1, 2, or 3)
        models: Dict of model name to model (e.g., {"weaver": weaver, "witness": witness})
        optimizers: Dict of optimizer name to optimizer
        rng_state: RNG state for reproducibility (captured if not provided)
        ema_state: EMA state dictionary
        metrics: Current metric values for logging
        config: Training configuration

    Returns:
        Path to saved checkpoint file

    Example:
        >>> save_checkpoint(
        ...     "checkpoints/step_5000.pt",
        ...     step=5000,
        ...     phase=1,
        ...     models={"weaver": weaver, "witness": witness},
        ...     optimizers={"weaver": weaver_opt, "witness": witness_opt},
        ... )
    """
    path = Path(path)

    # If path is a directory, create filename
    if path.is_dir() or not path.suffix:
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"checkpoint_step{step}.pt"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Capture RNG state if not provided
    if rng_state is None:
        rng_state = get_rng_state()

    # Build checkpoint dictionary
    checkpoint = {
        "step": step,
        "phase": phase,
        "models": {name: model.state_dict() for name, model in models.items()},
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        "rng_state": rng_state.state_dict(),
    }

    # Optional components
    if ema_state is not None:
        checkpoint["ema_state"] = ema_state

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if config is not None:
        checkpoint["config"] = config

    # Save atomically (write to temp, then rename)
    temp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.rename(path)

    logger.info(f"Saved checkpoint to {path} (step={step}, phase={phase})")
    return path


def load_checkpoint(
    path: str | Path,
    models: dict[str, nn.Module],
    optimizers: Optional[dict[str, optim.Optimizer]] = None,
    device: Optional[torch.device] = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """
    Load training state from checkpoint file.

    Args:
        path: Path to checkpoint file
        models: Dict of model name to model (will be loaded in-place)
        optimizers: Dict of optimizer name to optimizer (optional, will be loaded in-place)
        device: Device to map tensors to (default: current device)
        restore_rng: If True, restore RNG state for exact reproducibility

    Returns:
        Dictionary with checkpoint metadata:
        - step: Training step
        - phase: Training phase
        - ema_state: EMA state if present
        - metrics: Metrics if present
        - config: Config if present

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        KeyError: If required model/optimizer not in checkpoint

    Example:
        >>> meta = load_checkpoint(
        ...     "checkpoints/step_5000.pt",
        ...     models={"weaver": weaver, "witness": witness},
        ...     optimizers={"weaver": weaver_opt, "witness": witness_opt},
        ... )
        >>> print(f"Resumed from step {meta['step']}")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint
    map_location = device if device is not None else None
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Restore model states
    for name, model in models.items():
        if name not in checkpoint["models"]:
            raise KeyError(f"Model '{name}' not found in checkpoint")
        model.load_state_dict(checkpoint["models"][name])
        logger.debug(f"Loaded model state: {name}")

    # Restore optimizer states
    if optimizers is not None:
        for name, opt in optimizers.items():
            if name not in checkpoint["optimizers"]:
                raise KeyError(f"Optimizer '{name}' not found in checkpoint")
            opt.load_state_dict(checkpoint["optimizers"][name])
            logger.debug(f"Loaded optimizer state: {name}")

    # Restore RNG state for reproducibility
    if restore_rng and "rng_state" in checkpoint:
        rng_state = RNGState.from_state_dict(checkpoint["rng_state"])
        set_rng_state(rng_state)
        logger.debug("Restored RNG state")

    # Return metadata
    meta = {
        "step": checkpoint["step"],
        "phase": checkpoint["phase"],
    }

    if "ema_state" in checkpoint:
        meta["ema_state"] = checkpoint["ema_state"]

    if "metrics" in checkpoint:
        meta["metrics"] = checkpoint["metrics"]

    if "config" in checkpoint:
        meta["config"] = checkpoint["config"]

    logger.info(f"Loaded checkpoint from {path} (step={meta['step']}, phase={meta['phase']})")
    return meta


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """
    Find the most recent checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("checkpoint_step*.pt"))

    if not checkpoints:
        return None

    # Sort by step number extracted from filename
    def get_step(p: Path) -> int:
        try:
            # Extract step number from "checkpoint_step{N}.pt"
            return int(p.stem.replace("checkpoint_step", ""))
        except ValueError:
            return -1

    checkpoints.sort(key=get_step)
    return checkpoints[-1]


def cleanup_old_checkpoints(
    checkpoint_dir: str | Path,
    keep_last_n: int = 3,
) -> list[Path]:
    """
    Remove old checkpoints, keeping only the most recent N.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of checkpoints to keep

    Returns:
        List of removed checkpoint paths
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return []

    checkpoints = list(checkpoint_dir.glob("checkpoint_step*.pt"))

    if len(checkpoints) <= keep_last_n:
        return []

    # Sort by step number
    def get_step(p: Path) -> int:
        try:
            return int(p.stem.replace("checkpoint_step", ""))
        except ValueError:
            return -1

    checkpoints.sort(key=get_step)

    # Remove oldest checkpoints
    to_remove = checkpoints[:-keep_last_n]
    for path in to_remove:
        path.unlink()
        logger.debug(f"Removed old checkpoint: {path}")

    return to_remove
