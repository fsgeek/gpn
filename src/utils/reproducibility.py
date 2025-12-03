"""
Reproducibility utilities for GPN-1.

Provides deterministic training through seed control and RNG state management.
Per contracts/training.md: same seed must produce identical training trajectories.

Exports:
    - set_reproducibility: Initialize all RNG sources with a seed
    - get_rng_state: Capture complete RNG state for checkpointing
    - set_rng_state: Restore RNG state from checkpoint
"""

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class RNGState:
    """
    Complete RNG state for reproducibility.

    Captures all sources of randomness used during training:
    - Python random module
    - NumPy random
    - PyTorch CPU RNG
    - PyTorch CUDA RNG (per device)
    """

    python_state: tuple[Any, ...]
    numpy_state: dict[str, Any]
    torch_cpu_state: torch.Tensor
    torch_cuda_states: list[torch.Tensor]

    def state_dict(self) -> dict[str, Any]:
        """Convert to dictionary for checkpointing."""
        return {
            "python_state": self.python_state,
            "numpy_state": self.numpy_state,
            "torch_cpu_state": self.torch_cpu_state,
            "torch_cuda_states": self.torch_cuda_states,
        }

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "RNGState":
        """Restore from checkpoint dictionary."""
        return cls(
            python_state=state_dict["python_state"],
            numpy_state=state_dict["numpy_state"],
            torch_cpu_state=state_dict["torch_cpu_state"],
            torch_cuda_states=state_dict["torch_cuda_states"],
        )


def set_reproducibility(seed: int, deterministic: bool = True) -> None:
    """
    Initialize all random number generators for reproducibility.

    Args:
        seed: Seed value for all RNG sources
        deterministic: If True, enable deterministic algorithms (may impact performance)

    Note:
        Deterministic mode may cause some operations to be slower.
        Set deterministic=False if performance is critical and exact reproducibility
        is not required.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms
    if deterministic:
        # Set cuBLAS workspace config for deterministic behavior on CUDA >= 10.2
        # See: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        if torch.cuda.is_available():
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        torch.use_deterministic_algorithms(True, warn_only=True)
        # cuDNN determinism
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


def get_rng_state() -> RNGState:
    """
    Capture complete RNG state for checkpointing.

    Returns:
        RNGState containing all RNG states needed to resume training
        with identical randomness.

    Example:
        >>> state = get_rng_state()
        >>> # ... do some random operations ...
        >>> set_rng_state(state)  # Restore to exact same state
    """
    # Capture CUDA states for all devices
    cuda_states = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            cuda_states.append(torch.cuda.get_rng_state(i))

    return RNGState(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu_state=torch.get_rng_state(),
        torch_cuda_states=cuda_states,
    )


def set_rng_state(state: RNGState) -> None:
    """
    Restore complete RNG state from checkpoint.

    Args:
        state: RNGState previously captured with get_rng_state()

    Note:
        This should be called after loading a checkpoint to ensure
        training resumes with the exact same randomness as when
        the checkpoint was saved.
    """
    # Restore Python random
    random.setstate(state.python_state)

    # Restore NumPy - handle both old and new state formats
    np.random.set_state(state.numpy_state)

    # Restore PyTorch CPU
    torch.set_rng_state(state.torch_cpu_state)

    # Restore PyTorch CUDA
    if torch.cuda.is_available() and state.torch_cuda_states:
        for i, cuda_state in enumerate(state.torch_cuda_states):
            if i < torch.cuda.device_count():
                torch.cuda.set_rng_state(cuda_state, i)
