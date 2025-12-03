"""
Pytest configuration and fixtures for GPN-1 tests.

Provides:
    - Reproducibility fixtures for deterministic testing
    - Device fixtures for CPU/CUDA testing
    - Model factory fixtures
    - Sample data fixtures
"""

import pytest
import torch
import numpy as np
import random
from pathlib import Path
from typing import Generator

# =============================================================================
# Reproducibility Fixtures
# =============================================================================

@pytest.fixture
def seed() -> int:
    """Default seed for reproducible tests."""
    return 42


@pytest.fixture
def set_seed(seed: int) -> Generator[int, None, None]:
    """
    Set all random seeds for reproducibility.

    Saves and restores RNG state after test completes.
    """
    # Save current state
    torch_state = torch.get_rng_state()
    cuda_states = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    numpy_state = np.random.get_state()
    python_state = random.getstate()

    # Set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Enable deterministic operations
    torch.use_deterministic_algorithms(True, warn_only=True)

    yield seed

    # Restore state
    torch.set_rng_state(torch_state)
    for i, state in enumerate(cuda_states):
        torch.cuda.set_rng_state(state, i)
    np.random.set_state(numpy_state)
    random.setstate(python_state)
    torch.use_deterministic_algorithms(False)


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def device() -> torch.device:
    """Return available device (prefer CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device() -> torch.device:
    """Force CPU device for consistent testing."""
    return torch.device("cpu")


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def batch_size() -> int:
    """Default batch size for tests."""
    return 32


@pytest.fixture
def latent_dim() -> int:
    """Default latent dimension for generator."""
    return 64


@pytest.fixture
def num_classes() -> int:
    """Number of MNIST classes."""
    return 10


@pytest.fixture
def image_shape() -> tuple[int, int, int]:
    """MNIST image shape (C, H, W)."""
    return (1, 28, 28)


@pytest.fixture
def sample_noise(batch_size: int, latent_dim: int, cpu_device: torch.device, set_seed: int) -> torch.Tensor:
    """Generate reproducible sample noise."""
    return torch.randn(batch_size, latent_dim, device=cpu_device)


@pytest.fixture
def sample_labels(batch_size: int, num_classes: int, cpu_device: torch.device, set_seed: int) -> torch.Tensor:
    """Generate reproducible sample labels."""
    return torch.randint(0, num_classes, (batch_size,), device=cpu_device)


@pytest.fixture
def sample_images(batch_size: int, image_shape: tuple[int, int, int], cpu_device: torch.device, set_seed: int) -> torch.Tensor:
    """Generate reproducible sample images (random, not real MNIST)."""
    c, h, w = image_shape
    return torch.rand(batch_size, c, h, w, device=cpu_device)


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Temporary directory for checkpoint tests."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    """Temporary directory for logging tests."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def tmp_experiment_dir(tmp_path: Path) -> Path:
    """Temporary directory for experiment outputs."""
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    return exp_dir


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def default_config() -> dict:
    """Default training configuration for tests."""
    return {
        "seed": 42,
        "batch_size": 32,
        "latent_dim": 64,
        "num_classes": 10,
        "learning_rate": 2e-4,
        "total_steps": 100,  # Short for tests
        "phase1_steps": 30,
        "phase2_steps": 60,
        "log_interval": 10,
        "checkpoint_interval": 50,
        "device": "cpu",
    }


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "cuda: marks tests requiring CUDA"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "contract: marks contract tests"
    )


def pytest_collection_modifyitems(config, items):
    """Skip CUDA tests if CUDA not available."""
    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)
