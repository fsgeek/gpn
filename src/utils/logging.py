"""
TensorBoard logging wrapper for GPN-1.

Provides unified metrics logging interface per contracts/metrics.md.

Exports:
    - MetricsLogger: TensorBoard logging with phase-aware organization
"""

from pathlib import Path
from typing import Any, Optional
import time

from torch.utils.tensorboard import SummaryWriter
import torch


class MetricsLogger:
    """
    TensorBoard logging wrapper with phase-aware organization.

    Implements MetricsLoggerInterface per contracts/metrics.md:
    - log_scalar(name, value, step): Log scalar metric
    - log_scalars(name, values_dict, step): Log multiple related scalars
    - log_image(name, image, step): Log image tensor
    - log_histogram(name, values, step): Log value distribution
    - flush(): Ensure all logs are written
    - close(): Clean up resources

    Attributes:
        log_dir: Path to TensorBoard log directory
        experiment_name: Name for this experiment run
    """

    def __init__(
        self,
        log_dir: str | Path,
        experiment_name: Optional[str] = None,
        flush_secs: int = 120,
    ) -> None:
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Base directory for logs
            experiment_name: Optional experiment name (defaults to timestamp)
            flush_secs: How often to flush to disk (seconds)
        """
        self.log_dir = Path(log_dir)

        # Generate experiment name if not provided
        if experiment_name is None:
            experiment_name = f"run_{int(time.time())}"
        self.experiment_name = experiment_name

        # Create full path
        self.full_log_dir = self.log_dir / experiment_name
        self.full_log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer
        self._writer = SummaryWriter(
            log_dir=str(self.full_log_dir),
            flush_secs=flush_secs,
        )

        # Track current phase for organization
        self._current_phase: Optional[int] = None

    def set_phase(self, phase: int) -> None:
        """
        Set current training phase for metric organization.

        Args:
            phase: Training phase (1, 2, or 3)
        """
        self._current_phase = phase

    def log_scalar(
        self,
        name: str,
        value: float | torch.Tensor,
        step: int,
        phase_prefix: bool = True,
    ) -> None:
        """
        Log a scalar metric.

        Args:
            name: Metric name (e.g., "loss/grounding")
            value: Scalar value to log
            step: Training step
            phase_prefix: If True and phase is set, prefix with phase
        """
        if isinstance(value, torch.Tensor):
            value = value.item()

        tag = self._make_tag(name, phase_prefix)
        self._writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        name: str,
        values: dict[str, float | torch.Tensor],
        step: int,
        phase_prefix: bool = True,
    ) -> None:
        """
        Log multiple related scalar metrics.

        Args:
            name: Group name (e.g., "losses")
            values: Dict of metric names to values
            step: Training step
            phase_prefix: If True and phase is set, prefix with phase
        """
        tag = self._make_tag(name, phase_prefix)
        scalar_values = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in values.items()
        }
        self._writer.add_scalars(tag, scalar_values, step)

    def log_image(
        self,
        name: str,
        image: torch.Tensor,
        step: int,
        dataformats: str = "CHW",
    ) -> None:
        """
        Log an image tensor.

        Args:
            name: Image name/tag
            image: Image tensor
            step: Training step
            dataformats: Format string (default CHW for PyTorch)
        """
        self._writer.add_image(name, image, step, dataformats=dataformats)

    def log_images(
        self,
        name: str,
        images: torch.Tensor,
        step: int,
        nrow: int = 8,
    ) -> None:
        """
        Log a grid of images.

        Args:
            name: Image grid name/tag
            images: Batch of images (N, C, H, W)
            step: Training step
            nrow: Number of images per row in grid
        """
        from torchvision.utils import make_grid

        grid = make_grid(images, nrow=nrow, normalize=True)
        self._writer.add_image(name, grid, step)

    def log_histogram(
        self,
        name: str,
        values: torch.Tensor,
        step: int,
        bins: str = "tensorflow",
    ) -> None:
        """
        Log value distribution as histogram.

        Args:
            name: Histogram name
            values: Values to histogram
            step: Training step
            bins: Binning strategy
        """
        self._writer.add_histogram(name, values, step, bins=bins)

    def log_hparams(
        self,
        hparams: dict[str, Any],
        metrics: dict[str, float],
    ) -> None:
        """
        Log hyperparameters and final metrics.

        Args:
            hparams: Dictionary of hyperparameter names to values
            metrics: Dictionary of metric names to final values
        """
        self._writer.add_hparams(hparams, metrics)

    def log_text(self, name: str, text: str, step: int) -> None:
        """
        Log text content.

        Args:
            name: Text tag
            text: Text content
            step: Training step
        """
        self._writer.add_text(name, text, step)

    def _make_tag(self, name: str, phase_prefix: bool) -> str:
        """Create full tag with optional phase prefix."""
        if phase_prefix and self._current_phase is not None:
            return f"phase{self._current_phase}/{name}"
        return name

    def flush(self) -> None:
        """Ensure all logs are written to disk."""
        self._writer.flush()

    def close(self) -> None:
        """Clean up resources."""
        self._writer.close()

    def __enter__(self) -> "MetricsLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensure cleanup."""
        self.close()
