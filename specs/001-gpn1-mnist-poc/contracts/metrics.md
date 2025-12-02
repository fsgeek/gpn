# Contract: Metrics Interfaces

**Phase**: 1 (Design)
**Date**: 2025-12-02

## Mode Diversity Contract

```python
class ModeDiversityInterface(Protocol):
    """Mode collapse detection and diversity measurement."""

    def compute(
        self,
        generated_images: Tensor,
        judge: nn.Module
    ) -> ModeDiversityResult:
        """
        Compute mode diversity of generated samples.

        Args:
            generated_images: Batch of generated images, shape (N, 1, 28, 28)
            judge: Pre-trained classifier for predictions

        Returns:
            ModeDiversityResult with coverage metrics

        Invariants:
            - Uses Judge predictions (not Witness) for consistency
            - Thresholds match spec: 5% per-class, 50% single-class warning
        """
        ...

@dataclass
class ModeDiversityResult:
    """Result of mode diversity computation."""

    coverage: int  # Number of classes with >5% representation
    distribution: List[float]  # Per-class proportions (10 values, sum to 1.0)
    collapse_warning: bool  # True if any class >50%
    collapse_declared: bool  # True if coverage < 5

    def to_dict(self) -> Dict[str, Any]:
        """For logging/serialization."""
        ...
```

**Thresholds** (per spec clarification):
- Mode collapse declared: `coverage < 5` (fewer than 5 classes at >5% each)
- Early warning: `max(distribution) > 0.5` (any single class exceeds 50%)

## Quality Metrics Contract

```python
class QualityMetricsInterface(Protocol):
    """Image quality assessment using Judge."""

    def compute_accuracy(
        self,
        generated_images: Tensor,
        intended_labels: Tensor,
        judge: nn.Module
    ) -> float:
        """
        Compute Judge classification accuracy on generated samples.

        Args:
            generated_images: Batch of generated images
            intended_labels: What Weaver intended each image to be
            judge: Pre-trained classifier

        Returns:
            Accuracy in [0.0, 1.0]

        Invariants:
            - Success threshold: >80% (per spec SC-002)
        """
        ...

    def compute_recognizability(
        self,
        generated_images: Tensor,
        judge: nn.Module
    ) -> float:
        """
        Compute confidence of Judge predictions.

        Returns:
            Mean max-probability across samples

        Notes:
            - Proxy for "human recognizable as digits"
            - High confidence = clear digit structure
        """
        ...
```

## Convergence Metrics Contract

```python
class ConvergenceMetricsInterface(Protocol):
    """Convergence speed comparison between GPN and baseline."""

    def steps_to_threshold(
        self,
        losses: List[float],
        threshold: float
    ) -> Optional[int]:
        """
        Find step where loss first drops below threshold.

        Args:
            losses: Per-step loss values
            threshold: Target loss value

        Returns:
            Step number, or None if threshold never reached
        """
        ...

    def compare_convergence(
        self,
        gpn_losses: List[float],
        baseline_losses: List[float],
        threshold: float = 0.5
    ) -> ConvergenceComparison:
        """
        Compare convergence speed between GPN and baseline.

        Returns:
            ConvergenceComparison with statistical significance
        """
        ...

@dataclass
class ConvergenceComparison:
    """Result of convergence comparison."""

    gpn_steps: Optional[int]  # Steps to threshold for GPN
    baseline_steps: Optional[int]  # Steps to threshold for baseline
    gpn_faster: Optional[bool]  # True if GPN converged first
    difference: Optional[int]  # Step difference (positive = GPN faster)
    p_value: Optional[float]  # Statistical significance (requires multiple runs)
```

## Logging Contract

```python
class MetricsLoggerInterface(Protocol):
    """Unified metrics logging."""

    def log_step(
        self,
        step: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Log per-step metrics.

        Required keys:
            - grounding_loss
            - alignment_loss
            - empowerment_loss
            - total_loss
            - phase

        Invariants:
            - Every step logged (FR-008)
            - All loss components logged BEFORE aggregation
        """
        ...

    def log_diversity(
        self,
        step: int,
        result: ModeDiversityResult
    ) -> None:
        """Log mode diversity metrics."""
        ...

    def log_samples(
        self,
        step: int,
        images: Tensor,
        labels: Optional[Tensor] = None
    ) -> str:
        """
        Save sample images and return path.

        Args:
            step: Current training step
            images: Generated images, shape (N, 1, 28, 28)
            labels: Optional intended labels

        Returns:
            Path to saved image grid
        """
        ...

    def flush(self) -> None:
        """Ensure all metrics written to storage."""
        ...
```

## Statistical Analysis Contract

```python
class StatisticalAnalysisInterface(Protocol):
    """Statistical comparison across runs."""

    def compute_significance(
        self,
        gpn_runs: List[TrainingRun],
        baseline_runs: List[TrainingRun],
        metric: str
    ) -> StatisticalResult:
        """
        Compute statistical significance of difference.

        Args:
            gpn_runs: List of GPN training runs (minimum 3)
            baseline_runs: List of baseline runs (minimum 3)
            metric: Metric to compare (e.g., 'convergence_steps', 'mode_diversity')

        Returns:
            StatisticalResult with p-value and effect size

        Invariants:
            - Requires minimum 3 runs each (per spec SC-004)
            - p < 0.05 for significance claim
        """
        ...

@dataclass
class StatisticalResult:
    """Result of statistical comparison."""

    metric: str
    gpn_mean: float
    gpn_std: float
    baseline_mean: float
    baseline_std: float
    difference: float  # gpn_mean - baseline_mean
    p_value: float
    significant: bool  # p < 0.05
    effect_size: float  # Cohen's d
```

## Success Criteria Validation

```python
def validate_success_criteria(
    run: TrainingRun
) -> Dict[str, bool]:
    """
    Check if training run meets success criteria.

    Returns:
        {
            'SC-001': bool,  # Completed without NaN/explosion
            'SC-002': bool,  # Images recognizable (>80% Judge accuracy)
            'SC-003': bool,  # Mode diversity (>=8 classes at >5%)
            'SC-006': bool   # Reproducible (same seed = same trajectory)
        }

    Note: SC-004, SC-005, SC-007 require multiple runs or documentation.
    """
    ...
```
