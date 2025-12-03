"""
Metrics for GPN-1 evaluation.

Exports:
    - ModeDiversity: Mode collapse detection and coverage metrics
    - QualityMetrics: Judge accuracy and recognizability
    - ConvergenceMetrics: Convergence speed comparison
    - CollusionDetector: Phase-aware cooperative collapse detection
"""


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies during setup."""
    if name in ("ModeDiversity", "ModeDiversityResult"):
        from src.metrics.mode_diversity import ModeDiversity, ModeDiversityResult
        return ModeDiversity if name == "ModeDiversity" else ModeDiversityResult
    elif name in ("QualityMetrics", "CollusionDetector"):
        from src.metrics.quality import QualityMetrics, CollusionDetector
        return QualityMetrics if name == "QualityMetrics" else CollusionDetector
    elif name in ("ConvergenceMetrics", "ConvergenceComparison", "StatisticalResult"):
        from src.metrics.convergence import ConvergenceMetrics, ConvergenceComparison, StatisticalResult
        if name == "ConvergenceMetrics":
            return ConvergenceMetrics
        elif name == "ConvergenceComparison":
            return ConvergenceComparison
        else:
            return StatisticalResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModeDiversity",
    "ModeDiversityResult",
    "QualityMetrics",
    "CollusionDetector",
    "ConvergenceMetrics",
    "ConvergenceComparison",
    "StatisticalResult",
]
