"""
Epistemic honesty instrumentation for GPN training.

Provides three approaches for measuring epistemic states:
- Simple 2D: Alignment + Correctness (baseline)
- Bayesian: MC Dropout uncertainty (standard approach) - Week 2
- Neutrosophic: {T, I, F} decomposition (hypothesis)

Usage:
    from src.metrics.epistemic import Simple2DMetric, NeutrosophicMetric
    from src.metrics.epistemic import ComparativeTracker

    # Individual approach
    metric = Simple2DMetric(random_sampling=False)
    state = metric.compute(step, v_pred, v_seen, ...)

    # Comparative tracking (all approaches in parallel)
    tracker = ComparativeTracker(device=device)
    all_states = tracker.compute_all(step, v_pred, v_seen, ...)
"""

from src.metrics.epistemic.base import EpistemicMetric, EpistemicState
from src.metrics.epistemic.simple_2d import Simple2DMetric
from src.metrics.epistemic.bayesian_uncertainty import BayesianUncertaintyMetric
from src.metrics.epistemic.neutrosophic import NeutrosophicMetric
from src.metrics.epistemic.comparative_tracker import ComparativeTracker

__all__ = [
    'EpistemicMetric',
    'EpistemicState',
    'Simple2DMetric',
    'BayesianUncertaintyMetric',
    'NeutrosophicMetric',
    'ComparativeTracker',
]
