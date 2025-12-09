"""
Comparative Tracker for Epistemic Metrics

Orchestrates parallel execution of all epistemic approaches:
- Simple 2D (Approach A)
- Neutrosophic (Approach C)
- Bayesian (Approach B) - to be added in Week 2

Provides unified interface for computing and logging all metrics.
"""

import time
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from src.metrics.epistemic.base import EpistemicMetric, EpistemicState
from src.metrics.epistemic.simple_2d import Simple2DMetric
from src.metrics.epistemic.bayesian_uncertainty import BayesianUncertaintyMetric
from src.metrics.epistemic.neutrosophic import NeutrosophicMetric


class ComparativeTracker:
    """
    Tracks epistemic metrics across multiple approaches in parallel.

    Usage:
        tracker = ComparativeTracker(device=device)
        states = tracker.compute_all(step, v_pred, v_seen, ...)
        tracker.log_to_tensorboard(writer, step)
    """

    def __init__(
        self,
        device: torch.device | None = None,
        enable_simple_2d: bool = True,
        enable_neutrosophic: bool = True,
        enable_bayesian: bool = False,
        random_sampling: bool = False,
        sample_rate: float = 0.5,
        num_mc_samples: int = 10,
    ):
        """
        Initialize comparative tracker.

        Args:
            device: Device for tensor operations
            enable_simple_2d: Enable Simple 2D approach
            enable_neutrosophic: Enable Neutrosophic approach
            enable_bayesian: Enable Bayesian approach (MC dropout)
            random_sampling: Enable sparse tensor sampling (test for gaming)
            sample_rate: Fraction of metrics to compute when sampling
            num_mc_samples: Number of MC dropout samples for Bayesian approach
        """
        self.device = device or torch.device('cpu')
        self.random_sampling = random_sampling
        self.sample_rate = sample_rate

        # Initialize enabled approaches
        self.approaches: dict[str, EpistemicMetric] = {}

        if enable_simple_2d:
            self.approaches['simple_2d'] = Simple2DMetric(
                random_sampling=random_sampling,
                sample_rate=sample_rate,
                device=device,
            )

        if enable_neutrosophic:
            self.approaches['neutrosophic'] = NeutrosophicMetric(
                random_sampling=random_sampling,
                sample_rate=sample_rate,
                device=device,
            )

        if enable_bayesian:
            self.approaches['bayesian'] = BayesianUncertaintyMetric(
                random_sampling=random_sampling,
                sample_rate=sample_rate,
                device=device,
                num_mc_samples=num_mc_samples,
            )

        # Track computational overhead
        self.computation_times: dict[str, list[float]] = {
            name: [] for name in self.approaches.keys()
        }

    def compute_all(
        self,
        step: int,
        v_pred: torch.Tensor,
        v_seen: torch.Tensor,
        judge_logits: torch.Tensor,
        witness_logits: torch.Tensor,
        labels: torch.Tensor,
        fake_images: torch.Tensor,
        **kwargs,
    ) -> dict[str, EpistemicState]:
        """
        Compute metrics for all enabled approaches in parallel.

        Args:
            step: Current training step
            v_pred: Weaver's value predictions [B]
            v_seen: Witness's value predictions [B]
            judge_logits: Judge classifier outputs [B, num_classes]
            witness_logits: Witness classifier outputs [B, num_classes]
            labels: Ground truth labels [B]
            fake_images: Generated images [B, C, H, W]
            **kwargs: Additional approach-specific arguments

        Returns:
            Dictionary mapping approach names to EpistemicState
        """
        states = {}

        for name, approach in self.approaches.items():
            start_time = time.time()

            state = approach.compute(
                step=step,
                v_pred=v_pred,
                v_seen=v_seen,
                judge_logits=judge_logits,
                witness_logits=witness_logits,
                labels=labels,
                fake_images=fake_images,
                **kwargs,
            )

            elapsed = time.time() - start_time
            self.computation_times[name].append(elapsed)

            states[name] = state

        return states

    def log_to_tensorboard(
        self,
        writer: SummaryWriter,
        step: int,
        states: dict[str, EpistemicState],
    ):
        """
        Log all metrics to TensorBoard with approach-specific prefixes.

        Args:
            writer: TensorBoard SummaryWriter
            step: Current training step
            states: Dictionary of EpistemicState from compute_all()
        """
        for name, state in states.items():
            prefix = f'epistemic/{name}'

            # Log all numeric metrics
            for metric_key, metric_value in state.metrics.items():
                if isinstance(metric_value, (int, float)):
                    writer.add_scalar(
                        f'{prefix}/{metric_key}',
                        metric_value,
                        step,
                    )

            # Log computational overhead
            if self.computation_times[name]:
                avg_time = sum(self.computation_times[name]) / len(self.computation_times[name])
                writer.add_scalar(
                    f'{prefix}/computation_time_ms',
                    avg_time * 1000,
                    step,
                )

    def get_summary_statistics(self) -> dict[str, dict[str, Any]]:
        """
        Get summary statistics for all approaches.

        Returns:
            Dictionary mapping approach names to summary stats
        """
        summaries = {}

        for name, approach in self.approaches.items():
            if not approach.history:
                continue

            # Compute statistics over history
            history_length = len(approach.history)
            recent_states = approach.history[-100:]  # Last 100 steps

            # Average metrics
            avg_metrics = {}
            for key in recent_states[0].metrics.keys():
                if isinstance(recent_states[0].metrics[key], (int, float)):
                    values = [s.metrics[key] for s in recent_states if key in s.metrics]
                    if values:
                        avg_metrics[f'avg_{key}'] = sum(values) / len(values)

            # Computational overhead
            avg_time_ms = 0.0
            if self.computation_times[name]:
                avg_time_ms = (sum(self.computation_times[name]) / len(self.computation_times[name])) * 1000

            summaries[name] = {
                'history_length': history_length,
                'avg_computation_time_ms': avg_time_ms,
                **avg_metrics,
            }

        return summaries

    def save_states(self, path: str):
        """
        Save all approach histories to disk.

        Args:
            path: Directory path to save states
        """
        import json
        from pathlib import Path

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, approach in self.approaches.items():
            # Convert history to JSON-serializable format
            history_data = [
                {
                    'step': state.step,
                    'approach': state.approach,
                    'metrics': state.metrics,
                    'metadata': state.metadata,
                }
                for state in approach.history
            ]

            output_path = save_dir / f'{name}_history.json'
            with open(output_path, 'w') as f:
                json.dump(history_data, f, indent=2)

    def load_states(self, path: str):
        """
        Load approach histories from disk.

        Args:
            path: Directory path containing saved states
        """
        import json
        from pathlib import Path

        load_dir = Path(path)

        for name, approach in self.approaches.items():
            input_path = load_dir / f'{name}_history.json'

            if not input_path.exists():
                continue

            with open(input_path, 'r') as f:
                history_data = json.load(f)

            # Reconstruct EpistemicState objects
            approach.history = [
                EpistemicState(
                    step=item['step'],
                    approach=item['approach'],
                    metrics=item['metrics'],
                    metadata=item['metadata'],
                )
                for item in history_data
            ]

    def reset_all(self):
        """Reset all approach histories."""
        for approach in self.approaches.values():
            approach.reset_history()

        for name in self.computation_times.keys():
            self.computation_times[name] = []

    def get_approach(self, name: str) -> EpistemicMetric | None:
        """
        Get specific approach by name.

        Args:
            name: Approach name ('simple_2d', 'neutrosophic', 'bayesian')

        Returns:
            EpistemicMetric instance or None if not enabled
        """
        return self.approaches.get(name)
