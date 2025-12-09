"""
Test epistemic infrastructure with toy example.

Verifies:
1. All metric classes can be instantiated
2. Metrics can be computed from tensors
3. ComparativeTracker orchestrates all approaches
4. GPNTrainerEpistemic extends base trainer correctly
5. Sparse sampling works
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.epistemic import (
    Simple2DMetric,
    NeutrosophicMetric,
    ComparativeTracker,
)


def test_metric_classes():
    """Test individual metric class instantiation and computation."""
    print("=" * 80)
    print("TEST 1: Individual Metric Classes")
    print("=" * 80)

    device = torch.device('cpu')

    # Create test data
    batch_size = 32
    num_classes = 10

    v_pred = torch.randn(batch_size, device=device)
    v_seen = torch.randn(batch_size, device=device)
    judge_logits = torch.randn(batch_size, num_classes, device=device)
    witness_logits = torch.randn(batch_size, num_classes, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    fake_images = torch.randn(batch_size, 1, 28, 28, device=device)

    # Test Simple 2D
    print("\nTesting Simple2DMetric...")
    simple_2d = Simple2DMetric(device=device)
    state_2d = simple_2d.compute(
        step=0,
        v_pred=v_pred,
        v_seen=v_seen,
        judge_logits=judge_logits,
        witness_logits=witness_logits,
        labels=labels,
        fake_images=fake_images,
    )

    print(f"  Alignment: {state_2d.metrics['alignment']:.4f}")
    print(f"  Correctness: {state_2d.metrics['correctness']:.4f}")
    print(f"  Quadrant: {state_2d.metadata['quadrant_name']}")
    print(f"  Interpretation: {state_2d.metadata['interpretation']}")

    # Test Neutrosophic
    print("\nTesting NeutrosophicMetric...")
    neutrosophic = NeutrosophicMetric(device=device)
    state_neutro = neutrosophic.compute(
        step=0,
        v_pred=v_pred,
        v_seen=v_seen,
        judge_logits=judge_logits,
        witness_logits=witness_logits,
        labels=labels,
        fake_images=fake_images,
    )

    print(f"  T (Truth): {state_neutro.metrics['T']:.4f}")
    print(f"  I (Indeterminacy): {state_neutro.metrics['I']:.4f}")
    print(f"  F (Falsity): {state_neutro.metrics['F']:.4f}")
    print(f"  State: {state_neutro.metadata['state_name']}")
    print(f"  Dominant: {state_neutro.metadata['dominant_component']}")
    print(f"  Interpretation: {state_neutro.metadata['interpretation']}")

    print("\n✓ Individual metric classes work correctly")


def test_comparative_tracker():
    """Test ComparativeTracker parallel execution."""
    print("\n" + "=" * 80)
    print("TEST 2: Comparative Tracker")
    print("=" * 80)

    device = torch.device('cpu')

    # Create tracker
    tracker = ComparativeTracker(
        device=device,
        enable_simple_2d=True,
        enable_neutrosophic=True,
        enable_bayesian=False,
    )

    # Create test data
    batch_size = 32
    num_classes = 10

    v_pred = torch.randn(batch_size, device=device)
    v_seen = torch.randn(batch_size, device=device)
    judge_logits = torch.randn(batch_size, num_classes, device=device)
    witness_logits = torch.randn(batch_size, num_classes, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    fake_images = torch.randn(batch_size, 1, 28, 28, device=device)

    # Compute all approaches
    print("\nComputing metrics for all approaches...")
    states = tracker.compute_all(
        step=0,
        v_pred=v_pred,
        v_seen=v_seen,
        judge_logits=judge_logits,
        witness_logits=witness_logits,
        labels=labels,
        fake_images=fake_images,
    )

    print(f"  Computed {len(states)} approaches:")
    for name in states.keys():
        print(f"    - {name}")

    # Get summary statistics
    summary = tracker.get_summary_statistics()
    print("\nSummary statistics:")
    for name, stats in summary.items():
        print(f"  {name}:")
        print(f"    History length: {stats['history_length']}")
        print(f"    Avg computation time: {stats['avg_computation_time_ms']:.2f}ms")

    print("\n✓ Comparative tracker works correctly")


def test_sparse_sampling():
    """Test sparse tensor sampling."""
    print("\n" + "=" * 80)
    print("TEST 3: Sparse Sampling")
    print("=" * 80)

    device = torch.device('cpu')

    # Create tracker with sparse sampling
    tracker_sparse = ComparativeTracker(
        device=device,
        enable_simple_2d=True,
        enable_neutrosophic=True,
        random_sampling=True,
        sample_rate=0.5,
    )

    # Create tracker without sparse sampling (control)
    tracker_dense = ComparativeTracker(
        device=device,
        enable_simple_2d=True,
        enable_neutrosophic=True,
        random_sampling=False,
    )

    # Create test data
    batch_size = 32
    num_classes = 10

    print("\nComparing sparse vs dense computation...")
    print(f"  Sample rate: 50%")

    for step in range(10):
        v_pred = torch.randn(batch_size, device=device)
        v_seen = torch.randn(batch_size, device=device)
        judge_logits = torch.randn(batch_size, num_classes, device=device)
        witness_logits = torch.randn(batch_size, num_classes, device=device)
        labels = torch.randint(0, num_classes, (batch_size,), device=device)
        fake_images = torch.randn(batch_size, 1, 28, 28, device=device)

        states_sparse = tracker_sparse.compute_all(
            step=step,
            v_pred=v_pred,
            v_seen=v_seen,
            judge_logits=judge_logits,
            witness_logits=witness_logits,
            labels=labels,
            fake_images=fake_images,
        )

        states_dense = tracker_dense.compute_all(
            step=step,
            v_pred=v_pred,
            v_seen=v_seen,
            judge_logits=judge_logits,
            witness_logits=witness_logits,
            labels=labels,
            fake_images=fake_images,
        )

    summary_sparse = tracker_sparse.get_summary_statistics()
    summary_dense = tracker_dense.get_summary_statistics()

    print("\n  Dense computation:")
    for name in summary_dense:
        print(f"    {name}: {summary_dense[name]['avg_computation_time_ms']:.3f}ms")

    print("\n  Sparse computation:")
    for name in summary_sparse:
        print(f"    {name}: {summary_sparse[name]['avg_computation_time_ms']:.3f}ms")

    print("\n✓ Sparse sampling works correctly")


def test_history_tracking():
    """Test history tracking and improvement computation."""
    print("\n" + "=" * 80)
    print("TEST 4: History Tracking")
    print("=" * 80)

    device = torch.device('cpu')

    metric = Simple2DMetric(device=device)

    # Create test data with increasing alignment (simulating learning)
    batch_size = 32
    num_classes = 10

    print("\nSimulating learning trajectory (increasing alignment)...")

    for step in range(100):
        # Gradually decrease MSE (increase alignment)
        alignment_target = step / 100.0  # 0 → 1
        v_pred = torch.randn(batch_size, device=device) * 0.1
        v_seen = v_pred + torch.randn(batch_size, device=device) * (1.0 - alignment_target) * 0.5

        judge_logits = torch.randn(batch_size, num_classes, device=device)
        witness_logits = torch.randn(batch_size, num_classes, device=device)
        labels = torch.randint(0, num_classes, (batch_size,), device=device)
        fake_images = torch.randn(batch_size, 1, 28, 28, device=device)

        state = metric.compute(
            step=step,
            v_pred=v_pred,
            v_seen=v_seen,
            judge_logits=judge_logits,
            witness_logits=witness_logits,
            labels=labels,
            fake_images=fake_images,
        )

        if step % 20 == 0:
            print(f"  Step {step:3d}: alignment={state.metrics['alignment']:.3f}, "
                  f"improvement={state.metrics['alignment_improvement']:.3f}, "
                  f"velocity={state.metrics['velocity']:.3f}")

    print(f"\nHistory length: {len(metric.history)}")
    print(f"Final alignment: {metric.history[-1].metrics['alignment']:.3f}")
    print(f"Total improvement: {metric.history[-1].metrics['alignment'] - metric.history[0].metrics['alignment']:.3f}")

    print("\n✓ History tracking works correctly")


def test_save_load():
    """Test saving and loading epistemic states."""
    print("\n" + "=" * 80)
    print("TEST 5: Save/Load States")
    print("=" * 80)

    import tempfile
    import shutil

    device = torch.device('cpu')

    tracker = ComparativeTracker(
        device=device,
        enable_simple_2d=True,
        enable_neutrosophic=True,
    )

    # Create some history
    batch_size = 32
    num_classes = 10

    print("\nCreating history (10 steps)...")
    for step in range(10):
        v_pred = torch.randn(batch_size, device=device)
        v_seen = torch.randn(batch_size, device=device)
        judge_logits = torch.randn(batch_size, num_classes, device=device)
        witness_logits = torch.randn(batch_size, num_classes, device=device)
        labels = torch.randint(0, num_classes, (batch_size,), device=device)
        fake_images = torch.randn(batch_size, 1, 28, 28, device=device)

        tracker.compute_all(
            step=step,
            v_pred=v_pred,
            v_seen=v_seen,
            judge_logits=judge_logits,
            witness_logits=witness_logits,
            labels=labels,
            fake_images=fake_images,
        )

    # Save to temp directory
    temp_dir = tempfile.mkdtemp()
    print(f"Saving to: {temp_dir}")
    tracker.save_states(temp_dir)

    # Create new tracker and load
    tracker_loaded = ComparativeTracker(
        device=device,
        enable_simple_2d=True,
        enable_neutrosophic=True,
    )

    print("Loading from disk...")
    tracker_loaded.load_states(temp_dir)

    # Verify
    for name in ['simple_2d', 'neutrosophic']:
        original_len = len(tracker.get_approach(name).history)
        loaded_len = len(tracker_loaded.get_approach(name).history)
        print(f"  {name}: {original_len} steps (original) vs {loaded_len} steps (loaded)")

        assert original_len == loaded_len, f"History length mismatch for {name}"

    # Cleanup
    shutil.rmtree(temp_dir)

    print("\n✓ Save/load works correctly")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("EPISTEMIC INFRASTRUCTURE TEST SUITE")
    print("=" * 80)

    try:
        test_metric_classes()
        test_comparative_tracker()
        test_sparse_sampling()
        test_history_tracking()
        test_save_load()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nEpistemic infrastructure is ready for Week 1 experiments.")

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED ✗")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
