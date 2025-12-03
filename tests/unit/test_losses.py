"""
Unit tests for GPN-1 loss functions and EMA.

T019: Loss function tests
T020: EMA state update tests
"""

import pytest
import torch

from src.training.losses import grounding_loss, alignment_loss, empowerment_loss, CombinedLoss
from src.training.ema import EMAState


class TestGroundingLoss:
    """Unit tests for grounding_loss (T019)."""

    def test_output_is_scalar(self, set_seed: int):
        """Test that grounding loss produces a scalar."""
        logits = torch.randn(32, 10)
        labels = torch.randint(0, 10, (32,))
        judge_logits = torch.randn(32, 10)

        loss = grounding_loss(logits, labels, judge_logits)

        assert loss.dim() == 0, "Loss should be scalar"

    def test_output_is_positive(self, set_seed: int):
        """Test that grounding loss is non-negative."""
        logits = torch.randn(32, 10)
        labels = torch.randint(0, 10, (32,))
        judge_logits = torch.randn(32, 10)

        loss = grounding_loss(logits, labels, judge_logits)

        assert loss >= 0, "Loss should be non-negative"

    def test_perfect_prediction_low_loss(self, set_seed: int):
        """Test that perfect predictions give low loss."""
        # Create one-hot logits that match labels
        labels = torch.arange(10)
        logits = torch.eye(10) * 10  # Strong predictions
        judge_logits = logits.clone()

        loss = grounding_loss(logits, labels, judge_logits)

        assert loss < 1.0, "Perfect predictions should give low loss"

    def test_gradients_flow(self, set_seed: int):
        """Test that gradients flow through the loss."""
        logits = torch.randn(32, 10, requires_grad=True)
        labels = torch.randint(0, 10, (32,))
        judge_logits = torch.randn(32, 10)

        loss = grounding_loss(logits, labels, judge_logits)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0


class TestAlignmentLoss:
    """Unit tests for alignment_loss (T019)."""

    def test_output_is_scalar(self, set_seed: int):
        """Test that alignment loss produces a scalar."""
        v_pred = torch.randn(32, 16)
        v_seen = torch.randn(32, 16)

        loss = alignment_loss(v_pred, v_seen)

        assert loss.dim() == 0

    def test_output_is_non_negative(self, set_seed: int):
        """Test that alignment loss is non-negative."""
        v_pred = torch.randn(32, 16)
        v_seen = torch.randn(32, 16)

        loss = alignment_loss(v_pred, v_seen)

        assert loss >= 0

    def test_zero_for_identical_inputs(self, set_seed: int):
        """Test that loss is zero when v_pred equals v_seen."""
        v_pred = torch.randn(32, 16)
        v_seen = v_pred.clone()

        loss = alignment_loss(v_pred, v_seen)

        assert loss.abs() < 1e-6, "Loss should be ~0 for identical inputs"

    def test_increases_with_difference(self, set_seed: int):
        """Test that loss increases with larger differences."""
        v_pred = torch.zeros(32, 16)
        v_seen_small = torch.ones(32, 16) * 0.1
        v_seen_large = torch.ones(32, 16) * 1.0

        loss_small = alignment_loss(v_pred, v_seen_small)
        loss_large = alignment_loss(v_pred, v_seen_large)

        assert loss_large > loss_small

    def test_gradients_flow(self, set_seed: int):
        """Test that gradients flow through the loss."""
        v_pred = torch.randn(32, 16, requires_grad=True)
        v_seen = torch.randn(32, 16)

        loss = alignment_loss(v_pred, v_seen)
        loss.backward()

        assert v_pred.grad is not None


class TestEmpowermentLoss:
    """Unit tests for empowerment_loss (T019)."""

    def test_output_is_scalar(self, set_seed: int):
        """Test that empowerment loss produces a scalar."""
        v_pred = torch.randn(32, 16)
        v_seen = torch.randn(32, 16)
        ema_mean = torch.zeros(16)
        ema_var = torch.ones(16)

        loss = empowerment_loss(v_pred, v_seen, ema_mean, ema_var)

        assert loss.dim() == 0

    def test_output_is_non_negative(self, set_seed: int):
        """Test that empowerment loss is non-negative."""
        v_pred = torch.randn(32, 16)
        v_seen = torch.randn(32, 16)
        ema_mean = torch.zeros(16)
        ema_var = torch.ones(16)

        loss = empowerment_loss(v_pred, v_seen, ema_mean, ema_var)

        assert loss >= 0

    def test_variance_collapse_penalty(self, set_seed: int):
        """Test that variance collapse is penalized."""
        v_pred = torch.randn(32, 16)
        # Very low variance (collapsed)
        v_seen = torch.zeros(32, 16) + torch.randn(32, 16) * 0.001
        ema_mean = torch.zeros(16)
        ema_var = torch.ones(16)  # Normal variance

        loss = empowerment_loss(v_pred, v_seen, ema_mean, ema_var)

        # Should have penalty for low variance
        assert loss > 0


class TestCombinedLoss:
    """Unit tests for CombinedLoss."""

    @pytest.fixture
    def combined_loss(self) -> CombinedLoss:
        """Create CombinedLoss for testing."""
        return CombinedLoss(
            grounding_weight=1.0,
            alignment_weight=0.5,
            empowerment_weight=0.0,
        )

    def test_returns_total_and_components(
        self, combined_loss: CombinedLoss, set_seed: int
    ):
        """Test that forward returns total loss and components."""
        witness_logits = torch.randn(32, 10)
        labels = torch.randint(0, 10, (32,))
        judge_logits = torch.randn(32, 10)
        v_pred = torch.randn(32, 16)
        v_seen = torch.randn(32, 16)

        total, components = combined_loss(
            witness_logits, labels, judge_logits, v_pred, v_seen
        )

        assert total.dim() == 0
        assert "grounding" in components
        assert "alignment" in components
        assert "empowerment" in components
        assert "total" in components

    def test_weight_updates(self, combined_loss: CombinedLoss):
        """Test that weights can be updated."""
        combined_loss.update_weights(grounding=0.0, alignment=1.0)

        assert combined_loss.grounding_weight == 0.0
        assert combined_loss.alignment_weight == 1.0


class TestEMAState:
    """Unit tests for EMAState (T020)."""

    @pytest.fixture
    def ema_state(self) -> EMAState:
        """Create EMAState for testing."""
        return EMAState(dim=16, decay=0.99)

    def test_initialization(self, ema_state: EMAState):
        """Test initial state."""
        assert not ema_state.initialized
        assert ema_state.update_count == 0
        assert ema_state.mean.shape == (16,)
        assert ema_state.variance.shape == (16,)

    def test_first_update_initializes(self, ema_state: EMAState, set_seed: int):
        """Test that first update initializes the state."""
        values = torch.randn(32, 16)

        ema_state.update(values)

        assert ema_state.initialized
        assert ema_state.update_count == 1

    def test_mean_tracks_values(self, ema_state: EMAState, set_seed: int):
        """Test that mean tracks input values over time."""
        # Update with values centered around 5
        for _ in range(100):
            values = torch.randn(32, 16) + 5
            ema_state.update(values)

        # Mean should be close to 5
        assert (ema_state.mean - 5).abs().mean() < 0.5

    def test_variance_tracks_values(self, ema_state: EMAState, set_seed: int):
        """Test that variance tracks input variance over time."""
        # Update with standard normal values
        for _ in range(100):
            values = torch.randn(32, 16)
            ema_state.update(values)

        # Variance should be close to 1
        assert (ema_state.variance - 1).abs().mean() < 0.5

    def test_state_dict_round_trip(self, ema_state: EMAState, set_seed: int):
        """Test that state can be saved and restored."""
        # Add some updates
        for _ in range(10):
            values = torch.randn(32, 16)
            ema_state.update(values)

        # Save state
        state = ema_state.state_dict()

        # Create new state and restore
        new_state = EMAState(dim=16)
        new_state.load_state_dict(state)

        assert new_state.initialized == ema_state.initialized
        assert new_state.update_count == ema_state.update_count
        assert torch.allclose(new_state.mean, ema_state.mean)
        assert torch.allclose(new_state.variance, ema_state.variance)

    def test_update_only_on_witness_values(self, ema_state: EMAState, set_seed: int):
        """Test that EMA is designed to update only on Witness values."""
        # This is a design test - EMA should be called only after Witness forward
        initial_count = ema_state.update_count

        # Simulate Witness forward -> update
        v_seen = torch.randn(32, 16)
        ema_state.update(v_seen)

        assert ema_state.update_count == initial_count + 1

    def test_stagnation_detection(self, set_seed: int):
        """Test stagnation detection (T027a)."""
        ema = EMAState(dim=16, variance_threshold=1e-6, window_size=5)

        # Update with constant values (should trigger stagnation)
        constant_values = torch.ones(32, 16)
        for _ in range(10):
            ema.update(constant_values)

        # Should detect stagnation
        assert ema.is_stagnant

    def test_no_stagnation_with_varying_values(self, set_seed: int):
        """Test that varying values don't trigger stagnation."""
        ema = EMAState(dim=16, variance_threshold=1e-6, window_size=5)

        # Update with varying values
        for i in range(10):
            values = torch.randn(32, 16) * (i + 1)
            ema.update(values)

        # Should not detect stagnation
        assert not ema.is_stagnant
