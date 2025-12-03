"""
Unit tests for GPN-1 models.

T017: Weaver forward pass tests
T018: Witness forward pass tests
"""

import pytest
import torch

from src.models.weaver import Weaver, create_weaver
from src.models.witness import Witness, create_witness
from src.models.judge import Judge, create_judge


class TestWeaver:
    """Unit tests for Weaver (T017)."""

    @pytest.fixture
    def weaver(self) -> Weaver:
        """Create Weaver for testing."""
        return create_weaver(latent_dim=64, v_pred_dim=16)

    @pytest.fixture
    def batch_inputs(self, set_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Create batch of inputs."""
        z = torch.randn(32, 64)
        labels = torch.randint(0, 10, (32,))
        return z, labels

    def test_forward_output_shapes(
        self, weaver: Weaver, batch_inputs: tuple[torch.Tensor, torch.Tensor]
    ):
        """Test that forward pass produces correct output shapes."""
        z, labels = batch_inputs
        images, v_pred = weaver(z, labels)

        assert images.shape == (32, 1, 28, 28), f"Expected (32, 1, 28, 28), got {images.shape}"
        assert v_pred.shape == (32, 16), f"Expected (32, 16), got {v_pred.shape}"

    def test_forward_output_ranges(
        self, weaver: Weaver, batch_inputs: tuple[torch.Tensor, torch.Tensor]
    ):
        """Test that outputs are in expected ranges."""
        z, labels = batch_inputs
        images, v_pred = weaver(z, labels)

        # Images should be in [-1, 1] due to tanh
        assert images.min() >= -1.0, f"Image min {images.min()} < -1"
        assert images.max() <= 1.0, f"Image max {images.max()} > 1"

        # v_pred should be finite
        assert torch.isfinite(v_pred).all(), "v_pred contains non-finite values"

    def test_different_labels_different_outputs(
        self, weaver: Weaver, set_seed: int
    ):
        """Test that different labels produce different outputs."""
        z = torch.randn(1, 64)
        z_repeated = z.repeat(10, 1)
        labels = torch.arange(10)

        images, _ = weaver(z_repeated, labels)

        # Check that images are different for different labels
        for i in range(10):
            for j in range(i + 1, 10):
                diff = (images[i] - images[j]).abs().mean()
                assert diff > 0.01, f"Images for labels {i} and {j} are too similar"

    def test_generate_method(
        self, weaver: Weaver, batch_inputs: tuple[torch.Tensor, torch.Tensor]
    ):
        """Test the generate() convenience method."""
        z, labels = batch_inputs
        images = weaver.generate(z, labels)

        assert images.shape == (32, 1, 28, 28)

    def test_gradients_flow(
        self, weaver: Weaver, batch_inputs: tuple[torch.Tensor, torch.Tensor]
    ):
        """Test that gradients flow through the network."""
        z, labels = batch_inputs
        z.requires_grad = True

        images, v_pred = weaver(z, labels)
        loss = images.mean() + v_pred.mean()
        loss.backward()

        assert z.grad is not None, "Gradients did not flow to input"
        assert z.grad.abs().sum() > 0, "Gradients are zero"


class TestWitness:
    """Unit tests for Witness (T018)."""

    @pytest.fixture
    def witness(self) -> Witness:
        """Create Witness for testing."""
        return create_witness(num_classes=10, v_seen_dim=16)

    @pytest.fixture
    def batch_images(self, set_seed: int) -> torch.Tensor:
        """Create batch of images."""
        return torch.randn(32, 1, 28, 28)

    def test_forward_output_shapes(
        self, witness: Witness, batch_images: torch.Tensor
    ):
        """Test that forward pass produces correct output shapes."""
        logits, v_seen = witness(batch_images)

        assert logits.shape == (32, 10), f"Expected (32, 10), got {logits.shape}"
        assert v_seen.shape == (32, 16), f"Expected (32, 16), got {v_seen.shape}"

    def test_forward_output_ranges(
        self, witness: Witness, batch_images: torch.Tensor
    ):
        """Test that outputs are finite."""
        logits, v_seen = witness(batch_images)

        assert torch.isfinite(logits).all(), "Logits contain non-finite values"
        assert torch.isfinite(v_seen).all(), "v_seen contains non-finite values"

    def test_classify_method(
        self, witness: Witness, batch_images: torch.Tensor
    ):
        """Test the classify() convenience method."""
        logits = witness.classify(batch_images)

        assert logits.shape == (32, 10)

    def test_predict_method(
        self, witness: Witness, batch_images: torch.Tensor
    ):
        """Test the predict() convenience method."""
        predictions = witness.predict(batch_images)

        assert predictions.shape == (32,)
        assert predictions.min() >= 0
        assert predictions.max() <= 9

    def test_accuracy_method(
        self, witness: Witness, batch_images: torch.Tensor, set_seed: int
    ):
        """Test the accuracy() method."""
        labels = torch.randint(0, 10, (32,))
        accuracy = witness.accuracy(batch_images, labels)

        assert 0.0 <= accuracy <= 1.0

    def test_gradients_flow(
        self, witness: Witness, batch_images: torch.Tensor
    ):
        """Test that gradients flow through the network."""
        batch_images.requires_grad = True

        logits, v_seen = witness(batch_images)
        loss = logits.mean() + v_seen.mean()
        loss.backward()

        assert batch_images.grad is not None, "Gradients did not flow to input"


class TestJudge:
    """Unit tests for Judge."""

    @pytest.fixture
    def judge(self) -> Judge:
        """Create Judge for testing."""
        return create_judge(freeze=False)

    @pytest.fixture
    def batch_images(self, set_seed: int) -> torch.Tensor:
        """Create batch of images."""
        return torch.randn(32, 1, 28, 28)

    def test_forward_output_shape(
        self, judge: Judge, batch_images: torch.Tensor
    ):
        """Test that forward pass produces correct output shape."""
        logits = judge(batch_images)

        assert logits.shape == (32, 10)

    def test_freeze_method(self, judge: Judge, batch_images: torch.Tensor):
        """Test the freeze() method."""
        assert not judge.is_frozen

        judge.freeze()

        assert judge.is_frozen
        for param in judge.parameters():
            assert not param.requires_grad

    def test_frozen_no_gradients(self, judge: Judge, batch_images: torch.Tensor):
        """Test that frozen Judge doesn't accumulate gradients."""
        judge.freeze()
        batch_images.requires_grad = True

        logits = judge(batch_images)
        loss = logits.mean()
        loss.backward()

        # Input should still get gradients
        assert batch_images.grad is not None
        # But Judge parameters should not
        for param in judge.parameters():
            assert param.grad is None

    def test_predict_method(self, judge: Judge, batch_images: torch.Tensor):
        """Test the predict() convenience method."""
        predictions = judge.predict(batch_images)

        assert predictions.shape == (32,)
        assert predictions.min() >= 0
        assert predictions.max() <= 9
