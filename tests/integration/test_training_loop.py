"""
Integration tests for GPN-1 training loop.

T021: Single training step integration test
"""

import pytest
import torch
import torch.optim as optim

from src.models.weaver import create_weaver
from src.models.witness import create_witness
from src.models.judge import create_judge
from src.training.losses import grounding_loss, alignment_loss, CombinedLoss
from src.training.ema import EMAState
from src.training.curriculum import PhaseManager


class TestSingleTrainingStep:
    """Integration test for a single training step (T021)."""

    @pytest.fixture
    def models(self, cpu_device: torch.device):
        """Create all models for testing."""
        weaver = create_weaver(latent_dim=64, v_pred_dim=16, device=cpu_device)
        witness = create_witness(num_classes=10, v_seen_dim=16, device=cpu_device)
        judge = create_judge(freeze=True, device=cpu_device)
        return weaver, witness, judge

    @pytest.fixture
    def optimizers(self, models):
        """Create optimizers."""
        weaver, witness, _ = models
        weaver_opt = optim.Adam(weaver.parameters(), lr=2e-4, betas=(0.5, 0.999))
        witness_opt = optim.Adam(witness.parameters(), lr=2e-4, betas=(0.5, 0.999))
        return weaver_opt, witness_opt

    @pytest.fixture
    def training_components(self, cpu_device: torch.device):
        """Create training components."""
        ema = EMAState(dim=16, device=cpu_device)
        phase_manager = PhaseManager(phase1_steps=100, phase2_steps=200)
        loss_fn = CombinedLoss(grounding_weight=1.0, alignment_weight=0.1)
        return ema, phase_manager, loss_fn

    def test_single_step_executes(
        self,
        models,
        optimizers,
        training_components,
        set_seed: int,
    ):
        """Test that a single training step completes without error."""
        weaver, witness, judge = models
        weaver_opt, witness_opt = optimizers
        ema, phase_manager, loss_fn = training_components

        # Create batch
        batch_size = 32
        z = torch.randn(batch_size, 64)
        labels = torch.randint(0, 10, (batch_size,))

        # Forward pass
        weaver.train()
        witness.train()

        # Generate images
        images, v_pred = weaver(z, labels)

        # Witness classifies
        witness_logits, v_seen = witness(images)

        # Judge provides grounding
        with torch.no_grad():
            judge_logits = judge(images)

        # Compute losses
        total_loss, components = loss_fn(
            witness_logits, labels, judge_logits, v_pred, v_seen
        )

        # Backward pass
        weaver_opt.zero_grad()
        witness_opt.zero_grad()
        total_loss.backward()
        weaver_opt.step()
        witness_opt.step()

        # Update EMA
        ema.update(v_seen.detach())

        # Verify no NaN
        assert torch.isfinite(total_loss), "Loss is not finite"
        assert torch.isfinite(images).all(), "Images contain non-finite values"

    def test_gradients_are_reasonable(
        self,
        models,
        optimizers,
        training_components,
        set_seed: int,
    ):
        """Test that gradients are not exploding or vanishing."""
        weaver, witness, judge = models
        weaver_opt, witness_opt = optimizers
        ema, phase_manager, loss_fn = training_components

        # Create batch
        z = torch.randn(32, 64)
        labels = torch.randint(0, 10, (32,))

        # Forward pass
        images, v_pred = weaver(z, labels)
        witness_logits, v_seen = witness(images)

        with torch.no_grad():
            judge_logits = judge(images)

        # Compute loss
        total_loss, _ = loss_fn(
            witness_logits, labels, judge_logits, v_pred, v_seen
        )

        # Backward
        weaver_opt.zero_grad()
        witness_opt.zero_grad()
        total_loss.backward()

        # Check gradient norms
        weaver_grad_norm = sum(
            p.grad.norm().item() for p in weaver.parameters() if p.grad is not None
        )
        witness_grad_norm = sum(
            p.grad.norm().item() for p in witness.parameters() if p.grad is not None
        )

        # Gradients should be non-zero but not huge
        assert weaver_grad_norm > 1e-6, "Weaver gradients are vanishing"
        assert witness_grad_norm > 1e-6, "Witness gradients are vanishing"
        assert weaver_grad_norm < 1e6, "Weaver gradients are exploding"
        assert witness_grad_norm < 1e6, "Witness gradients are exploding"

    def test_phase_transition(
        self,
        models,
        training_components,
        set_seed: int,
    ):
        """Test that phase transitions work correctly."""
        _, _, _ = models
        ema, phase_manager, loss_fn = training_components

        # Check initial phase
        assert phase_manager.current_phase == 1

        # Advance to phase 2
        changed = phase_manager.step(100)
        assert changed
        assert phase_manager.current_phase == 2

        # Get new weights
        g, a, e = phase_manager.get_weights()
        assert g == 1.0
        assert a == 0.5
        assert e == 0.3

    def test_multiple_steps_without_nan(
        self,
        models,
        optimizers,
        training_components,
        set_seed: int,
    ):
        """Test that multiple training steps don't produce NaN."""
        weaver, witness, judge = models
        weaver_opt, witness_opt = optimizers
        ema, phase_manager, loss_fn = training_components

        for step in range(10):
            # Create batch
            z = torch.randn(32, 64)
            labels = torch.randint(0, 10, (32,))

            # Forward
            images, v_pred = weaver(z, labels)
            witness_logits, v_seen = witness(images)

            with torch.no_grad():
                judge_logits = judge(images)

            # Loss
            total_loss, _ = loss_fn(
                witness_logits, labels, judge_logits, v_pred, v_seen
            )

            # Backward
            weaver_opt.zero_grad()
            witness_opt.zero_grad()
            total_loss.backward()
            weaver_opt.step()
            witness_opt.step()

            # EMA update
            ema.update(v_seen.detach())

            # Check for NaN
            assert torch.isfinite(total_loss), f"NaN loss at step {step}"

    def test_ema_updates_on_witness_forward(
        self,
        models,
        training_components,
        set_seed: int,
    ):
        """Test that EMA is updated only after Witness forward pass."""
        weaver, witness, _ = models
        ema, _, _ = training_components

        initial_count = ema.update_count

        # Weaver forward only
        z = torch.randn(32, 64)
        labels = torch.randint(0, 10, (32,))
        images, _ = weaver(z, labels)

        # No EMA update yet
        assert ema.update_count == initial_count

        # Witness forward
        _, v_seen = witness(images)

        # Now update EMA (simulating training loop behavior)
        ema.update(v_seen.detach())

        assert ema.update_count == initial_count + 1
