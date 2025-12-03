"""
Contract tests for reproducibility.

T022: Same seed must produce identical training trajectories.
"""

import pytest
import torch
import torch.optim as optim

from src.models.weaver import create_weaver
from src.models.witness import create_witness
from src.models.judge import create_judge
from src.training.losses import CombinedLoss
from src.training.ema import EMAState
from src.utils.reproducibility import set_reproducibility, get_rng_state, set_rng_state


class TestReproducibility:
    """Contract tests for reproducibility (T022)."""

    def run_training_steps(
        self,
        seed: int,
        num_steps: int = 5,
        device: torch.device = torch.device("cpu"),
    ) -> list[tuple[float, torch.Tensor]]:
        """
        Run training steps and return losses and sample outputs.

        Args:
            seed: Random seed
            num_steps: Number of training steps
            device: Device to use

        Returns:
            List of (loss, sample_output) tuples
        """
        set_reproducibility(seed)

        # Create models
        weaver = create_weaver(latent_dim=64, v_pred_dim=16, device=device)
        witness = create_witness(num_classes=10, v_seen_dim=16, device=device)
        judge = create_judge(freeze=True, device=device)

        # Create optimizers
        weaver_opt = optim.Adam(weaver.parameters(), lr=2e-4, betas=(0.5, 0.999))
        witness_opt = optim.Adam(witness.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # Create training components
        ema = EMAState(dim=16, device=device)
        loss_fn = CombinedLoss(grounding_weight=1.0, alignment_weight=0.1)

        results = []

        for step in range(num_steps):
            # Create batch
            z = torch.randn(32, 64, device=device)
            labels = torch.randint(0, 10, (32,), device=device)

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

            # Store results
            results.append((total_loss.item(), images[0].detach().clone()))

        return results

    def test_same_seed_same_trajectory(self, cpu_device: torch.device):
        """Test that same seed produces identical trajectories."""
        seed = 42

        # Run twice with same seed
        results1 = self.run_training_steps(seed, num_steps=5, device=cpu_device)
        results2 = self.run_training_steps(seed, num_steps=5, device=cpu_device)

        # Compare results
        for i, ((loss1, img1), (loss2, img2)) in enumerate(zip(results1, results2)):
            assert abs(loss1 - loss2) < 1e-5, f"Losses differ at step {i}: {loss1} vs {loss2}"
            assert torch.allclose(img1, img2, atol=1e-5), f"Images differ at step {i}"

    def test_different_seeds_different_trajectories(self, cpu_device: torch.device):
        """Test that different seeds produce different trajectories."""
        results1 = self.run_training_steps(42, num_steps=3, device=cpu_device)
        results2 = self.run_training_steps(123, num_steps=3, device=cpu_device)

        # At least one step should be different
        any_different = False
        for (loss1, img1), (loss2, img2) in zip(results1, results2):
            if abs(loss1 - loss2) > 1e-3 or not torch.allclose(img1, img2, atol=1e-3):
                any_different = True
                break

        assert any_different, "Different seeds produced identical trajectories"

    def test_rng_state_save_restore(self, cpu_device: torch.device):
        """Test that RNG state can be saved and restored for exact resumption."""
        set_reproducibility(42)

        # Create model
        weaver = create_weaver(latent_dim=64, device=cpu_device)

        # Run a few steps
        for _ in range(3):
            z = torch.randn(32, 64, device=cpu_device)
            labels = torch.randint(0, 10, (32,), device=cpu_device)
            _, _ = weaver(z, labels)

        # Save RNG state
        rng_state = get_rng_state()

        # Generate some more values
        z1 = torch.randn(32, 64, device=cpu_device)
        labels1 = torch.randint(0, 10, (32,), device=cpu_device)
        images1, v_pred1 = weaver(z1, labels1)

        # Restore RNG state
        set_rng_state(rng_state)

        # Generate again - should be identical
        z2 = torch.randn(32, 64, device=cpu_device)
        labels2 = torch.randint(0, 10, (32,), device=cpu_device)
        images2, v_pred2 = weaver(z2, labels2)

        assert torch.allclose(z1, z2), "Noise not reproduced after RNG restore"
        assert torch.equal(labels1, labels2), "Labels not reproduced after RNG restore"
        assert torch.allclose(images1, images2), "Images not reproduced after RNG restore"

    def test_checkpoint_resume_reproducibility(self, cpu_device: torch.device):
        """Test that resuming from checkpoint reproduces exact trajectory."""
        set_reproducibility(42)

        # Create models
        weaver = create_weaver(latent_dim=64, device=cpu_device)
        witness = create_witness(device=cpu_device)

        weaver_opt = optim.Adam(weaver.parameters(), lr=2e-4)
        witness_opt = optim.Adam(witness.parameters(), lr=2e-4)

        # Run 3 steps
        for _ in range(3):
            z = torch.randn(32, 64, device=cpu_device)
            labels = torch.randint(0, 10, (32,), device=cpu_device)
            images, _ = weaver(z, labels)
            logits, _ = witness(images)
            loss = logits.mean()
            weaver_opt.zero_grad()
            witness_opt.zero_grad()
            loss.backward()
            weaver_opt.step()
            witness_opt.step()

        # Save state (simulating checkpoint)
        rng_state = get_rng_state()
        weaver_state = weaver.state_dict()
        witness_state = witness.state_dict()
        weaver_opt_state = weaver_opt.state_dict()
        witness_opt_state = witness_opt.state_dict()

        # Continue for 2 more steps
        continued_outputs = []
        for _ in range(2):
            z = torch.randn(32, 64, device=cpu_device)
            labels = torch.randint(0, 10, (32,), device=cpu_device)
            images, _ = weaver(z, labels)
            continued_outputs.append(images[0].clone())

        # Now restore and continue again
        set_rng_state(rng_state)
        weaver.load_state_dict(weaver_state)
        witness.load_state_dict(witness_state)
        weaver_opt.load_state_dict(weaver_opt_state)
        witness_opt.load_state_dict(witness_opt_state)

        restored_outputs = []
        for _ in range(2):
            z = torch.randn(32, 64, device=cpu_device)
            labels = torch.randint(0, 10, (32,), device=cpu_device)
            images, _ = weaver(z, labels)
            restored_outputs.append(images[0].clone())

        # Compare
        for i, (cont, rest) in enumerate(zip(continued_outputs, restored_outputs)):
            assert torch.allclose(cont, rest, atol=1e-5), (
                f"Restored trajectory differs at step {i}"
            )
