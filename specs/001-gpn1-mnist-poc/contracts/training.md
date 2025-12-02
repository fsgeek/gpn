# Contract: Training Interfaces

**Phase**: 1 (Design)
**Date**: 2025-12-02

## Trainer Interface

```python
class GPNTrainerInterface(Protocol):
    """Three-phase GPN training loop."""

    def train_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        """
        Execute one training step.

        Args:
            batch: (images, labels) from MNIST dataloader

        Returns:
            Dictionary with all loss components:
                - 'grounding_loss': float (0.0 in Phase 3)
                - 'alignment_loss': float
                - 'empowerment_loss': float
                - 'total_loss': float
                - 'phase': int (1, 2, or 3)

        Invariants:
            - All losses finite (no NaN/Inf)
            - Witness updated first, then Weaver (alternating)
            - EMA updated only during Witness update
            - Judge not called in Phase 3
        """
        ...

    def save_checkpoint(self, path: str) -> None:
        """
        Save complete state for reproducibility.

        Must include:
            - Weaver state_dict
            - Witness state_dict
            - Weaver optimizer state_dict
            - Witness optimizer state_dict
            - EMA state
            - Current step
            - RNG states (torch, numpy, random)
            - Config
        """
        ...

    def load_checkpoint(self, path: str) -> None:
        """
        Restore complete state.

        After loading:
            - Training continues from saved step
            - Same seed produces identical trajectory
        """
        ...

    @property
    def current_step(self) -> int:
        """Current training step."""
        ...

    @property
    def current_phase(self) -> int:
        """Current training phase (1, 2, or 3)."""
        ...
```

## Training Step Sequence

```python
def train_step_implementation(self, batch):
    """Reference implementation of training step."""

    images, labels = batch

    # Phase check
    phase = self.phase_manager.get_phase(self.current_step)
    weights = self.phase_manager.get_weights(self.current_step)

    # Generate samples
    z = torch.randn(images.size(0), self.weaver.latent_dim)
    generated_images, v_pred = self.weaver(z)

    # === WITNESS UPDATE ===
    # Detach Weaver outputs
    with torch.no_grad():
        gen_detached = generated_images.detach()
        v_pred_detached = v_pred.detach()

    # Witness forward on real + generated
    _, v_seen_real = self.witness(images)
    logits_gen, v_seen_gen = self.witness(gen_detached)

    # Witness losses
    witness_class_loss = F.cross_entropy(logits_gen, labels)
    witness_align_loss = F.mse_loss(v_pred_detached, v_seen_gen)

    # Update Witness
    self.witness_optimizer.zero_grad()
    (witness_class_loss + witness_align_loss).backward()
    self.witness_optimizer.step()

    # Update EMA (only here!)
    self.ema_state.update(v_seen_gen.detach())

    # === WEAVER UPDATE ===
    # Fresh forward pass (important: not reusing detached)
    generated_images, v_pred = self.weaver(z)
    _, v_seen = self.witness(generated_images)

    # Grounding loss (Phase 1-2 only)
    if phase < 3:
        grounding_loss = self.grounding_loss(generated_images, labels, self.judge)
    else:
        grounding_loss = torch.tensor(0.0)

    # Alignment loss
    alignment_loss = F.mse_loss(v_pred, v_seen)

    # Empowerment loss
    empowerment_loss = self.empowerment_loss(v_seen, self.ema_state)

    # Weighted total
    total_loss = (
        weights['grounding'] * grounding_loss +
        weights['alignment'] * alignment_loss +
        weights['empowerment'] * empowerment_loss
    )

    # Update Weaver
    self.weaver_optimizer.zero_grad()
    total_loss.backward()
    self.weaver_optimizer.step()

    self.current_step += 1

    return {
        'grounding_loss': grounding_loss.item(),
        'alignment_loss': alignment_loss.item(),
        'empowerment_loss': empowerment_loss.item(),
        'total_loss': total_loss.item(),
        'phase': phase
    }
```

## EMA State Contract

```python
class EMAStateInterface(Protocol):
    """EMA tracking for empowerment loss."""

    mean: Tensor
    variance: Tensor
    decay: float

    def update(self, v_seen: Tensor) -> None:
        """
        Update running statistics.

        Args:
            v_seen: Witness attribute output, shape (batch, attribute_dim)

        Invariants:
            - Called exactly once per training step
            - Called only during Witness update (not Weaver update)
            - Mean and variance updated with same decay rate
        """
        ...

    def reset(self) -> None:
        """Reset to initial state (for new training run)."""
        ...

    def state_dict(self) -> Dict[str, Tensor]:
        """For checkpointing."""
        ...

    def load_state_dict(self, state: Dict[str, Tensor]) -> None:
        """For checkpoint restoration."""
        ...
```

## Phase Manager Contract

```python
class PhaseManagerInterface(Protocol):
    """Manages phase transitions and weight schedules."""

    def get_phase(self, step: int) -> int:
        """
        Return phase for given step.

        Returns:
            1: Scaffolding (0 <= step < phase1_end)
            2: Relationship (phase1_end <= step < phase2_end)
            3: Drift Test (step >= phase2_end)
        """
        ...

    def get_weights(self, step: int) -> Dict[str, float]:
        """
        Return loss weights for given step.

        Returns:
            {
                'grounding': float,  # 0.0 in Phase 3
                'alignment': float,
                'empowerment': float
            }

        Invariants:
            - grounding weight decays linearly in Phase 2
            - grounding weight is exactly 0.0 in Phase 3
            - empowerment weight increases linearly in Phase 2
        """
        ...
```

## Baseline Trainer Contract

```python
class GANTrainerInterface(Protocol):
    """Standard adversarial training for baseline comparison."""

    def train_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        """
        Execute one GAN training step.

        Returns:
            {
                'generator_loss': float,
                'discriminator_loss': float,
                'total_loss': float
            }

        Invariants:
            - Same batch size, learning rate as GPN
            - Matched architecture capacity
            - Same number of total steps
        """
        ...
```

## Reproducibility Contract

```python
def set_reproducibility(seed: int) -> None:
    """
    Configure all random sources for reproducibility.

    Must set:
        - torch.manual_seed(seed)
        - torch.cuda.manual_seed_all(seed)
        - numpy.random.seed(seed)
        - random.seed(seed)
        - torch.backends.cudnn.deterministic = True
        - torch.backends.cudnn.benchmark = False
        - torch.use_deterministic_algorithms(True) where possible
    """
    ...

def get_rng_state() -> Dict[str, Any]:
    """Capture all RNG states for checkpointing."""
    ...

def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore all RNG states from checkpoint."""
    ...
```
