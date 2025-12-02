# Research: GPN-1 MNIST Proof-of-Concept

**Phase**: 0 (Technology Research)
**Date**: 2025-12-02
**Spec**: [spec.md](spec.md)

## Technology Decisions

### Framework: PyTorch 2.x

**Decision**: Use PyTorch as the deep learning framework.

**Rationale**:
- Dynamic computation graph suits research/experimentation workflow
- Native support for custom loss functions and training loops
- Strong reproducibility features (deterministic algorithms, seed management)
- TensorBoard integration for metrics logging
- Excellent community support for GAN architectures

**Alternatives Considered**:
- JAX/Flax: More functional paradigm, excellent for research, but smaller ecosystem for GAN work
- TensorFlow/Keras: Static graph complications for custom training loops

### Network Architecture: Standard MNIST CNN

**Decision**: Use simple CNN architectures matching standard GAN literature.

**Weaver (Generator)**:
- Input: 100-dim latent vector z
- Architecture: Transposed convolutions (standard DCGAN-style)
- Output: 28x28 grayscale image + 16-dim v_pred vector
- v_pred head: Linear layer with tanh activation (bounds to [-1, 1])

**Witness (Discriminator-analog)**:
- Input: 28x28 grayscale image
- Architecture: Standard CNN classifier
- Output: 10-class logits + 16-dim v_seen vector
- v_seen head: Linear layer with tanh activation

**Judge (External Grounding)**:
- Architecture: Simple CNN (2-3 conv layers + FC)
- Training: Standard cross-entropy on MNIST
- Target: >95% accuracy before freezing
- Role: Frozen evaluator in Phase 1-2, removed in Phase 3

**Rationale**:
- MNIST is simple enough that architecture is not the variable under test
- Standard architectures enable fair comparison with GAN baseline
- Dual-head outputs (classification + attribute vector) are straightforward additions

### Loss Implementation

**Grounding Loss**: Cross-entropy between Judge's classification of Weaver output and intended label.

**Alignment Loss**: MSE(v_pred, v_seen) - measures whether Weaver's claimed attributes match Witness's observations.

**Empowerment Loss**: Goldilocks KL divergence tracking EMA of Witness outputs.
- Diagonal Gaussian assumption for tractability
- Track both mean AND variance (per design-v1.md)
- Penalty for variance shrinkage (prevents mode collapse through this channel)
- EMA updated only on Witness forward pass

**Implementation Note**: All loss components logged individually before aggregation (FR-008).

### Training Loop Design

**Alternating Updates** (per design-v1.md):
1. Witness update: Forward pass with Weaver detached, update Witness weights
2. Weaver update: Fresh forward pass, update Weaver weights

**EMA State**: Updated only during Witness forward pass to prevent double-dipping.

**Phase Transitions**:
- Phase 1 (0-5000 steps): High grounding weight, Judge active
- Phase 2 (5000-10000 steps): Decay grounding weight, increase empowerment weight
- Phase 3 (10000+ steps): Judge completely removed from graph (not just zeroed)

### Reproducibility Strategy

**Seed Management**:
- Global seed sets torch, numpy, random
- torch.use_deterministic_algorithms(True) where possible
- CUDA deterministic ops enabled

**Checkpointing**:
- Full state: model weights, optimizer state, RNG states, step count
- Checkpoint at phase boundaries (5000, 10000) minimum
- Configurable interval for intermediate checkpoints

**Logging**:
- All loss components at every step
- Mode diversity metrics at configurable intervals
- Sample images at configurable intervals

### Baseline GAN Implementation

**Architecture**: Matched capacity to GPN-1 (same Weaver/Witness architectures, minus attribute heads).

**Training**: Standard adversarial training with:
- Binary cross-entropy loss
- Same optimizer, learning rate, batch size
- Same number of steps (12,000)

**Purpose**: Control condition for fair comparison.

## Open Questions

1. **Learning Rate**: Start with 2e-4 (Adam default for GANs), may need tuning.
2. **EMA Decay Rate**: Start with 0.999, may need adjustment based on update frequency.
3. **Phase Transition Weights**: Linear decay vs step function for grounding weight in Phase 2.

## Dependencies

```
torch>=2.0
torchvision
numpy
matplotlib
tensorboard
pytest
```

## References

- DCGAN: Radford et al., 2015 (architecture reference)
- design-v1.md (GPN architecture specification)
- spec.md (functional requirements and success criteria)
