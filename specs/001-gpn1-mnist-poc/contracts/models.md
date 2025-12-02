# Contract: Model Interfaces

**Phase**: 1 (Design)
**Date**: 2025-12-02

## Weaver Interface

```python
class WeaverInterface(Protocol):
    """Generator with costly signaling."""

    latent_dim: int
    attribute_dim: int

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Generate image and attribute prediction.

        Args:
            z: Latent vector, shape (batch, latent_dim)

        Returns:
            image: Generated image, shape (batch, 1, 28, 28), range [-1, 1]
            v_pred: Claimed attributes, shape (batch, attribute_dim), range [-1, 1]

        Invariants:
            - Output image normalized to [-1, 1]
            - v_pred bounded by tanh activation
            - Deterministic given same input and RNG state
        """
        ...
```

## Witness Interface

```python
class WitnessInterface(Protocol):
    """Classifier with attribute estimation."""

    attribute_dim: int
    num_classes: int = 10

    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Classify image and estimate attributes.

        Args:
            image: Input image, shape (batch, 1, 28, 28)

        Returns:
            logits: Classification logits, shape (batch, num_classes)
            v_seen: Observed attributes, shape (batch, attribute_dim), range [-1, 1]

        Invariants:
            - logits are unnormalized (for cross-entropy)
            - v_seen bounded by tanh activation
            - attribute_dim matches Weaver's attribute_dim
        """
        ...
```

## Judge Interface

```python
class JudgeInterface(Protocol):
    """Frozen external classifier."""

    num_classes: int = 10

    def forward(self, image: Tensor) -> Tensor:
        """
        Classify image (no training).

        Args:
            image: Input image, shape (batch, 1, 28, 28)

        Returns:
            logits: Classification logits, shape (batch, num_classes)

        Invariants:
            - All parameters frozen (requires_grad=False)
            - Pre-trained to >95% accuracy on MNIST test set
            - Deterministic (no dropout in eval mode)
        """
        ...

    def load_pretrained(self, path: str) -> None:
        """Load pre-trained weights and freeze."""
        ...
```

## Model Factory

```python
def create_weaver(
    latent_dim: int = 100,
    attribute_dim: int = 16
) -> WeaverInterface:
    """Create Weaver with specified dimensions."""
    ...

def create_witness(
    attribute_dim: int = 16,
    num_classes: int = 10
) -> WitnessInterface:
    """Create Witness matching Weaver's attribute_dim."""
    ...

def create_judge(
    checkpoint_path: str,
    num_classes: int = 10
) -> JudgeInterface:
    """Create Judge from pre-trained checkpoint."""
    ...

def create_baseline_gan() -> Tuple[nn.Module, nn.Module]:
    """
    Create Generator and Discriminator for baseline comparison.

    Returns matched-capacity models without attribute heads.
    """
    ...
```

## Dimension Constraints

| Component | Dimension | Value | Notes |
|-----------|-----------|-------|-------|
| Latent vector z | latent_dim | 100 | Standard GAN convention |
| Attribute vector | attribute_dim | 16 | Configurable per spec |
| Image | H x W | 28 x 28 | MNIST standard |
| Classes | num_classes | 10 | Digits 0-9 |
| Batch | batch_size | 64 | Per spec clarification |

## Value Ranges

| Output | Range | Activation | Notes |
|--------|-------|------------|-------|
| Generated image | [-1, 1] | tanh | Normalized for training |
| v_pred | [-1, 1] | tanh | Bounded for MSE loss |
| v_seen | [-1, 1] | tanh | Matches v_pred range |
| Classification logits | unbounded | none | For cross-entropy |
