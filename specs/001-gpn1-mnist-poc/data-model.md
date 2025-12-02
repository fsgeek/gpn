# Data Model: GPN-1 MNIST Proof-of-Concept

**Phase**: 1 (Design)
**Date**: 2025-12-02
**Spec**: [spec.md](spec.md)

## Core Entities

### Weaver

The generative agent. Produces images and communicates intended attributes.

```python
class Weaver(nn.Module):
    """
    Generator network with costly signaling.

    Input: latent vector z (100-dim)
    Output:
        - image: 28x28 grayscale
        - v_pred: 16-dim attribute vector (claimed features)
    """
    latent_dim: int = 100
    attribute_dim: int = 16  # configurable per spec clarification

    # Architecture: transposed convolutions -> image
    # Separate head: linear -> tanh -> v_pred
```

**Key Behaviors**:
- Forward pass produces both image and v_pred
- v_pred represents "what I claim this image contains"
- Objective: Witness's growth (not fooling Witness)

### Witness

The observing/classifying agent. Learns from Weaver's teaching.

```python
class Witness(nn.Module):
    """
    Classifier network with attribute estimation.

    Input: 28x28 grayscale image
    Output:
        - logits: 10-class classification
        - v_seen: 16-dim attribute vector (observed features)
    """
    attribute_dim: int = 16  # matches Weaver

    # Architecture: convolutions -> features
    # Classification head: linear -> 10 logits
    # Attribute head: linear -> tanh -> v_seen
```

**Key Behaviors**:
- Forward pass produces both classification and v_seen
- v_seen represents "what I observe in this image"
- Alignment with v_pred indicates pedagogical success

### Judge

External grounding agent. Frozen after pre-training.

```python
class Judge(nn.Module):
    """
    Pre-trained MNIST classifier. Frozen during GPN training.

    Input: 28x28 grayscale image
    Output: 10-class logits

    Training: Standard cross-entropy on MNIST
    Target accuracy: >95% before freezing
    """
    # Simple CNN architecture
    # Included in repo with training script
```

**Key Behaviors**:
- Never updated during GPN training
- Provides reality anchor in Phase 1-2
- Completely removed from graph in Phase 3 (not just weight=0)

### EMAState

Tracks exponential moving average of Witness outputs.

```python
@dataclass
class EMAState:
    """
    Tracks running statistics of Witness outputs for empowerment loss.

    Updated ONLY on Witness forward pass (prevents double-dipping).
    """
    mean: Tensor  # EMA of v_seen means
    variance: Tensor  # EMA of v_seen variances (diagonal Gaussian)
    decay: float = 0.999  # EMA decay rate

    def update(self, v_seen: Tensor) -> None:
        """Update EMA statistics. Call only during Witness updates."""
        ...
```

**Key Behaviors**:
- Diagonal Gaussian assumption for tractability
- Variance tracking prevents mode collapse through shrinkage
- Update timing critical for correct gradient flow

### TrainingConfig

Configuration for a training run.

```python
@dataclass
class TrainingConfig:
    """Complete configuration for reproducible training run."""

    # Architecture
    latent_dim: int = 100
    attribute_dim: int = 16

    # Training
    batch_size: int = 64
    total_steps: int = 12000
    learning_rate: float = 2e-4

    # Phase boundaries
    phase1_end: int = 5000
    phase2_end: int = 10000

    # Loss weights (per-phase)
    grounding_weight_initial: float = 1.0
    grounding_weight_final: float = 0.0
    alignment_weight: float = 1.0
    empowerment_weight_initial: float = 0.1
    empowerment_weight_final: float = 1.0

    # Reproducibility
    seed: int = 42
    checkpoint_interval: int = 1000
    sample_interval: int = 500

    # Paths
    judge_checkpoint: str = "checkpoints/judge.pt"
    output_dir: str = "experiments/"
```

### TrainingRun

A complete training execution with outputs.

```python
@dataclass
class TrainingRun:
    """
    Record of a complete training run.

    Stored in experiments/<run_id>/ for reproducibility.
    """
    run_id: str
    config: TrainingConfig

    # State
    checkpoints: List[str]  # paths to checkpoint files

    # Metrics (logged to TensorBoard)
    grounding_loss: List[float]
    alignment_loss: List[float]
    empowerment_loss: List[float]
    mode_diversity: List[Dict[int, float]]  # digit -> proportion

    # Outputs
    sample_images: List[str]  # paths to generated samples
    final_metrics: Dict[str, float]
```

### Experiment

Collection of related training runs.

```python
@dataclass
class Experiment:
    """
    Collection of training runs for comparison.

    E.g., GPN vs baseline comparison, or hyperparameter sweep.
    """
    experiment_id: str
    description: str
    runs: List[TrainingRun]

    # Analysis
    comparison_metrics: Dict[str, Any]
    conclusions: str
```

## Loss Functions

### GroundingLoss

Reality anchor through external Judge.

```python
def grounding_loss(
    generated_image: Tensor,
    intended_label: Tensor,
    judge: Judge
) -> Tensor:
    """
    Cross-entropy between Judge's classification and intended label.

    Active in Phase 1-2 with decaying weight.
    Removed entirely in Phase 3 (Judge not called).
    """
    logits = judge(generated_image)
    return F.cross_entropy(logits, intended_label)
```

### AlignmentLoss

Measures pedagogical success.

```python
def alignment_loss(
    v_pred: Tensor,  # Weaver's claimed attributes
    v_seen: Tensor   # Witness's observed attributes
) -> Tensor:
    """
    MSE between claimed and observed attributes.

    Low alignment loss = Weaver's signals are interpretable.
    """
    return F.mse_loss(v_pred, v_seen)
```

### EmpowermentLoss

Goldilocks regulation of learning dynamics.

```python
def empowerment_loss(
    v_seen: Tensor,
    ema_state: EMAState
) -> Tensor:
    """
    KL divergence for Goldilocks zone maintenance.

    Penalizes:
    - Too little learning (mean stagnation)
    - Too much chaos (variance explosion)
    - Variance shrinkage (mode collapse signal)

    Uses diagonal Gaussian assumption.
    """
    current_mean = v_seen.mean(dim=0)
    current_var = v_seen.var(dim=0)

    # KL divergence between current and EMA distributions
    kl = gaussian_kl(
        current_mean, current_var,
        ema_state.mean, ema_state.variance
    )

    # Additional penalty for variance shrinkage
    shrinkage_penalty = F.relu(ema_state.variance - current_var).sum()

    return kl + shrinkage_penalty
```

## Phase Transitions

```python
class PhaseManager:
    """Manages training phase transitions and weight schedules."""

    def get_phase(self, step: int) -> int:
        """Return current phase (1, 2, or 3)."""
        if step < self.config.phase1_end:
            return 1
        elif step < self.config.phase2_end:
            return 2
        else:
            return 3

    def get_weights(self, step: int) -> Dict[str, float]:
        """Return loss weights for current step."""
        phase = self.get_phase(step)

        if phase == 1:
            return {
                'grounding': self.config.grounding_weight_initial,
                'alignment': self.config.alignment_weight,
                'empowerment': self.config.empowerment_weight_initial
            }
        elif phase == 2:
            # Linear decay/increase during Phase 2
            progress = (step - self.config.phase1_end) / (self.config.phase2_end - self.config.phase1_end)
            return {
                'grounding': lerp(self.config.grounding_weight_initial, 0.0, progress),
                'alignment': self.config.alignment_weight,
                'empowerment': lerp(self.config.empowerment_weight_initial, self.config.empowerment_weight_final, progress)
            }
        else:  # Phase 3
            return {
                'grounding': 0.0,  # Judge not even called
                'alignment': self.config.alignment_weight,
                'empowerment': self.config.empowerment_weight_final
            }
```

## Metrics

### Mode Diversity

```python
def compute_mode_diversity(
    generated_images: Tensor,
    judge: Judge,
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Compute mode coverage of generated samples.

    Returns:
        coverage: number of classes with >threshold representation
        distribution: per-class proportions
        collapse_warning: True if any class >50%
        collapse_declared: True if <5 classes at >5% each
    """
    predictions = judge(generated_images).argmax(dim=1)
    counts = torch.bincount(predictions, minlength=10)
    proportions = counts.float() / counts.sum()

    coverage = (proportions > threshold).sum().item()
    max_proportion = proportions.max().item()

    return {
        'coverage': coverage,
        'distribution': proportions.tolist(),
        'collapse_warning': max_proportion > 0.5,
        'collapse_declared': coverage < 5
    }
```

### Convergence Speed

```python
def compute_convergence(
    grounding_losses: List[float],
    threshold: float = 0.8
) -> int:
    """
    Steps to reach threshold Judge accuracy on generated samples.

    Used for GPN vs baseline comparison.
    """
    for step, loss in enumerate(grounding_losses):
        if loss < -math.log(threshold):  # CE loss threshold
            return step
    return len(grounding_losses)  # Did not converge
```

## Relationships

```
TrainingConfig ──creates──> TrainingRun
                              │
                              ├── uses ──> Weaver
                              ├── uses ──> Witness
                              ├── uses ──> Judge (Phase 1-2 only)
                              ├── uses ──> EMAState
                              │
                              └── produces ──> Metrics, Samples, Checkpoints

Experiment ──contains──> [TrainingRun, ...]
```
