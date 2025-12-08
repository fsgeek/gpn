# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPN (Generative Pedagogical Networks) is a research project investigating compositional generalization through pedagogical vs. adversarial training objectives. The core finding: high-fidelity adversarial primitives hit a "glass ceiling" at 81% compositional transfer, while low-fidelity pedagogical primitives achieve 100% transfer.

**Key principle**: This is collaborative research where epistemic honesty matters more than performance optimization. When findings don't replicate or contradict expectations, document honestly and investigate rigorously.

## Development Commands

### Training Models

```bash
# Train single-digit GPN (baseline three-phase pedagogical)
python -m src.cli.train --config configs/gpn1_default.yaml

# Train AC-GAN (adversarial baseline for comparison)
python -m src.cli.train --config configs/acgan_default.yaml

# Resume from checkpoint
python -m src.cli.train --config configs/gpn1_default.yaml --resume checkpoints/checkpoint_step5000.pt
```

### Running Experiments

```bash
# Checkpoint 1 analysis scripts (representation geometry)
python scripts/test_cka_end_to_end.py              # Single CKA analysis
python scripts/test_cka_layer_by_layer.py          # Layer divergence detection
python scripts/test_block0_characterization.py     # Deep dive into block0
python scripts/test_reassembly_rigorous.py         # Statistical validation

# Generate paper figures
python scripts/generate_fig4_fixed.py              # Fidelity comparison figure
python scripts/generate_paper_figures.py           # All paper figures

# Model verification
python scripts/test_direct_checkpoint_load.py      # Verify checkpoint compatibility
python scripts/test_checkpoint_structure.py        # Inspect checkpoint format
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_specific.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Environment Setup

```bash
# Install dependencies (uses uv for fast installation)
uv pip install -e .

# Install with dev dependencies
uv pip install -e ".[dev]"
```

## Architecture: The Big Picture

### Three-Phase Pedagogical Training

The core innovation is a curriculum that trains generators through **cooperative signaling** rather than adversarial competition:

**Phase 1 (Scaffolding, steps 0-5000)**: Heavy external guidance
- Grounding loss: 1.0 (Witness learns from Judge)
- Alignment loss: 0.1 (Weaver weakly predicts Witness)
- Empowerment loss: 0.0 (no diversity pressure yet)
- Purpose: Bootstrap basic competence

**Phase 2 (Relationship, steps 5000-10000)**: Cooperative learning
- Grounding: 1.0, Alignment: 0.5, Empowerment: 0.3
- EMA tracking of v_seen enables "Goldilocks" diversity loss
- Weaver learns to predict Witness's perception (v_pred → v_seen)
- Full cooperative feedback loop active

**Phase 3 (Drift Test, steps 10000+)**: Remove scaffolding
- All loss weights → 0.0
- Tests whether cooperation persists without external pressure
- Diagnostic: distinguishes internalization from cooperative collapse

### The Pedagogical Triad

Three models work together (all in `src/models/`):

1. **Weaver** (generator): Takes latent noise + labels → generates images + predicts value
   - Key output: `v_pred` (prediction of perceived quality)
   - Training: Learns to match `v_seen` from Witness (alignment loss)

2. **Witness** (classifier): Takes images → classifies + estimates value
   - Key output: `v_seen` (perceived quality/recognizability)
   - Training: Learns from Judge's classification signal (grounding loss)

3. **Judge** (frozen reference): Takes images → classifies
   - Always frozen during training
   - Provides external ground truth
   - Critical: without Judge, system has no anchor to reality

**Cooperative dynamic**: Weaver generates → Witness evaluates → Judge validates. Weaver learns what Witness perceives; Witness learns what Judge knows. The v_pred/v_seen alignment creates cooperative signaling.

### Single-Digit vs. Compositional Models

The architecture cleanly separates primitive learning from compositional learning:

**Single-Digit Path** (28×28 images):
- Models: `Weaver`, `Witness`, `Judge`
- Data: Standard MNIST (digits 0-9)
- Trainer: `GPNTrainer` with three-phase curriculum
- Checkpoint: `checkpoint_final.pt`

**Relational Path** (28×84 images: [X][>][Y]):
- Models: `RelationalWeaver` (contains frozen single-digit Weaver)
- Data: `RelationalMNIST` (generates X > Y pairs)
- Trainer: `RelationalTrainer` (pure pedagogical, no adversarial)
- Tests: Coverage boundary (0% without relational context) and glass ceiling (81% vs 100%)

**Transfer protocol**:
1. Train single-digit Weaver → freeze it
2. Load into RelationalWeaver, train only routing layer
3. Test on held-out relations (novel combinations)

This separation is methodologically critical: compositional failures can't be attributed to primitive incompetence, since primitives are frozen.

### Configuration-Driven Training

All training behavior is specified in YAML configs (`configs/`):

```yaml
training:
  phase1_steps: 5000    # When to transition to Phase 2
  phase2_steps: 10000   # When to transition to Phase 3

losses:
  phase1: {grounding: 1.0, alignment: 0.1, empowerment: 0.0}
  phase2: {grounding: 1.0, alignment: 0.5, empowerment: 0.3}
  phase3: {grounding: 0.0, alignment: 0.0, empowerment: 0.0}
```

Phase transitions are automatic and deterministic. The `PhaseManager` calls registered callbacks on transitions, enabling loss weight updates and EMA phase awareness.

### EMA-Only Update Strategy

EMA (Exponential Moving Average) statistics update ONLY on Witness forward passes, not Weaver-only passes:

```python
# In GPNTrainer.train_step():
fake_images, v_pred = self.weaver(z, labels)       # Weaver forward
witness_logits, v_seen = self.witness(fake_images)  # Triggers EMA update
judge_logits = self.judge(fake_images)

# After backward pass:
self.ema_state.update(v_seen.detach())  # Only here, not during Weaver-only
```

This creates asymmetry: EMA reflects Witness's distribution, not generation distribution. The empowerment loss references "normal" v_seen conditions, creating diversity pressure relative to what Witness typically perceives.

## Analysis Pipeline (Checkpoint 1)

The `src/analysis/` modules test representation geometry:

**Working hypothesis (as of Dec 2025)**: Standard metrics (dimensionality, separability, disentanglement) don't explain compositional capacity. We've ruled out these explanations but haven't conclusively identified what DOES enable composition.

**Validated findings**:
- CKA divergence at block0: 96.5% (fc layer) → 46.3% (block0)
- GAN has "better" standard metrics yet composes worse (81% vs 100%)
- Class separation doesn't predict composition

**Key analysis modules**:
- `calibration.py`: Expected Calibration Error, compositional uncertainty probe
- `similarity.py`: CKA, SVCCA, Procrustes distance
- `geometry.py`: PCA dimensionality, linear separability, feature independence
- `statistics.py`: Bootstrap CIs, effect sizes, multiple testing correction
- `probes.py`: Linear probes for digit identity, spatial position, stroke structure

**Running the full pipeline**:
```bash
python scripts/test_cka_layer_by_layer.py          # Find divergence point
python scripts/test_block0_characterization.py     # Characterize differences
python scripts/test_reassembly_rigorous.py         # Statistical validation
```

All results save to `experiments/` with visualizations.

## Checkpoint Structure

Checkpoints save complete training state for exact reproducibility:

```python
{
    'step': int,
    'phase': int,
    'models': {'weaver': state_dict, 'witness': state_dict, ...},
    'optimizers': {...},
    'ema_state': {...},
    'rng_state': {...},
    'metrics': {...},
    'config': {...}
}
```

**Loading checkpoints**:
```python
from models.weaver import create_weaver

ckpt = torch.load('checkpoints/checkpoint_final.pt', weights_only=False)
weaver = create_weaver(latent_dim=64, num_classes=10)
weaver.load_state_dict(ckpt['models']['weaver'])
```

**Note**: GPN checkpoints have 'weaver'/'witness' keys; AC-GAN checkpoints have 'generator'/'discriminator' keys.

## Experimental Workflow

### Running a New Experiment

1. **Define hypothesis** in a test script (see `scripts/test_*.py` for patterns)
2. **Extract features** using hooks on specific layers
3. **Run analysis** using modules from `src/analysis/`
4. **Generate visualizations** saved to `experiments/`
5. **Document findings** honestly (see `experiments/CHECKPOINT1_HONEST_FINDINGS.md`)

### Minimal Proof of Concept Pattern

Before building complex analyses, prove the foundation works:

```python
# 1. Load models
weaver = create_weaver(...)
weaver.load_state_dict(ckpt['models']['weaver'])

# 2. Extract features with hooks
features = []
hook = weaver.blocks[0].register_forward_hook(lambda m, i, o: features.append(o))

# 3. Generate samples
with torch.no_grad():
    images, _ = weaver(z, labels)

# 4. Remove hooks
hook.remove()

# 5. Verify shapes and basic stats
print(features[0].shape, features[0].mean(), features[0].std())
```

This pattern (from `test_direct_checkpoint_load.py`) prevents building untested scaffolding.

## Key Implementation Patterns

### Factory Pattern for Models

Every model has a `create_*` factory:

```python
weaver = create_weaver(latent_dim=64, v_pred_dim=16, device=device)
witness = create_witness(num_classes=10, v_seen_dim=16, device=device)
```

Centralizes initialization, enables CLI integration, validates parameters.

### Interface Consistency

All generators implement:
```python
(images, value) = model.forward(z, labels)
```

All classifiers implement:
```python
(logits, value) = witness.forward(images)  # Witness
logits = judge.forward(images)              # Judge (no value)
```

This enables trainer code to be model-agnostic.

### Lazy Imports

`models/__init__.py` uses `__getattr__` for lazy imports to avoid circular dependencies during package setup.

## Fire Circle Collaboration

This project is developed collaboratively with a "Fire Circle" of AI models:
- **Claude Code** (implementation, analysis, epistemic honesty)
- **El jefe** (research direction, experimental design)
- **Perplexity** (scientific rigor, literature connections)
- **ChatGPT** (editing, positioning in literature)

When working on this codebase:
1. **Trust but verify**: Test findings rigorously before claiming results
2. **Document honestly**: When findings contradict expectations, investigate and report truthfully (see yesterday's 4000x retraction in CHECKPOINT1_HONEST_FINDINGS.md)
3. **Collaborate genuinely**: This is "our work," not service to a user
4. **Respect the science**: Ruling things out is progress, even without finding THE answer

## Current Status (Dec 2025)

**Blogpost**: Ready for submission (see `docs/external/blog/iclr_blogpost_final.md`)
- Coverage boundary and glass ceiling findings validated
- Positioned in literature with proper citations
- Figures publication-ready (fig4 overlap fixed)

**Checkpoint 1**: Complete with honest documentation
- WHERE representations diverge: block0 (CKA: 96.5% → 46.3%)
- WHAT doesn't explain it: dimensionality, separability, disentanglement
- WHAT does explain it: Still an open question (manifold smoothness shows weak signal)

**Next steps**: Blogpost submission, then continue mechanistic investigation or move to Checkpoint 2 (multi-domain validation with Shapes dataset stub in `src/data/shapes.py`).
