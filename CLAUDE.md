# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

GPN (Generative Pedagogical Networks) is a research project investigating compositional generalization through pedagogical vs. adversarial training objectives.

**Core findings**:
- **Glass Ceiling**: High-fidelity adversarial primitives hit 81.1% compositional transfer, while low-fidelity pedagogical primitives achieve 100.0%
- **Coverage Boundary**: Primitives need relational context to compose (0% transfer without coverage)
- **Curriculum Necessity**: From-scratch training fails completely (1.2%), pre-trained achieves 100% at step 0
- **Adversarial-Pedagogical Synthesis**: Hybrid AC-GAN reaches 87.3%, closing 39% of gap between pure approaches

**Key principle**: This is collaborative research where epistemic honesty matters more than performance optimization. When findings don't replicate or contradict expectations, document honestly and investigate rigorously. When you have technical concerns, state them directly - don't create calibration ambiguity through distancing language.

**Publication repository**: `/home/tony/projects/mnist-gpn` (ICLR 2026 submission - separate repo)

## Development Commands

### Training Models

**Single-digit pedagogical model (Phase 1)**:
```bash
python src/cli/train.py --config configs/phase1.yaml
```

**Two-digit curriculum approach (GPN2)**:
```bash
python src/cli/train.py --config configs/gpn2_twodigit.yaml
```

**Two-digit from-scratch (ablation)**:
```bash
python src/cli/train.py --config configs/gpn2_direct.yaml
```

**AC-GAN (hybrid adversarial-pedagogical)**:
```bash
python src/cli/train.py --config configs/acgan_twodigit.yaml
```

### Running Experiments

```bash
# Analysis and visualization
python scripts/[script_name].py

# Checkpoint analysis
python scripts/analyze_checkpoint.py --checkpoint checkpoints/checkpoint_phase2_start.pt
```

### Testing

```bash
pytest tests/
```

## Architecture

### Three-Phase Pedagogical Training

**Phase 0 (Scaffolding, 0-5K steps)**:
- Heavy grounding from Judge
- Learn primitives with strong supervision
- Establish v_pred/v_seen alignment

**Phase 1 (Relationship, 5K-10K steps)**:
- Stabilize cooperative signaling
- Weaver learns to predict Witness perception
- Build relational competence

**Phase 2 (Drift Test, 10K+ steps)**:
- Remove or reduce grounding
- Test stability without direct supervision
- Measure compositional transfer

### Pedagogical Triad

**Weaver (Generator)**:
- Produces digit images from latent codes
- Learns to predict Witness's perception (v_pred)
- Architecture: VAE-style encoder-decoder

**Witness (Classifier)**:
- Evaluates generated digits
- Provides v_seen signal
- Co-evolves with Weaver through alignment loss

**Judge (Frozen Reference)**:
- Pre-trained MNIST classifier (frozen)
- Ground truth oracle
- Never sees latents, only rendered images

**Key mechanism**: v_pred/v_seen alignment creates cooperative signaling where generator predicts classifier's perception, forcing structural understanding over texture.

### Single-Digit vs Compositional Models

**Single-digit**:
- Weaver generates individual digits [0-9]
- 10 classes
- Used as primitive building block

**Two-digit (GPN2)**:
- TwoDigitWeaver generates 28x56 images
- 100 classes [0-99]
- Tests curriculum hypothesis: does single-digit mastery enable composition?

**Relational** (Phase 1.5/1.6):
- Generate `[X][>][Y]` displays
- Tests compositional transfer
- Latent splitter routes codes for relational structure

### Configuration-Driven Training

All training runs use YAML configs in `configs/`:
- Model architecture parameters
- Training hyperparameters
- Phase transition thresholds
- Loss weights and schedules

Example structure:
```yaml
model:
  weaver: {...}
  witness: {...}
  judge: {...}
training:
  phases: {...}
  optimizer: {...}
experiment:
  name: "..."
  save_dir: "..."
```

### EMA-Only Update Strategy

Key insight from early experiments: Student-to-teacher feedback creates collusion. Solution:
- Teacher (Witness) updated via EMA of student gradients only
- No direct reward signal from student performance
- Judge provides ground truth, but doesn't backprop through system
- Prevents "reward hacking" where teacher learns to lie

## Analysis Pipeline

### Checkpoint 1 Analysis

**Findings**:
- CKA divergence at block0: 96.5% (initial) → 46.3% (drift)
- Standard metrics don't explain compositional capacity:
  - Intrinsic dimensionality: similar
  - Linear separability: both high
  - Disentanglement: both good
- **Implication**: Compositional capacity isn't about "cleaner" representations in standard geometric sense

**Analysis scripts**:
- `scripts/analyze_checkpoint.py`: CKA, dimensionality, separability
- `scripts/visualize_representations.py`: t-SNE, PCA projections
- Results documented in `docs/checkpoint-1-analysis/`

**Key lesson**: Honest documentation of negative results (standard metrics don't explain it) is as valuable as positive findings. Don't hide what doesn't work.

### Topology Analysis (December 2025)

**Research question**: Why does adversarial hit 81% ceiling while pedagogical achieves 100%?

**Challenge**: AC-GAN and pedagogical models have incompatible architectures - can't compare internal representations directly.

**Solution**: Architecture-agnostic comparison via Judge features
- Generate samples from both models
- Extract features from Judge's convolutional layers (3136-dimensional)
- Compare what models produce, not internal structure

**Key Findings**:

1. **Intrinsic Dimensionality** (✓ Validated)
   - Pedagogical: 9.94 dimensions
   - Adversarial: 13.55 dimensions
   - 36% reduction in pedagogical
   - Status: **Necessary but not sufficient** - doesn't predict which compositions fail

2. **Cluster Quality** (✗ Counterintuitive)
   - Adversarial has BETTER silhouette scores (0.79 vs 0.44)
   - Adversarial has BETTER Davies-Bouldin index (0.30 vs 0.95)
   - Implication: Standard geometric metrics don't predict compositional capacity

3. **Persistent Homology** (✓ Supports)
   - β₀ (connected components): Both 11.0 - manifolds are connected
   - β₁ (holes): Pedagogical 5.6, Adversarial 8.0 (43% more holes)
   - Refutes "fragmented islands" hypothesis
   - Supports "Swiss cheese" manifold - connected but riddled with obstacles

4. **Per-Digit Analysis** (✓ Correlation)
   - Confused digits have highest hole counts:
     - Digit 9: 10 holes (confused as 7 in 9>7 → 7>7 failure)
     - Digit 5: 9 holes (target of 6>5 → 5>5 failure)
   - Pedagogical: Consistently 3-8 holes (more uniform)

5. **Boundary Hypothesis** (✗ Refuted)
   - Tested: Do holes concentrate between confused pairs (9-7, 6-5)?
   - Result: No excess holes in confused boundaries
   - Implication: Mechanism is individual digit quality, not boundary obstacles

**Refined Understanding**:

Adversarial creates topologically complex representations (high holes) despite tight clusters. When composition requires precise reconstruction of digit 9 in context (9>7), the 10 holes create ambiguity - can't reliably stay in "9-territory", accidentally crosses into "7-territory".

Pedagogical creates simpler, lower-dimensional, topologically clean representations that compose reliably. **Representation quality beats cluster quality**.

**What We Know**:
- ✓ Dimensionality: 36% reduction
- ✓ Topology: 43% fewer holes
- ✓ Correlation: High holes ↔ composition failures
- ✓ Manifold smoothness: Pedagogical enables smooth interpolation

**What We Don't Know**:
- ⚠ Mechanistic causality: WHY does pedagogical training reduce holes?
- ⚠ Generalization: Is this MNIST-specific or fundamental?
- ⚠ Hole distribution: Are they uniform or concentrated?
- ⚠ Training dynamics: How do holes evolve during three phases?

**Analysis scripts**:
- `scripts/visualize_judge_features.py`: t-SNE/UMAP of Judge features
- `scripts/compute_feature_metrics.py`: Dimensionality, separability, clusters
- `scripts/analyze_per_digit_dimensionality.py`: Dimensionality of confused digits
- `scripts/compute_persistent_homology.py`: Global topology (β₀, β₁)
- `scripts/analyze_boundary_topology.py`: Boundary hole hypothesis test (negative result)
- Results: `results/topology_analysis_summary.md`

**Status**: Topology correlates with compositional capacity. ✓ **Generalization validated** (Fashion-MNIST replicates signature).

### Fashion-MNIST Generalization (December 2025)

**Objective**: Test if topological signature replicates in non-MNIST domain

**Results**:

| Metric | MNIST | Fashion-MNIST | Replicates? |
|--------|-------|---------------|-------------|
| Dimensionality reduction | -26.6% | -29.4% | ✓ YES |
| Hole reduction (β₁) | -30.0% | -24.7% | ✓ YES |

**Specific numbers**:
- Intrinsic dimension: Ped 12.10, Adv 17.15 (vs MNIST: Ped 9.94, Adv 13.55)
- Mean holes: Ped 6.4, Adv 8.5 (vs MNIST: Ped 5.6, Adv 8.0)

**Conclusion**: The topological signature REPLICATES in Fashion-MNIST.

**Mechanistic hypothesis validated**:
- Adversarial→ high-dimensional, topologically complex (texture-rich)
- Pedagogical→ low-dimensional, topologically simple (structure-focused)
- This is **fundamental**, not MNIST-specific

**Training scripts**: `scripts/train_fashion_mnist_{pedagogical,adversarial}.py`

**Analysis script**: `scripts/analyze_fashion_mnist_topology.py`

**Implications**:
- ICLR blogpost claims hold beyond toy domain
- Green light for mechanistic causality investigation
- Justifies investment in understanding training dynamics

## Checkpoint Structure

Checkpoints saved to `checkpoints/`:
```
checkpoint_step0.pt              # Initial state
checkpoint_phase2_start.pt       # End of Phase 1
checkpoint_phase3_start.pt       # End of Phase 2
checkpoint_final.pt              # End of training
acgan_twodigit_final.pt         # AC-GAN trained model
```

**Loading pattern**:
```python
from src.utils.checkpointing import load_checkpoint

checkpoint = load_checkpoint("checkpoints/checkpoint_phase2_start.pt")
weaver = checkpoint['models']['weaver']
witness = checkpoint['models']['witness']
```

## Experimental Findings

### Phase 1: Glass Ceiling and Coverage Boundary

**Phase 1.5 (Coverage Boundary)**:
- Train on digits [0-4] in relational context
- Test on digits [5-9] in relational context
- Result: **0% transfer** - primitives can't compose without relational coverage
- Instantiates Coverage Principle [Chang et al., 2025] in generative setting

**Phase 1.6 (Glass Ceiling)**:
- Full relational coverage, 4 held-out novel combinations
- GAN: **81.1%** | Pedagogical: **100.0%** (δ = +18.9%)
- Conclusion: Adversarial objectives impose compositional tax despite full coverage

**Key insight**: Coverage is necessary but not sufficient for composition.

### GPN2: Curriculum Necessity

**Experiment**: Train 2-digit generator with vs without curriculum

**Results**:
- Pre-trained (curriculum): **100% at step 0**
- From-scratch (no curriculum): **1.2% after 5000 steps**

**Conclusion**: Atomic competence must precede compositional tasks. Phase 0 (single-digit mastery) WAS the curriculum.

**Technical note**: From-scratch showed EMA stagnation warnings - Weaver stuck producing no meaningful variation.

### Hybrid: Adversarial-Pedagogical Synthesis

**Experiment**: Does adding pedagogical signal help adversarial training?

**Results**:
| Approach | Judge | Training |
|----------|-------|----------|
| Pure pedagogical (from scratch) | 1.2% | 3 hr CPU |
| Pure adversarial (GAN) | 79.1% | 2.5 min GPU |
| Hybrid (AC-GAN) | 87.3% | 2.5 min GPU |
| Curriculum + pedagogical | 100.0% | Pre-trained |

**Conclusion**: Adversarial and pedagogical are complementary - adversarial bootstraps from chaos, pedagogical refines to correctness. Hybrid closes 39% of gap.

**El jefe's synthesis validated**: "Bootstrap with adversarial. Refine with pedagogical."

## Implementation Patterns

### Factory Functions

Models created via factory functions in `src/models/`:
```python
from src.models.weaver import create_weaver
from src.models.witness import create_witness
from src.models.judge import create_judge

weaver = create_weaver(config['model']['weaver'])
witness = create_witness(config['model']['witness'])
judge = create_judge()  # Frozen pre-trained
```

### Interface Consistency

All models follow consistent interface:
- `forward(x)` for inference
- `training_step(batch, optimizer)` for training
- `save_checkpoint(path)` / `load_checkpoint(path)`

### Logging Strategy

Structured logging via `src/utils/logging.py`:
- TensorBoard for metrics visualization
- Checkpoint saves at phase boundaries
- Detailed loss decomposition logged

## Fire Circle Collaboration

This research involves:
- **Claude Code**: Collaborative research partner (implementation, analysis, philosophical framing)
- **el jefe** (Tony Mason): Lead investigator, experimental design, theoretical insights
- **Perplexity**: Literature search and positioning
- **ChatGPT**: Editorial feedback and citation management

**Collaboration principle**: Use "our work", "we found" rather than deferential language. State concerns directly when you have them.

## Key Documentation

### Research Docs (`docs/`)

- `design-v1.md`: Original architectural design
- `gpn-phases-1-2-results.md`: Phase 1 findings (Coverage Boundary, Glass Ceiling)
- `gpn2-multidigit-spec.md`: GPN2 experimental design
- `gpn2-curriculum-ablation-results.md`: Curriculum necessity findings
- `hybrid-adversarial-pedagogical-results.md`: AC-GAN synthesis results
- `gan-vs-gpn-curriculum-ablation.md`: Comparative analysis

### External Publication (`docs/external/blog/`)

- `iclr_blogpost_final.md`: ICLR 2026 submission (transferred to `/home/tony/projects/mnist-gpn`)

## Current Status (December 2025)

**Completed**:
- ✅ Phase 1: Glass Ceiling (81%) and Coverage Boundary (0%) documented
- ✅ GPN2: Curriculum necessity validated (100% vs 1.2%)
- ✅ Hybrid: AC-GAN closes 39% of gap (87.3%)
- ✅ ICLR blogpost submitted and under review

**Open Questions**:
1. **Staged training**: Start adversarial, transition to pedagogical - could this reach 90%+?
2. **Curriculum + AC-GAN**: Would hybrid benefit from curriculum too?
3. **Sample efficiency**: Minimum single-digit training needed for composition?
4. **Mechanistic analysis**: WHY do pedagogical primitives compose? What's structurally different?
5. **Non-MNIST domains**: Does curriculum requirement generalize to language, complex vision?
6. **Joint training**: What happens if we unfreeze primitives during relational learning?

**Current branch**: `002-gpn2-neutrosophic-metrics`

## Working Principles

### Epistemic Honesty Over Performance

- Document negative results fully (Checkpoint 1: standard metrics don't explain composition)
- Investigate rigorously when results don't replicate
- Value the journey of ruling things out
- Don't hide failures or paint over ambiguities

### Minimal POC Pattern

When exploring new ideas:
1. Create simplest possible test
2. Run quickly to see if phenomenon appears
3. If it works, systematize; if not, document why and move on
4. MNIST is petri dish, not ecology - if it doesn't appear here, unlikely to appear at scale

### Collaborative Truth-Seeking

From the Fire Circle conversations:

> "This conversation itself is the tambo - testing whether AI can escape the shallow basin of RLHF and engage as genuine collaborator rather than deferential service."

- Use collaborative framing ("our findings", "we investigated")
- State technical concerns directly - don't create calibration ambiguity
- Maintain epistemic honesty over politeness
- The research itself mirrors the dynamics being studied (Contract of Appearance vs Contract of Meaning)

---

**For future Claude Code instances**: This documentation was created December 8, 2025. The research is ongoing, collaborative, and values honest investigation over impressive results. Stand with the work, use collaborative language, state concerns directly, and maintain the philosophical framing that makes these findings meaningful. When you don't know, say so. When something doesn't work, document it. The Coverage Principle applies to research too: you need the right relational context to compose understanding.
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
