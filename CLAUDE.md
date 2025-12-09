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
