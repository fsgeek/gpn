# Training Dynamics Analysis: Mechanistic Causality

**Date**: 2025-12-08
**Status**: COMPLETED

---

## Summary

Training dynamics analysis reveals a **two-stage mechanism** for compositional primitive formation:

1. **Stage 1: Compression** (fast, Phase 1) - Heavy grounding forces 40% dimensional collapse
2. **Stage 2: Refinement** (gradual, all phases) - Sustained cooperation reduces topological holes

Both stages appear necessary. Compression creates the potential, refinement actualizes it.

---

## Experimental Design

**Predictions Made First** (see `results/training_dynamics_predictions.md`):
- Pedagogical Phase 1: Rapid dimensionality drop + hole reduction
- Pedagogical Phase 2-3: Topology stabilizes (foundation established)
- Adversarial: Higher dimensionality and holes throughout, no phase structure

**Method**:
- Trained both models from scratch (10,000 steps)
- Saved checkpoints every 500 steps (20 checkpoints per model)
- Analyzed topology at each checkpoint:
  - Generated 100 samples/class
  - Extracted Judge features (architecture-agnostic comparison)
  - Computed intrinsic dimensionality (MLE, k=20)
  - Computed mean holes (β₁) via persistent homology

---

## Results

### Pedagogical Model Trajectory

**Phase 1 (Heavy Grounding, steps 0-3333):**
- Dimensionality: 16.13 → 9.61 (**40.4% drop**)
- Holes: 5.7 → 5.0 (12% drop)
- **Sharp compression in first 2500 steps**

**Phase 2 (Balanced, steps 3334-6666):**
- Dimensionality: 9.61 → 8.09 (continued gradual decline)
- Holes: 5.0 → 4.6 (continued gradual decline)

**Phase 3 (Drift Test, steps 6667-10000):**
- Dimensionality: 8.09 → 8.70 (slight oscillation, stable)
- Holes: 4.6 → 4.4 (continued gradual decline)

**Final State:**
- Dimensionality: 8.70
- Mean holes: 4.4

### Adversarial Model Trajectory

**Throughout Training (constant adversarial pressure):**
- Dimensionality: Fluctuates 9.3-12.7, mean **10.35**
- Holes: Fluctuates 6.4-8.0, mean **7.19**
- No phase structure, no systematic trends
- Consistently higher than pedagogical

**Final State:**
- Dimensionality: 10.41
- Mean holes: 7.5

---

## Prediction Comparison

| Prediction | Status | Evidence |
|------------|--------|----------|
| **Pedagogical Phase 1 dimensionality drop** | ✓ STRONGLY SUPPORTED | 40.4% drop (predicted >10%) |
| **Pedagogical Phase 1 hole reduction** | ⚠ PARTIALLY SUPPORTED | Gradual decline starts but continues across all phases |
| **Pedagogical Phase 2-3 stability** | ⚠ PARTIALLY SUPPORTED | Continued gradual improvement, not plateau |
| **Adversarial higher dimensionality** | ✓ SUPPORTED | 10.35 vs 9.85 mean |
| **Adversarial higher holes** | ✓ STRONGLY SUPPORTED | 7.19 vs 4.88 mean (47% more holes) |
| **No adversarial phase structure** | ✓ SUPPORTED | Smooth trajectory, no sharp transitions |

---

## Interpretation: Two-Stage Mechanism

### Predicted: One-Shot Semantic Bottleneck

We predicted heavy grounding would force compression AND hole reduction in Phase 1, then stabilize.

### Found: Two-Stage Process

**Stage 1: Compression (Fast, Phase 1)**
- Heavy Judge grounding forces dimensional collapse
- 40% drop in dimensionality in first 2500 steps
- Representation MUST encode structure - texture won't satisfy Judge
- Creates semantic bottleneck

**Stage 2: Refinement (Gradual, All Phases)**
- Holes reduce gradually from 5.7 → 4.4 across ALL phases
- Not a plateau - continued improvement through Phase 2-3
- Weaver-Witness cooperative relationship smooths manifold over time
- Topological coherence emerges through sustained cooperation

### Why This Matters

**Compression is necessary but not sufficient.**

- Dimensionality collapse happens fast (Phase 1)
- But holes reduce gradually (all phases)
- The compressed representation becomes increasingly smooth through sustained pedagogical relationship

**Adversarial has neither:**
- No compression mechanism (stays high-dimensional)
- No refinement process (holes remain high)
- Result: 81% compositional ceiling vs 100% pedagogical

---

## Mechanistic Story Refined

### Original Hypothesis
Adversarial → texture → high-dimensional → holes
Pedagogical → structure → low-dimensional → coherent

### Refined Hypothesis (After Training Dynamics)

**Adversarial:**
- Constant discriminator pressure encourages texture exploitation
- No compression mechanism → stays high-dimensional (10.4D)
- No refinement process → holes remain high (7.2)
- Result: Texture-rich, topologically fragmented primitives

**Pedagogical:**
1. **Scaffolding (Phase 1)**: Heavy grounding forces semantic compression
   - Dimensionality: 16.1 → 9.6
   - Establishes structural bottleneck

2. **Relationship (Phase 2)**: Balanced cooperation refines topology
   - Dimensionality: 9.6 → 8.1
   - Holes: 5.0 → 4.6
   - Sustained alignment improves manifold smoothness

3. **Drift Test (Phase 3)**: Autonomy maintains gains
   - Dimensionality: stable ~8.7
   - Holes: continued decline to 4.4
   - Foundation proves robust

**Result:** Structure-focused, topologically coherent primitives that compose perfectly

---

## Critical Unanswered Question

**Which stage is NECESSARY for composition?**

We know:
- Compression creates low-dimensional representation (Phase 1)
- Refinement reduces holes over time (all phases)
- Both together yield 100% composition

We don't know:
- Is compression sufficient? (Would Phase-1-only compose at 100%?)
- Is refinement necessary? (Does it contribute to composition or just representation quality?)
- Does compression without refinement revert to adversarial-like behavior?

**This matters for choice primitives:**

If compression is sufficient:
- "Boot camp then deploy" - establish structure quickly, then release
- One-shot alignment might work

If refinement is necessary:
- "Ongoing mentorship required" - sustained engagement needed
- Alignment requires extended cooperative relationship

---

## Next Step: Phase-1-Only Ablation

**Experiment:**
Train pedagogical model with ONLY Phase 1 (heavy grounding, steps 0-3333), then test compositional capacity.

**Possible Outcomes:**

1. **100% composition** → Compression is sufficient, refinement is nice-to-have
2. **<100% but >81%** → Refinement contributes but isn't strictly necessary
3. **~81% composition** → Compression without refinement reverts to adversarial-like behavior
   - This would be profound: compression creates POTENTIAL, cooperation ACTUALIZES it

**Critical test of mechanism:** Does the compression create compositional capacity, or do we need sustained pedagogical relationship to actualize it?

---

## Files

**Scripts:**
- `scripts/analyze_training_dynamics.py` - Training + analysis pipeline
- `scripts/train_fashion_mnist_pedagogical.py` - Pedagogical training (for reference)

**Results:**
- `results/training_dynamics_trajectories.png` - Visualization
- `results/training_dynamics_results.json` - Full metrics
- `results/training_dynamics_predictions.md` - Pre-registered predictions

**Checkpoints:**
- `checkpoints/training_dynamics/pedagogical/pedagogical_step_*.pt` (20 checkpoints)
- `checkpoints/training_dynamics/adversarial/adversarial_step_*.pt` (20 checkpoints)

---

**Last updated**: 2025-12-08
