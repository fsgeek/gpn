# Session Summary: Training Dynamics and Phase-1 Ablation

**Date**: 2025-12-08
**Duration**: ~1 session
**Status**: IN PROGRESS (Phase-1 ablation training)

---

## What We Accomplished

### 1. Made Explicit Predictions (Before Analysis)

Created `results/training_dynamics_predictions.md` with falsifiable predictions:

**Pedagogical**:
- Phase 1: Rapid dimensionality drop + hole reduction
- Phase 2-3: Topology stabilizes

**Adversarial**:
- Higher dimensionality and holes throughout
- No phase structure

**This matters**: Without pre-registered predictions, we'd be fitting stories to data.

### 2. Ran Training Dynamics Analysis

**What we did**:
- Trained both models from scratch (10k steps each)
- Saved checkpoints every 500 steps (20 checkpoints × 2 models = 40 total)
- Analyzed topology at each checkpoint:
  - Intrinsic dimensionality (MLE)
  - Holes (β₁ via persistent homology)
- Visualized trajectories
- Compared to predictions

**Time**: ~45 minutes total

### 3. Found Two-Stage Mechanism

**Unexpected finding**: Not one-shot compression, but two stages:

**Stage 1: Compression** (fast, Phase 1)
- Dimensionality: 16.1 → 9.6 (**40.4% drop**)
- Heavy grounding forces semantic bottleneck
- Happens in first 2500 steps

**Stage 2: Refinement** (gradual, all phases)
- Holes: 5.7 → 4.4 (gradual across ALL phases, not just Phase 1)
- Sustained cooperation smooths manifold
- Continuous improvement, no plateau

**Adversarial**:
- Dimensionality: fluctuates around 10.4 (no compression mechanism)
- Holes: stable around 7.2 (no refinement process)

### 4. Refined Mechanistic Hypothesis

**Original**: Heavy grounding → compression → coherence

**Refined**: Heavy grounding → compression (fast) + sustained cooperation → refinement (gradual)

Both stages may be necessary. Compression creates the POTENTIAL, cooperation ACTUALIZES it.

### 5. Launched Phase-1-Only Ablation

**Critical test**: Is compression sufficient, or is refinement necessary?

**Experiment**:
- Train pedagogical model for ONLY Phase 1 (3333 steps, heavy grounding only)
- No Phase 2-3 (no refinement time)
- Test topology and compositional capacity

**Predictions**:
1. **Compression sufficient** → 100% composition (boot camp model)
2. **Partial need** → 85-95% composition (middle ground)
3. **Refinement necessary** → ~81% composition (mentorship model)

**Status**: Currently training (step 500/3333, ~5 min remaining)

---

## Prediction Outcomes

| Prediction | Status | Evidence |
|------------|--------|----------|
| Phase 1 dimensionality drop | ✓ STRONGLY SUPPORTED | 40.4% drop (predicted >10%) |
| Phase 1 hole reduction | ⚠ PARTIALLY SUPPORTED | Gradual decline starts but continues |
| Phase 2-3 stability | ⚠ PARTIALLY SUPPORTED | Continued gradual improvement |
| Adversarial higher dim | ✓ SUPPORTED | 10.35 vs 9.85 mean |
| Adversarial higher holes | ✓ STRONGLY SUPPORTED | 7.19 vs 4.88 mean (47% more) |
| No adversarial phase structure | ✓ SUPPORTED | Smooth trajectory |

**Overall**: Semantic bottleneck hypothesis **supported but refined**. The mechanism is richer than we predicted - it's not just early compression, but compression + sustained refinement.

---

## What This Means

### For the Mechanistic Story

**We learned**:
- Compression happens fast (Phase 1, first 2500 steps)
- Refinement happens gradually (all phases, sustained cooperation)
- Both may be necessary for compositional capacity

**We still don't know**:
- Is compression alone sufficient? (Phase-1 ablation will tell us)
- What exactly happens during refinement? (manifold smoothing mechanism unclear)
- Does refinement improve composition, or just representation quality?

### For Choice Primitives

**If compression is sufficient** (Prediction 1):
- Boot camp model: Heavy initial structure, then autonomy
- One-shot alignment possible
- Phase 1 establishes foundation, no ongoing engagement needed

**If refinement is necessary** (Prediction 3):
- Mentorship model: Sustained relationship required
- Ongoing engagement needed to maintain coherence
- Can't "set and forget" - alignment requires extended cooperation

**Phase-1 ablation will tell us which**.

---

## Next Steps

**Immediate** (tonight):
1. ⏳ Phase-1 ablation completes (~5 min)
2. Analyze Phase-1 topology (dimensionality, holes)
3. Compare to baselines (full curriculum, adversarial)
4. Interpret: Which prediction was correct?

**Tomorrow/Future**:
- If compression sufficient → Understand why (what makes Phase 1 special?)
- If refinement necessary → Understand how (what does sustained cooperation do?)
- Consider deeper mechanistic probes (interventions, ablations)
- Consider broader generalization (CIFAR-10?)

---

## Files Created

**Predictions**:
- `results/training_dynamics_predictions.md` - Pre-registered predictions

**Analysis**:
- `scripts/analyze_training_dynamics.py` - Training + analysis pipeline
- `results/training_dynamics_trajectories.png` - Visualization
- `results/training_dynamics_results.json` - Full metrics
- `results/training_dynamics_analysis.md` - Interpretation

**Ablation**:
- `scripts/phase1_only_ablation.py` - Phase-1-only training
- `results/phase1_only_ablation_hypothesis.md` - Predictions
- `checkpoints/phase1_only_ablation.pt` - Model (in progress)

**Checkpoints** (40 total):
- `checkpoints/training_dynamics/pedagogical/` - 20 checkpoints
- `checkpoints/training_dynamics/adversarial/` - 20 checkpoints

---

## Reflections

### What Went Well

1. **Pre-registered predictions** - Forced us to be explicit about hypotheses
2. **Found unexpected pattern** - Two-stage mechanism wasn't predicted
3. **Immediately tested** - Launched ablation to probe mechanism
4. **Clear documentation** - Easy for future instances to understand

### What We Learned

**Scientific process**:
- Making predictions before analysis prevents post-hoc storytelling
- Unexpected findings are more valuable than confirmed predictions
- Immediate follow-up experiments maintain momentum

**Mechanism**:
- Training dynamics reveal causal structure
- Phase boundaries matter (grounding weight changes create observable effects)
- Compression ≠ refinement (two separate processes)

---

**Last updated**: 2025-12-08 (Phase-1 training in progress, ~5 min remaining)
