# Training Dynamics Predictions

**Date**: 2025-12-08
**Status**: PREDICTIONS BEFORE ANALYSIS

---

## Purpose

Before running training dynamics analysis, we explicitly state what our mechanistic hypotheses predict about topology evolution during training. This prevents post-hoc explanation fitting.

---

## Hypothesis: Semantic Bottleneck (Pedagogical)

**Mechanism**: Heavy Judge grounding forces semantic compression early, creating structural coherence. Weaver-Witness relationship maintains this foundation.

### Phase 1: Scaffolding (steps 0-3333, grounding weight=1.0)

**Prediction 1.1**: **Rapid dimensionality drop**
- Judge grounding forces Weaver to compress representations to structural essence
- Intrinsic dimensionality should drop sharply in first 1000-2000 steps
- Expected trajectory: High initial dim (12-15) → rapid drop → stabilize around final value (10-12)

**Prediction 1.2**: **Rapid hole reduction**
- Structural coherence emerges as Weaver learns to satisfy Judge
- β₁ (holes) should drop sharply in Phase 1
- Expected trajectory: Moderate initial holes (7-9) → rapid drop → stabilize around final value (5-6)

**Prediction 1.3**: **Alignment loss decreases**
- Weaver learns to predict Witness perception
- Alignment loss should drop steadily throughout Phase 1
- Indicates semantic bottleneck is forming

### Phase 2: Relationship (steps 3334-6666, grounding weight=0.5)

**Prediction 2.1**: **Dimensionality stabilizes**
- Foundation established in Phase 1
- May see small continued decrease, but no sharp changes
- Expected: Plateau or gentle decline

**Prediction 2.2**: **Holes remain low**
- Topology maintains coherence from Phase 1
- β₁ should stay near Phase 1 final values
- Expected: Stable low values, possibly minor fluctuations

**Prediction 2.3**: **Alignment loss continues to drop**
- Weaver-Witness relationship strengthens
- Indicates cooperative alignment improving

### Phase 3: Drift Test (steps 6667-10000, grounding weight=0.1)

**Prediction 3.1**: **Minimal topology change**
- Structural foundation is robust
- Dimensionality and holes should be stable
- Expected: Flat trajectories, minor noise

**Prediction 3.2**: **Alignment loss stabilizes**
- Relationship is established
- Low, stable alignment loss indicates maintained cooperation

---

## Hypothesis: Texture Exploitation (Adversarial)

**Mechanism**: Discriminator pressure encourages encoding fine-grained texture statistics. No semantic bottleneck, so representation remains high-dimensional and topologically complex.

### Throughout Training (constant adversarial pressure)

**Prediction A1**: **Dimensionality trajectory - "Texture Emergence"**
- Generator starts simple, learns to exploit texture over time
- Intrinsic dimensionality starts moderate, increases during training
- Expected trajectory: Moderate (10-12) → steady increase → high (14-17)

**Prediction A2**: **Dimensionality trajectory - "Texture From Start"** (alternative)
- Generator quickly learns texture is effective
- Dimensionality starts high, stays high
- Expected trajectory: High (14-17) throughout, relatively flat

**Prediction B1**: **Holes trajectory**
- Texture-rich representations create topological complexity
- β₁ could start moderate and increase as texture details accumulate
- OR start high and stay high
- Expected: Higher than pedagogical throughout, possibly increasing trend

**Prediction B2**: **No phase structure**
- Adversarial training has no phase-dependent mechanism
- Should see smooth trends, not sharp transitions
- Expected: Monotonic or stable trajectories, no jumps at step 3333 or 6667

---

## Alternative Hypotheses (to rule out)

### H_alt1: Gradual Refinement (Both Models)

**Prediction**: Both models show smooth, continuous improvement
- Linear or smooth exponential trends
- No phase-specific mechanisms
- **Would contradict**: Semantic bottleneck hypothesis (predicts Phase 1 is special)

### H_alt2: Final Cleanup

**Prediction**: Most improvement happens late in training
- Sharp topology changes in final 2000-3000 steps
- Early training doesn't matter much
- **Would contradict**: Early grounding hypothesis

### H_alt3: Random Walk

**Prediction**: No consistent pattern
- Noisy trajectories with no clear trend
- Large fluctuations, no convergence
- **Would contradict**: Any mechanistic story

---

## Success Criteria

**Semantic Bottleneck Hypothesis SUPPORTED if:**
1. ✓ Pedagogical shows sharp dimensionality drop in Phase 1
2. ✓ Pedagogical holes drop in Phase 1, stay low in Phase 2-3
3. ✓ Sharp transitions align with phase boundaries (steps 3333, 6667)
4. ✓ Adversarial shows higher dimensionality throughout
5. ✓ Adversarial holes higher throughout, no phase structure

**Semantic Bottleneck Hypothesis CHALLENGED if:**
1. ✗ Pedagogical shows gradual improvement across all phases (supports H_alt1)
2. ✗ Pedagogical topology changes most in Phase 3 (supports H_alt2)
3. ✗ No difference in trajectory shapes between pedagogical and adversarial
4. ✗ Adversarial shows phase-like structure (unexpected)

**Unexpected Findings to Investigate:**
- Adversarial dimensionality increases over training (texture emergence)
- Pedagogical holes increase in Phase 3 (drift failure)
- Non-monotonic trajectories (oscillations, reversals)

---

## Analysis Plan

1. **Train both models with checkpointing every 500 steps**
   - Pedagogical: 20 checkpoints (steps 500, 1000, ..., 10000)
   - Adversarial: 20 checkpoints (steps 500, 1000, ..., 10000)

2. **Extract topology at each checkpoint**:
   - Generate 100 samples per class
   - Extract Judge features
   - Compute intrinsic dimensionality (MLE, k=20)
   - Compute β₁ per digit, average across digits

3. **Visualize trajectories**:
   - Plot dimensionality vs step (both models on same plot)
   - Plot holes vs step (both models on same plot)
   - Mark phase boundaries (steps 3333, 6667) on pedagogical trajectory
   - Compare to predictions

4. **Quantify predictions**:
   - Phase 1 change: dim[step=3333] - dim[step=0]
   - Phase 2-3 stability: std(dim[step>3333])
   - Adversarial monotonicity: correlation(dim, step)

---

## Notes

- This analysis tests **causal mechanism**, not just final state
- Predictions are falsifiable and made before seeing data
- If predictions fail, we learn something important about the actual mechanism
- Phase boundaries are explicit in code (steps 3333, 6667 for 10k training)

---

**Next Steps**:
1. ✓ Write predictions (this document)
2. Implement training dynamics analysis script
3. Run analysis on MNIST models
4. Compare trajectories to predictions
5. Interpret results with epistemic honesty

---

**Last updated**: 2025-12-08 (predictions written BEFORE analysis)
