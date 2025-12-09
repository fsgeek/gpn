# Phase-1-Only Ablation: Hypothesis and Predictions

**Date**: 2025-12-08
**Status**: IN PROGRESS (training running)

---

## Motivation

Training dynamics analysis revealed a **two-stage mechanism**:

1. **Stage 1: Compression** (fast, Phase 1)
   - Heavy grounding → 40% dimensional collapse (16D → 10D)
   - Forces structural encoding

2. **Stage 2: Refinement** (gradual, all phases)
   - Holes reduce from 5.7 → 4.4 across all phases
   - Sustained cooperation improves manifold smoothness

**Critical Question**: Which stage is NECESSARY for compositional capacity?

---

## Hypothesis

**Compression alone may be insufficient for composition.**

The dimensional collapse happens fast (Phase 1), but topological refinement happens gradually (all phases). If composition requires smooth, hole-free representations, then Phase-1-only training should show:

- ✓ Dimensional compression (achieved in Phase 1)
- ✗ High topological holes (no refinement time)
- ? Compositional capacity uncertain

---

## Experiment Design

**Train pedagogical model for ONLY Phase 1:**
- 3333 steps (instead of 10,000)
- Heavy grounding throughout (weight=1.0)
- No Phase 2 (balanced) or Phase 3 (drift test)

**Compare to baselines:**
- Full curriculum: 8.7D, 4.4 holes, 100% composition
- Adversarial: 10.4D, 7.5 holes, 81% composition

---

## Predictions

### Prediction 1: Compression Sufficient (100% composition)

**If true**:
- Phase 1 creates the semantic bottleneck
- Compression forces structural encoding
- Topology refinement is nice-to-have, not necessary
- **Implication**: One-shot alignment works ("boot camp then deploy")

**Expected Metrics**:
- Dimensionality: ~10D (compressed from 16D in Phase 1)
- Holes: ~5.0 (no refinement, stays at Phase 1 endpoint)
- Composition: 100% (structural encoding is sufficient)

### Prediction 2: Partial Refinement Needed (85-95% composition)

**If true**:
- Compression creates foundation
- Some refinement helps but isn't strictly necessary
- Middle ground between adversarial and full curriculum

**Expected Metrics**:
- Dimensionality: ~10D
- Holes: ~5.0
- Composition: 85-95% (better than adversarial, worse than full)

### Prediction 3: Refinement Necessary (~81% composition)

**If true**:
- Compression creates POTENTIAL for composition
- Sustained cooperation ACTUALIZES that potential
- Phase-1-only reverts to adversarial-like behavior
- **Implication**: Ongoing mentorship required, not one-shot

**Expected Metrics**:
- Dimensionality: ~10D (compressed)
- Holes: ~5.0 (unrefined)
- Composition: ~81% (similar to adversarial despite compression)

**This would be profound**: It would mean compression and refinement are BOTH necessary. You can compress the representation space, but without sustained cooperation to smooth the manifold, compositional capacity doesn't emerge.

---

## Why This Matters

**For choice primitives and alignment:**

- **If Prediction 1**: Establish strong initial structure, then release
  - Boot camp model of alignment
  - Heavy initial constraints, then autonomy

- **If Prediction 3**: Sustained relationship required
  - Mentorship model of alignment
  - Ongoing engagement needed to maintain coherence
  - Can't just "set and forget"

**For the mechanistic story:**

- Prediction 1: Semantic bottleneck is sufficient
- Prediction 3: Semantic bottleneck + cooperative refinement both required

---

## Analysis Plan

Once training completes:

1. **Topology Analysis**:
   - Generate 100 samples/digit
   - Extract Judge features
   - Compute intrinsic dimensionality
   - Compute holes (β₁)

2. **Compare to Baselines**:
   - Full curriculum (10k steps): 8.7D, 4.4 holes
   - Adversarial: 10.4D, 7.5 holes
   - Phase-1-only: ?D, ? holes

3. **Compositional Test** (if available):
   - Dual-digit composition
   - Compare to full (100%) and adversarial (81%)

4. **Interpret**:
   - Which prediction was correct?
   - What does this tell us about mechanism?
   - What are implications for alignment?

---

## Current Status

- ⏳ Training Phase-1-only model (3333 steps, heavy grounding)
- ⏳ Estimated completion: ~10 minutes
- 📝 Topology analysis ready
- ❓ Compositional test TBD

---

**Last updated**: 2025-12-08 (training in progress)
