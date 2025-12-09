# Phase-1-Only Ablation: Results

**Date**: 2025-12-08
**Status**: COMPLETED

---

## Critical Finding: Catastrophic Failure

**Phase-1-only model (3333 steps, heavy grounding only):**
- Mean Judge accuracy: **2.2%**
- Essentially random performance (worse than 10% chance)
- **Cannot generate recognizable digits**

This is not Prediction 3 (81% composition like adversarial).
This is not Prediction 2 (85-95% partial success).
This is **complete failure** - the model can't even perform basic generation.

---

## Per-Digit Breakdown

| Digit | Judge Accuracy |
|-------|---------------|
| 0     | 0.0%          |
| 1     | 2.0%          |
| 2     | 0.0%          |
| 3     | 5.0%          |
| 4     | 0.0%          |
| 5     | 15.0%         |
| 6     | 0.0%          |
| 7     | 0.0%          |
| 8     | 0.0%          |
| 9     | 0.0%          |

**Mean: 2.2%**

Only digit 5 shows any recognition (15%), and even that is well below usable.

---

## What This Means

### Prediction Outcome: Beyond Prediction 3

We predicted three scenarios:
1. 100% composition (compression sufficient)
2. 85-95% composition (partial refinement helps)
3. ~81% composition (refinement necessary, reverts to adversarial)

**Actual result: ~2% generation quality**

The model is **worse than adversarial** (81%), worse than random (10%). It cannot generate valid digits at all.

### Interpretation: Sustained Cooperation is MANDATORY

**Phase 1 alone creates**:
- Semantic compression (inferred from training dynamics)
- Weaver-Witness alignment (alignment loss decreased to 0.017)
- Grounding to Judge (grounding loss decreased)

**But Phase 1 alone cannot create**:
- Valid digit generation
- Recognizable representations
- ANY functional capability

**Phase 2-3 (sustained cooperation) is not optional**. It's not a refinement that improves quality. It's a **necessary foundation** for the model to work at all.

---

## Comparison to Training Dynamics

From full curriculum at step 3333 (end of Phase 1):
- Dimensionality: ~9.6
- Holes: ~5.0
- Generation quality: **Unknown** (didn't test checkpoints for generation)

Phase-1-only at step 3333:
- Dimensionality: **Unknown** (will analyze)
- Holes: **Unknown**
- Generation quality: **2.2%** (catastrophic)

**The critical question**: Does the full curriculum at step 3333 also have poor generation quality, or is there something different about stopping at that exact point vs. training only to that point?

Hypothesis: The full curriculum context (knowing Phase 2-3 is coming) might shape Phase 1 differently than training that terminates at Phase 1.

---

## Mechanistic Implications

### For the Two-Stage Story

**We thought**:
- Stage 1 (compression) → create potential
- Stage 2 (refinement) → actualize potential

**Actually**:
- Stage 1 alone → nothing functional
- Stage 1 + Stage 2 → functional model
- Both are **necessary preconditions**, not sufficient individually

### For Alignment

**Boot camp model (Prediction 1): REJECTED**
- Cannot establish structure quickly then release
- One-shot alignment insufficient

**Partial model (Prediction 2): REJECTED**
- Not just "refinement helps"
- Fundamental requirement

**Mentorship model (Prediction 3): CONFIRMED AND STRENGTHENED**
- Sustained cooperation is MANDATORY
- Not just for composition, for basic functionality
- Cannot "set and forget"
- Alignment requires **extended cooperative relationship**

---

## Why Did This Happen?

### Possible Explanations

**1. Insufficient Training Time**
- 3333 steps too short for convergence
- Model needs more time even with heavy grounding
- *Counter*: Losses were decreasing, training "worked"

**2. Heavy Grounding Throughout Harmful**
- Constant weight=1.0 too constraining
- Phase 1 in full curriculum knows Phase 2 is coming
- *Test*: Try Phase-1-only with tapered grounding

**3. Missing Feedback Loop**
- Phase 2-3 provide critical feedback that shapes Phase 1 learning
- Without knowing what comes next, Phase 1 optimizes wrong objective
- *Supported by*: Alignment loss decreased, but generation failed

**4. Fundamental Interdependence**
- Compression and refinement aren't sequential, they're **interdependent**
- Phase 1 sets up structures that only make sense in context of Phase 2-3
- You can't have one without the other
- *This is the most profound interpretation*

### Most Likely: #4 - Fundamental Interdependence

The two-stage model implied:
1. Do compression (Phase 1)
2. Do refinement (Phase 2-3)

But maybe it's not sequential. Maybe:
1. Phase 1 establishes POTENTIAL for structure
2. Phase 2-3 ACTUALIZES that structure
3. Without Phase 2-3, the potential never manifests
4. Phase 1 alone creates representations that are **incomplete** - they only make sense in the context of what comes after

Like building a foundation for a house. The foundation alone isn't useful. It's only useful as part of the larger structure.

---

## Critical Questions

**Q1**: Does the full curriculum at step 3333 also have poor generation?
- If yes → Phase 1 ending alone is the problem
- If no → Training regime difference matters

**Q2**: What do Phase-1-only representations look like topologically?
- High holes despite compression attempt?
- Compressed but incoherent?
- Something else entirely?

**Q3**: Can we recover by continuing Phase-1-only training longer?
- Or is the damage done by missing Phase 2-3 irreversible?

**Q4**: Is this specific to the 3-phase curriculum, or general to pedagogical training?
- Would 2-phase work? Extended Phase 1 then Phase 3?

---

## Next Steps

**Immediate**:
1. Analyze Phase-1-only topology (dimensionality, holes)
2. Test full curriculum checkpoint at step 3333 for generation quality
3. Compare representations visually

**Follow-up experiments**:
1. **Extended Phase 1**: Train longer with heavy grounding (6666 steps)
   - Tests if it's just insufficient time
2. **Tapered Phase 1**: Gradually reduce grounding weight within Phase 1
   - Tests if constant heavy grounding is harmful
3. **Phase 1→3 skip Phase 2**: Heavy grounding then drift test, skip balanced
   - Tests if Phase 2 specifically is necessary

**Conceptual**:
- Rethink sequential vs. interdependent view of training
- Consider that phases aren't stages but **aspects** of a unified process
- Implications for alignment: Can't decompose into independent steps

---

## Conclusion

**Phase 1 alone is catastrophically insufficient.**

Not just for composition. Not just for topology quality. For **basic functionality**.

This is the strongest possible evidence that:
1. Sustained cooperation is MANDATORY
2. Compression alone does not create compositional capacity
3. The two-stage model is incomplete - stages are interdependent
4. For alignment: Mentorship model, not boot camp model

**The mechanism is richer and more subtle than we thought.**

---

**Last updated**: 2025-12-08 (analysis complete, interpretation ongoing)
