# Feature Specification: Neutrosophic Relationship Metrics (GPN-2)

**Branch**: `002-gpn2-neutrosophic-metrics` | **Date**: 2025-12-04
**Depends on**: GPN-1 (single-digit MNIST baseline)

## Problem Statement

Current GPN implementation uses scalar health metrics that destroy epistemic texture. We cannot distinguish:
- Genuine synchronization vs collusion/gaming
- Honest uncertainty vs degraded signal
- True alignment vs pattern-matched compliance

Scalar metrics (single numbers) force us to collapse complex relational states into a single dimension, losing critical information about the **quality of the pedagogical relationship**.

## Proposed Solution

Replace scalar health tracking with **neutrosophic feedback structure** using {T, I, F} state tracking:

- **T (Truth)**: Evidence of genuine synchronization between Weaver and Witness
- **I (Indeterminacy)**: Honest uncertainty about internal states
- **F (Falsity)**: Evidence of collusion, gaming, or misalignment

This preserves the epistemic texture that scalar metrics destroy.

## Requirements

### Functional Requirements

1. **Neutrosophic State Tracking**
   - Track {T, I, F} values for Weaver-Witness relationship
   - Compute T, I, F from observable behaviors (not latent states)
   - Update neutrosophic state at each training step
   - Log neutrosophic values alongside traditional metrics

2. **Truth (T) Detection**
   - Measure genuine synchronization: Do Witness observations match Weaver claims when both are verifiable by Judge?
   - Require evidence of learning progress (not just static alignment)
   - Distinguish from trivial synchronization (e.g., always outputting same class)

3. **Indeterminacy (I) Detection**
   - Track epistemic uncertainty in Weaver's v_pred or Witness's v_seen
   - Measure variance/entropy in attribute predictions
   - Distinguish honest uncertainty from noise

4. **Falsity (F) Detection**
   - Detect collusion: Are Weaver and Witness aligned but both wrong (Judge disagrees)?
   - Detect gaming: Is Witness accuracy high but diversity low (mode collapse)?
   - Detect misalignment: Do Weaver claims diverge from Witness observations?

5. **Validation Against Known Failure Modes**
   - Test that {T, I, F} can detect mode collapse (high F, low T)
   - Test that {T, I, F} can detect collusion (high alignment, low Judge accuracy)
   - Test that {T, I, F} can detect healthy learning (increasing T, decreasing I, low F)

### Non-Functional Requirements

1. **Computational Cost**: Neutrosophic computation must not significantly slow training (<10% overhead)
2. **Interpretability**: T, I, F values must be human-interpretable (documented semantics)
3. **Reproducibility**: Same training run should produce same neutrosophic trajectories
4. **Integration**: Must work with existing GPN-1 architecture without requiring full rewrite

## Success Criteria

### Must Have
- [ ] T, I, F values computed and logged for every training step
- [ ] T increases during Phase 1 (scaffolding with Judge active)
- [ ] F remains low (<0.3) during successful training
- [ ] F increases (>0.5) when we artificially induce mode collapse
- [ ] Neutrosophic state distinguishes healthy learning from gaming in validation experiments

### Should Have
- [ ] Visualizations showing {T, I, F} trajectories over training
- [ ] Documentation explaining how to interpret T, I, F values
- [ ] Ablation study: Does neutrosophic tracking catch failures that scalar metrics miss?

### Could Have
- [ ] Use neutrosophic state to adaptively adjust training (e.g., increase grounding weight when F is high)
- [ ] Multi-dimensional neutrosophic tracking (separate T/I/F for different aspects: classification, diversity, alignment)

## Out of Scope

- Pedagogical RLHF (GPN-3) - that comes after this
- Multi-agent Fire Circle (GPN-4) - that comes much later
- Neutrosophic logic formalism (we use intuitive T/I/F, not full neutrosophic sets)

## Hypotheses to Test

**H1**: Neutrosophic metrics {T, I, F} can distinguish genuine learning from gaming where scalar metrics cannot.

**Falsification criteria**:
- If induced mode collapse shows same T/I/F pattern as healthy learning → FALSIFIED
- If collusion (high alignment, low Judge accuracy) does not increase F → FALSIFIED
- If we cannot operationalize "genuine synchronization" as computable T value → REFORMULATE

**H2**: Phase 3 drift (when Judge removed) shows different neutrosophic signatures for internalized vs fragile relationships.

**Falsification criteria**:
- If all Phase 3 runs show same T/I/F regardless of Phase 1-2 performance → FALSIFIED
- If T/I/F cannot predict whether relationship holds post-Judge → NOT INFORMATIVE

## Dependencies

- GPN-1 baseline (completed): Single-digit MNIST with Weaver/Witness/Judge
- Existing empowerment loss infrastructure
- Judge evaluation framework

## Alternatives Considered

1. **Multi-dimensional scalar tracking**: Track alignment, diversity, accuracy separately as scalars
   - Rejected: Still doesn't capture the epistemic texture of T/I/F distinction

2. **Confidence-weighted metrics**: Add uncertainty estimates to scalar health
   - Rejected: Doesn't distinguish uncertainty (I) from falsity (F)

3. **Adversarial validation**: Add discriminator to detect gaming
   - Rejected: Adds adversarial dynamics we're trying to avoid

## References

- `docs/design-v1.md` (lines 44-49): Original neutrosophic metrics proposal
- `docs/gpn-phases-1-2-results.md`: GPN-1 baseline results
- Neutrosophic logic: Smarandache, F. (1999). A Unifying Field in Logics: Neutrosophic Logic

## Open Questions

1. How do we compute T from observables? → **Resolved in research.md**: Weighted combination of alignment + Judge accuracy + learning progress
2. What thresholds distinguish healthy vs unhealthy F values? → **Resolved in research.md**: F > 0.7 (mode collapse), F > 0.6 (collusion), F > 0.5 (gaming), F < 0.3 (healthy)
3. Can we validate T/I/F using synthetic failure modes before real experiments? → **Resolved in research.md**: Yes—four validation experiments designed (mode collapse, collusion, healthy baseline, gaming)
