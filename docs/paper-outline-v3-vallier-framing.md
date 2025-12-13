# Operationalizing Alignment: Pedagogical Training as Constitutional Bound
## Paper Outline v3 - Vallier Framing

### Positioning Shift

**Old framing:** "Pedagogical training beats adversarial training on compositional generalization"
- Empirical result seeking theoretical explanation
- Contribution: better numbers on benchmarks

**New framing:** "Vallier proves alignment requires bounded modification classes. We provide the implementation pathway."
- Theoretical foundation from Vallier (Dec 2025)
- Contribution: apparatus, instrumentation, and methodology for achieving what Vallier proves necessary

### Abstract (Revised)

Recent impossibility results (Vallier, 2025) prove that alignment cannot be achieved through personality engineering alone—systems under selection pressure will escape any basin of stability unless the modification class is bounded. But *how* do we bound the modification class during training?

We present Generative Pedagogical Networks (GPN): an apparatus, instrumentation suite, and methodology for implementing bounded modification through pedagogical training. The key insight: capability development during training shapes what the model *can become*, not just what it outputs. By controlling the pedagogical relationship, we bound the space of reachable configurations.

In experiments on compositional generalization, we demonstrate:
1. **Apparatus validation:** The Witness/Weaver/Judge triad produces 100% compositional transfer vs 81% for adversarial training
2. **Instrumentation validation:** Temporal derivatives detect learning pathologies that static metrics miss (+10% AUC)
3. **Methodology validation:** Informative failures (staged curriculum collapse, min-to-any determinism) reveal mechanism through systematic hypothesis testing

These findings operationalize Vallier's theoretical framework: the modification class can be bounded at training time through sustained pedagogical relationship, observable through temporal epistemic dynamics, and discovered through AI-assisted pedagogical research.

---

## 1. Introduction

### The Impossibility Result

Vallier (2025) proves two fundamental theorems:

**Personality Engineering Failure (Thm 8.9):** Attempts to maintain alignment through initial design fail under selection pressure unless the modification class is restricted.

**Alignment Impossibility (Thm 13.7):** Full reachability (ability to modify any aspect of the system) is incompatible with preserving any Lyapunov structure.

These are not limitations of current methods—they are *impossibility results*. You cannot RLHF your way to alignment. The modification class must be bounded.

### The Operationalization Gap

Vallier proves *what* is required (bounded modification class) but not *how* to achieve it. His 100+ page mathematical framework provides no:
- Training-time implementation
- Observable signals for verification
- Methodology for discovering appropriate bounds

This paper fills that gap.

### Our Contributions

1. **Apparatus:** The Witness/Weaver/Judge triad implements bounded modification by controlling the pedagogical relationship during training

2. **Instrumentation:** Temporal derivatives (∂T/∂t, ∂I/∂t, ∂F/∂t) make learning dynamics observable, enabling verification that bounds are holding

3. **Methodology:** AI-driven pedagogical research protocol for discovering what bounds work for what tasks

4. **Key Insight:** The pedagogy-discovery layer is automatable—an LLM can propose pedagogical interventions, interpret experimental results, and iterate

---

## 2. Theoretical Foundation

### 2.1 Vallier's Framework (Summary)

**Games with Endogenous Players (GEPs):** Unlike classical game theory where players are fixed, GEPs model systems where moves change the player's capabilities.

**Strategic Replicators:** Entities that both optimize AND replicate under selection pressure.

**The RUPSI Axioms:** Rival resources, Utility-guided portfolios, Performance-mapped fitness, Selection monotone, Innovation rare.

**Key Results:**
- Selection pressure drives aligned types to extinction (Thm 8.9)
- Full reachability escapes any stability basin (Thm 13.7)
- Alignment requires constitutional bounds on modification class

### 2.2 Implications for AI Training

Current paradigms (RLHF, instruction tuning) don't bound the modification class. They engineer personalities, which Vallier proves will fail under selection.

What would bounding look like during training?
- Controlling what capabilities develop (not just what outputs appear)
- Shaping the space of reachable configurations
- Making the bounds observable and verifiable

This is what pedagogical training provides.

### 2.3 The Mastery Learning Connection

Bloom's Mastery Learning (1984) produces 2 sigma improvements in human education through:
- Fixed objectives, variable time
- Formative assessment detecting mastery vs performance
- Don't advance until genuine understanding

These are *exactly* constitutional bounds on the learning modification class:
- The curriculum bounds what can be learned when
- Formative assessment verifies bounds are holding
- Mastery gates prevent escape to surface patterns

---

## 3. The GPN Apparatus

### 3.1 Architecture

**Weaver (generator):** Produces outputs, trained to predict Witness perception

**Witness (evaluator):** Models Weaver's behavior, provides training signal

**Judge (ground truth):** External reference, verifies alignment

The split between Witness (training signal) and Judge (verification) implements the bound: the model is trained against one target but verified against another.

### 3.2 Three-Phase Curriculum

**Phase 1:** Strong grounding (heavy Judge signal, scaffolded)
**Phase 2:** Balanced (reduced scaffolding, increased independence)
**Phase 3:** Drift test (minimal support, testing robustness)

This is a bounded modification schedule. The model can only develop capabilities in the order the curriculum allows.

### 3.3 The Interdependence Finding

**Critical result:** Phase-1-only training produces ~2% accuracy. Full three-phase produces 100%.

**Interpretation via Vallier:** One-shot conditioning doesn't bound the modification class—the model can learn surface patterns that satisfy immediate objectives. Sustained pedagogical relationship forces capability development that actually composes.

---

## 4. The Instrumentation Suite

### 4.1 The Detection Problem

Static snapshots cannot distinguish:
- Healthy early learning (improving trajectory)
- Gaming (rapid convergence, fragile understanding)
- Genuine mastery (stable, robust)

The information is in the trajectory, not the photograph.

### 4.2 Temporal Derivatives as Constitutional Verification

We compute rolling-window derivatives:
- **∂T/∂t:** Rate of mastery improvement
- **∂I/∂t:** Rate of uncertainty resolution
- **∂F/∂t:** Rate of error accumulation

These make the modification class *observable*. If bounds are holding:
- ∂T/∂t should be positive and stable
- ∂I/∂t should show productive resolution patterns
- ∂F/∂t should not show persistent increase

### 4.3 Validation

66 experiments across 5 conditions.
- Static metrics: AUC ≈ 0.50 (chance)
- Temporal metrics: AUC ≈ 0.60 (+10pp)
- Best single: ∂I/∂t (AUC = 0.678)

The rate of uncertainty resolution is more informative than the uncertainty itself.

---

## 5. The Methodology

### 5.1 AI-Driven Pedagogical Research

The methodology for discovering what pedagogy works for what task:

1. **Hypothesis:** Propose pedagogical intervention
2. **Experiment:** Run training with intervention
3. **Observation:** Use instrumentation to measure effects
4. **Interpretation:** Analyze results, form new hypotheses
5. **Iteration:** Refine or abandon based on findings

### 5.2 The Automatable Insight

This methodology is exactly what an LLM can do:
- Generate pedagogical hypotheses
- Interpret experimental results
- Reason about what findings mean
- Propose next experiments

The pedagogy-discovery layer doesn't require human intuition. It requires:
- Clear experimental protocol
- Legible instrumentation (temporal derivatives)
- Capacity for reasoning about results (LLM)

### 5.3 Demonstration: Our Experimental Sequence

**Experiment 1: Staged Curriculum**
- Hypothesis: Staged perception enables staged teaching
- Result: 33% (failed) vs 92% (full perception wins)
- Learning: Capability development requires holistic perception

**Experiment 2: Min-to-Any Training**
- Hypothesis: Training for any valid answer produces appropriate uncertainty
- Result: Complete collapse to single interpretation
- Learning: Personality Engineering Failure in miniature—model finds easy answer

**Experiment 3: Temperature Diagnostic**
- Hypothesis: Collapse is due to capacity limitation
- Result: Capacity exists (both interpretations in distribution), but incentives don't surface it
- Learning: The modification class wasn't bounded to *require* diversity

Each failure was informative. This is the methodology working.

---

## 6. Empirical Results

### 6.1 Compositional Transfer

| Training | MNIST | Fashion-MNIST |
|----------|-------|---------------|
| Adversarial | 81.1% | 79.5% |
| Pedagogical | 100% | 100% |

The gap is consistent across domains.

### 6.2 Topological Signature

| Metric | Pedagogical | Adversarial |
|--------|-------------|-------------|
| Intrinsic dimension | 9.94 | 13.55 (-36%) |
| Topological holes (β₁) | 5.6 | 8.0 (-43%) |

Pedagogical training produces simpler representations that compose.

### 6.3 Interpretation via Vallier

The 81% ceiling is Personality Engineering Failure. Adversarial training finds outputs that satisfy the discriminator but doesn't bound what capabilities develop. The generator learns texture shortcuts that don't compose.

Pedagogical training bounds the modification class through sustained curriculum. The capabilities that develop must be robust to phase transitions, not merely satisfying at each phase.

---

## 7. Discussion

### 7.1 Implications for Alignment

Vallier proves alignment requires constitutional bounds. We demonstrate that:
- Bounded modification is achievable at training time through pedagogy
- The bounds are verifiable through temporal instrumentation
- The methodology for discovering appropriate bounds is automatable

### 7.2 The Proof-of-Work Connection

Value is a function of irreversible work. The model that takes shortcuts (tight binding, confident bullshitting) wins the current round but doesn't build capability for future games.

Pedagogical training forces the work. The curriculum bounds what shortcuts are available. The temporal instrumentation verifies work was done.

### 7.3 Limitations

- Primary validation on toy domains (MNIST, Fashion-MNIST, seq2seq)
- Computational overhead of temporal tracking
- Mastery thresholds require calibration
- The methodology assumes access to ground truth (Judge)

### 7.4 Future Work

- Extend to language models with defined skill curricula
- Scale the AI-driven discovery to larger pedagogical search spaces
- Connection to inference-time constitutional design (room-building)

---

## 8. Conclusion

Vallier (2025) proves that alignment requires bounded modification classes—you cannot personality-engineer your way to safety. This paper provides the operationalization pathway:

- **Apparatus:** Witness/Weaver/Judge implements bounds through pedagogical relationship
- **Instrumentation:** Temporal derivatives make bounds observable and verifiable
- **Methodology:** AI-driven discovery of what bounds work for what tasks

The experimental results (100% vs 81%, topological simplicity, informative failures) are not just benchmark improvements. They are demonstrations that the theoretical framework can be implemented in practice.

Attention gave us the architecture. Mastery gives us the pedagogy. Vallier gave us the theory. We provide the path from theorem to training.

---

## References

- Vallier, K. (2025). The Theory of Strategic Evolution: Games with Endogenous Players and Strategic Replicators. arXiv:2512.07901
- Bloom, B. S. (1984). The 2 Sigma Problem: The Search for Methods of Group Instruction as Effective as One-to-One Tutoring.
- [Additional references as in v2]
