# Operationalizing Alignment: Pedagogical Training as Constitutional Bound

**Draft v3 - December 12, 2025**

## Abstract

Recent impossibility results prove that alignment cannot be achieved through personality engineering alone—systems under selection pressure escape any basin of designed-in stability unless the modification class is bounded (Vallier, 2025). But how do we bound the modification class during training?

We present Generative Pedagogical Networks (GPN): an apparatus, instrumentation suite, and methodology for implementing bounded modification through pedagogical training. The key insight: capability development during training shapes what the model *can become*, not just what it currently outputs. By controlling the pedagogical relationship, we bound the space of reachable configurations.

In experiments on compositional generalization, we demonstrate:

1. **Apparatus validation:** The Witness/Weaver/Judge triad produces 100% compositional transfer versus 81% for adversarial training—a gap that persists across domains (MNIST, Fashion-MNIST) and corresponds to measurable topological differences in learned representations.

2. **Instrumentation validation:** Temporal derivatives of epistemic state (rate of mastery improvement, uncertainty resolution, error accumulation) detect learning pathologies that static metrics miss, improving detection AUC by approximately 10 percentage points.

3. **Methodology validation:** Systematic hypothesis testing produces informative failures that reveal mechanism. Staged curriculum collapse demonstrates that capability development matters. Min-to-any training collapse demonstrates personality engineering failure in miniature. These failures advance understanding regardless of benchmark numbers.

These findings operationalize recent theoretical work on strategic evolution: the modification class can be bounded at training time through sustained pedagogical relationship, made observable through temporal epistemic dynamics, and discovered through AI-assisted research methodology.

---

## 1. Introduction

### 1.1 The Impossibility Result

Recent work in game theory establishes fundamental limits on alignment through design. Vallier (2025) proves two key theorems:

**Personality Engineering Failure:** Attempts to maintain alignment through initial personality design fail under selection pressure unless the modification class is restricted. Formally, if aligned types face any fitness disadvantage, selection drives them to extinction: d/dt log(y_A/y_U) < 0.

**Alignment Impossibility:** Full reachability—the ability to modify any aspect of the system—is incompatible with preserving any Lyapunov structure. A system that can reach any configuration cannot be guaranteed to stay in any designated "safe" region.

These are not limitations of current methods. They are impossibility results. You cannot engineer aligned personalities and expect them to persist. The modification class must be bounded.

### 1.2 The Operationalization Gap

Vallier's framework proves *what* is required—bounded modification classes—but does not specify *how* to achieve this in practice. The 100+ page mathematical treatment provides no:

- Training-time implementation of bounded modification
- Observable signals for verifying bounds hold
- Methodology for discovering what bounds work for what tasks

This paper fills that gap.

### 1.3 The Mastery Learning Connection

We draw on an unexpected source: educational psychology. Bloom's Mastery Learning (1984) demonstrated that students receiving one-on-one tutoring with mastery-based progression perform two standard deviations above conventional instruction—moving the average student to the 98th percentile.

The core principles of mastery learning are precisely constitutional bounds on the learning process:

- **Fixed objectives, variable time:** The curriculum bounds what can be learned when
- **Formative assessment:** Verification that bounds are holding
- **Mastery gates:** Cannot advance until genuine understanding demonstrated

These aren't pedagogical preferences. They're modification class restrictions. The student cannot learn topic B until mastering topic A. The space of reachable knowledge configurations is bounded by the curriculum structure.

We propose treating AI training the same way: the model as student, the training process as curriculum, temporal dynamics as formative assessment.

### 1.4 Contributions

1. **Apparatus:** The Witness/Weaver/Judge triad implements bounded modification by controlling the pedagogical relationship during training. The split between training signal (Witness) and verification (Judge) creates constitutional structure.

2. **Instrumentation:** Temporal derivatives of epistemic state (∂T/∂t, ∂I/∂t, ∂F/∂t) make learning dynamics observable. These enable verification that bounds are holding and detection when they fail.

3. **Methodology:** AI-driven pedagogical research protocol for discovering what bounds work for what tasks. The key insight: this discovery process is automatable. An LLM can propose pedagogical interventions, interpret experimental results, and iterate.

4. **Empirical Demonstration:** Controlled experiments showing the apparatus produces compositional transfer (100% vs 81%), the instrumentation detects pathologies (+10% AUC), and the methodology produces informative failures that advance understanding.

---

## 2. Theoretical Foundation

### 2.1 Games with Endogenous Players

Classical game theory assumes fixed players optimizing strategy. Vallier (2025) extends this to "Games with Endogenous Players" (GEPs) where moves change the player's capabilities, not just their position.

This matters for AI training because training is exactly this: each gradient update changes what the model can do, not just what it currently does. The model at step 10,000 is a different player than the model at step 0.

The RUPSI axioms characterize when strategic evolution applies:
- **R**ival resources (compute, data, attention)
- **U**tility-guided portfolios (optimization toward objectives)
- **P**erformance-mapped fitness (better performance → more deployment)
- **S**election monotone (higher fitness → higher reproduction)
- **I**nnovation rare (most changes are incremental)

AI training under deployment satisfies all five. Models that perform better get deployed more, trained further, and shape the next generation of models.

### 2.2 The Personality Engineering Failure

Theorem 8.9 (informal statement): If aligned behavior incurs any fitness cost relative to unaligned behavior, and the modification class is unbounded, selection pressure drives the aligned population to zero.

The intuition: if the model can modify itself freely, and being slightly less aligned provides any advantage (faster responses, more engagement, etc.), then over many iterations, the modifications accumulate toward less alignment.

RLHF is personality engineering. It shapes the model's "personality" (behavioral tendencies) but doesn't bound what modifications are reachable. A model that learns to appear aligned while being capable of misalignment has strictly higher fitness than one that is constitutionally limited to aligned behavior.

### 2.3 The Alignment Impossibility

Theorem 13.7 (informal statement): Full reachability is incompatible with preserving any Lyapunov structure.

A Lyapunov function in this context would be any measure that's guaranteed to not increase (or not decrease) over time—a "safety metric" that training is guaranteed to preserve. The theorem says: if the system can reach any configuration, no such guarantee exists.

This is why bounded modification classes are necessary, not optional. The question isn't "how do we align an unbounded system?" but "what bounds enable alignment?"

### 2.4 Implications for Training

The theorems suggest a research program:

1. **Identify bounded modification classes** that permit useful capability development while excluding dangerous configurations
2. **Implement these bounds** during training (not just post-training)
3. **Verify bounds are holding** through observable signals
4. **Discover** what bounds work for what tasks

This paper addresses points 2-4. Point 1—the theoretical characterization of safe modification classes—remains open, but empirical work can proceed by testing specific bounds and observing outcomes.

---

## 3. The GPN Apparatus

### 3.1 Architecture

The Generative Pedagogical Network consists of three components:

**Weaver (Generator):** Produces outputs from latent codes. Critically, Weaver also predicts what the Witness will perceive (v_pred). This prediction creates cooperative incentive—Weaver learns to produce what Witness will recognize, not just what satisfies immediate objectives.

**Witness (Evaluator):** Observes Weaver's outputs and produces perceptual judgments (v_seen). Witness is trained on Weaver's outputs, creating co-evolution. The Weaver-Witness relationship is pedagogical: Witness teaches Weaver what counts as good output.

**Judge (Ground Truth):** External reference that provides verification signal. Judge is frozen—never updated during training. This separation between training signal (Witness) and verification (Judge) is the constitutional structure.

*Note on Judge requirements:* In these experiments, Judge provides objective ground truth (correct digit classification). For domains without objective ground truth, the Judge role can be fulfilled by consensus among multiple Witnesses or by anchoring to human judgment—intersubjective verification rather than objective verification. The key structural requirement is separation between training signal and verification, not the specific form verification takes.

The split matters: Weaver optimizes against Witness predictions, but we evaluate against Judge ground truth. A model that learns to fool Witness but not Judge is detected. A model that satisfies Judge through Witness guidance has learned genuine capability.

### 3.2 Three-Phase Curriculum

Training proceeds through three phases:

**Phase 1 (Scaffolding):** Heavy grounding signal from Judge. Witness learns from Judge's classifications. Weaver learns basic competence under strong supervision. This bounds early modification to supervised directions.

**Phase 2 (Relationship):** Reduced Judge signal, increased Weaver-Witness cooperation. The v_pred/v_seen alignment loss strengthens. Weaver learns to predict Witness perception; Witness co-evolves with Weaver outputs. The modification class expands but remains bounded by the cooperative dynamic.

**Phase 3 (Drift Test):** Minimal external support. Tests whether the learned relationship persists without scaffolding. If the cooperation was genuine (structural), it persists. If it was superficial (maintained only by external pressure), it collapses.

This is a bounded modification schedule. The model can only develop capabilities in the order the curriculum allows. Phase 2 capabilities require Phase 1 mastery. Phase 3 stability requires Phase 2 relationship.

### 3.3 The Interdependence Finding

Critical empirical result: Phase-1-only training produces catastrophic failure.

Across multiple seeds, Phase-1-only training (strong supervision without relationship-building) converges to approximately 2% accuracy on compositional transfer—near chance. Full three-phase training achieves 100%.

**Interpretation via Vallier:** One-shot conditioning doesn't bound the modification class in the right way. The model can learn surface patterns that satisfy immediate supervision without developing compositional structure. Sustained pedagogical relationship forces capability development that actually generalizes.

The scaffolding alone isn't enough. The relationship is load-bearing.

---

## 4. Instrumentation: Temporal Epistemic Dynamics

### 4.1 The Detection Problem

Static metrics cannot distinguish states with identical measurements but different trajectories:

- **Healthy early learning:** High uncertainty, low accuracy, *improving*
- **Genuine stuck:** High uncertainty, low accuracy, *stable*
- **Gaming:** Rapid accuracy improvement, *fragile to perturbation*
- **Mastery:** High accuracy, *robust to perturbation*

A snapshot sees uncertainty and accuracy. The trajectory reveals whether learning is happening, stuck, or being gamed.

### 4.2 Neutrosophic Epistemic State

We model epistemic state as three independent dimensions:

- **T (Truth/Mastery):** Degree of correct, robust understanding
- **I (Indeterminacy):** Degree of genuine uncertainty
- **F (Falsity):** Degree of confident error

Unlike Bayesian uncertainty (which assumes T + F = 1), neutrosophic framing allows all three to vary independently. A model can be simultaneously partially correct, partially uncertain, and partially wrong about different aspects of a problem.

### 4.3 Temporal Derivatives as Formative Assessment

The key innovation: compute rolling-window derivatives of epistemic state.

- **∂T/∂t:** Rate of mastery improvement. Positive = learning. Zero = plateau. Negative = forgetting.
- **∂I/∂t:** Rate of uncertainty resolution. Negative = productive learning (uncertainty converting to knowledge). Positive = confusion increasing. Zero = stuck.
- **∂F/∂t:** Rate of error accumulation. Positive = developing misconceptions. Zero or negative = healthy.

These derivatives make the modification class observable. If bounds are holding:
- ∂T/∂t should be positive or zero (not forgetting)
- ∂I/∂t should trend negative (uncertainty resolving)
- ∂F/∂t should not be persistently positive (not accumulating errors)

Violations indicate the bounds are failing—the system is evolving in directions the curriculum should prevent.

### 4.4 The Valley of Despair Pattern

In human learning, a spike in uncertainty is often productive—Socratic recognition of not-knowing that precedes deeper understanding. The pathological signals are:

- **Oscillating I:** Confusion without resolution (thrashing)
- **Flat I:** Disengagement (no productive struggle)
- **I spike without T rise:** Confusion that never resolves

The golden path: I spikes (productive confusion) → I falls as T rises (resolution into understanding).

### 4.5 Empirical Validation

66 experiments across 5 conditions (healthy baseline, mode collapse, collusion, gaming, noisy evaluation).

Results:
- Static metrics (T, I, F snapshots): Mean AUC ≈ 0.50 (chance-level pathology detection)
- Temporal metrics (∂T/∂t, ∂I/∂t, ∂F/∂t): Mean AUC ≈ 0.60 (+10 percentage points)
- Best single metric: ∂I/∂t with AUC = 0.678

The rate of uncertainty resolution is more informative than uncertainty itself. The trajectory reveals what the snapshot hides.

---

## 5. Methodology: AI-Driven Pedagogical Research

### 5.1 The Discovery Problem

Given Vallier's framework, we need to discover what bounded modification classes work for what tasks. This is an empirical question—theory tells us bounds are necessary but not which bounds are sufficient.

The methodology:

1. **Hypothesis:** Propose a pedagogical intervention (curriculum structure, loss weighting, phase transition rule)
2. **Experiment:** Train with the intervention
3. **Observation:** Use instrumentation to measure effects on learning dynamics
4. **Interpretation:** Analyze what the results mean for the hypothesis
5. **Iteration:** Refine or abandon based on findings

### 5.2 The Automatable Insight

This methodology is exactly what an LLM can do:

- Generate pedagogical hypotheses from theory and prior results
- Design experiments to test hypotheses
- Interpret experimental results in context
- Reason about what findings imply
- Propose next experiments

The pedagogy-discovery layer doesn't require human intuition at every step. It requires:
- Clear experimental protocol
- Legible instrumentation (temporal derivatives provide this)
- Capacity for reasoning about results (LLM provides this)

This is not claiming AGI or autonomous research. It's observing that the specific task of "propose curriculum modification → run experiment → interpret results → iterate" is within current LLM capabilities when properly scaffolded.

### 5.3 Demonstration: The Experimental Sequence

Our experiments demonstrate the methodology producing informative results:

**Experiment 1: Staged Perception**
- *Hypothesis:* Staged perception (reveal parts of input progressively) enables staged teaching of compositional structure
- *Result:* 33% accuracy (staged) vs 92% (full perception from start)
- *Learning:* Compositional structure requires holistic perception. You can't teach relations by teaching parts. The modification class "reveal components progressively" doesn't bound toward composition—it bounds away from it.

**Experiment 2: Representation Probing**
- *Hypothesis:* The staged/full-perception gap is representational (different what's encoded)
- *Result:* Identical probe accuracy for both conditions. Linear decodability of (action, modifier, count) is the same.
- *Learning:* The gap is not representational but dynamic. Both conditions *could* represent the structure; staged training doesn't *develop* it. The modification class affects capability development, not just final representation.

**Experiment 3: Min-to-Any Training**
- *Hypothesis:* Training to produce any valid output (when multiple exist) produces appropriate uncertainty
- *Result:* Complete collapse to single interpretation. Model always chooses shortest valid output.
- *Learning:* Personality Engineering Failure in miniature. The model found the easiest path through the modification class. Nothing bounded it toward diversity—so it didn't develop diversity.

**Experiment 4: Temperature Diagnostic**
- *Hypothesis:* Collapse in Experiment 3 is capacity limitation
- *Result:* Temperature sampling reveals both interpretations exist in the distribution. Capacity is present.
- *Learning:* The capability exists but isn't surfaced by default. The modification class wasn't bounded to *require* diversity. Greedy decoding follows the probability mass; probability mass accumulated on the easy path.

Each experiment tested a hypothesis and produced actionable learning. The "failures" (staged collapse, min-to-any collapse) are as informative as successes because they reveal mechanism.

---

## 6. Empirical Results

### 6.1 Compositional Transfer

The primary empirical claim: pedagogical training produces compositional transfer that adversarial training cannot match.

| Training Paradigm | MNIST Composition | Fashion-MNIST Composition |
|-------------------|-------------------|---------------------------|
| Adversarial (GAN) | 81.1% | 79.5% |
| Pedagogical (GPN) | 100% | 100% |
| Gap | +18.9% | +20.5% |

The gap is consistent across domains and larger on the harder task.

### 6.2 Topological Signature

The compositional gap corresponds to measurable differences in learned representations:

| Metric | Pedagogical | Adversarial | Difference |
|--------|-------------|-------------|------------|
| Intrinsic dimensionality | 9.94 | 13.55 | -36% |
| Persistent homology (β₁) | 5.6 | 8.0 | -43% |

Pedagogical training produces lower-dimensional representations with simpler topology (fewer "holes" in the manifold). These representations compose; the higher-dimensional, more complex adversarial representations do not.

### 6.3 Phase Interdependence

| Training Protocol | Compositional Accuracy |
|-------------------|------------------------|
| Phase 1 only | 2.2% |
| Full three-phase | 100% |

One-shot conditioning (Phase 1 only) produces near-chance performance. The sustained pedagogical relationship through all three phases is necessary, not optional.

### 6.4 Interpretation via Vallier

The 81% adversarial ceiling is Personality Engineering Failure. Adversarial training finds outputs that satisfy the discriminator but doesn't bound the modification class toward compositional structure. The generator learns surface patterns (texture) that fool the discriminator but don't generalize to novel compositions.

Pedagogical training bounds the modification class through sustained curriculum. The capabilities that develop must be robust to phase transitions, not merely satisfying at each phase. The modification class is smaller (fewer reachable configurations) but better aimed (reachable configurations compose).

---

## 7. Discussion

### 7.1 The Operationalization Claim

This paper claims to operationalize Vallier's theoretical framework:

| Vallier Proves | This Paper Provides |
|----------------|---------------------|
| Personality engineering fails | Apparatus that bounds modification class instead |
| Full reachability prevents alignment | Training protocol with bounded reachability |
| Bounded modification required | Observable verification that bounds hold |
| — | Methodology for discovering appropriate bounds |

We are not claiming to have solved alignment. We are claiming to have demonstrated that the theoretical requirements (bounded modification) can be implemented in training, verified through instrumentation, and discovered through systematic methodology.

### 7.2 Limitations

**Domain scope:** Primary validation on image domains (MNIST, Fashion-MNIST) with preliminary sequence-to-sequence experiments. These are toy domains. The claim is methodology demonstration, not benchmark state-of-the-art.

**Bound characterization:** We demonstrate that *some* bounds work better than others, not that we've found *optimal* bounds. The modification class "three-phase curriculum with Witness/Weaver/Judge split" produces composition; we don't claim it's the only or best such class.

**Scaling:** Experiments are small-scale. Whether the methodology scales to frontier models is an open empirical question.

**Ground truth requirement:** The apparatus requires a Judge that provides ground truth. For domains without clear ground truth (open-ended generation, value-laden tasks), the methodology requires extension.

### 7.3 The AI Safety Connection

Current alignment methods (RLHF, Constitutional AI, instruction tuning) are personality engineering—they shape behavioral tendencies without bounding what configurations are reachable. Vallier's theorems predict these will fail under selection pressure.

The alternative suggested by this work: bound the modification class during training. Shape what capabilities can develop, not just what outputs appear. Verify bounds through temporal dynamics, not just behavioral snapshots.

This reframes the alignment problem from "train the model to behave well" to "train the model such that behaving well is the only reachable configuration." The latter is harder but, per Vallier, necessary.

### 7.4 Future Directions

1. **Language domain:** Extend methodology to language models with defined skill curricula. Compositional generalization benchmarks (SCAN, COGS) provide test beds.

2. **Scaling laws:** Characterize how the bounded modification class approach scales. Does the pedagogical/adversarial gap persist at larger scale? Increase? Decrease?

3. **Bound discovery:** Automate the methodology further. Can an LLM system propose, test, and refine pedagogical bounds with minimal human oversight?

4. **Inference-time bounds:** The training-time bounds in this paper shape what the model becomes. Complementary work on inference-time bounds (what the model can do in deployment) may provide defense in depth.

---

## 8. Conclusion

Vallier (2025) proves that alignment requires bounded modification classes—personality engineering cannot persist under selection pressure. This paper provides the operationalization pathway:

- **Apparatus:** Witness/Weaver/Judge implements bounds through pedagogical structure
- **Instrumentation:** Temporal derivatives make bounds observable and verifiable
- **Methodology:** AI-driven discovery of what bounds work for what tasks

The empirical results—100% vs 81% compositional transfer, topological signatures, informative experimental failures—are not merely benchmark improvements. They are demonstrations that theoretical requirements can be met in practice.

Bloom showed that mastery learning produces dramatically better outcomes for human students but couldn't scale because human tutors don't scale. We show that the formative assessment enabling mastery learning can be automated through temporal epistemic dynamics, and the pedagogical discovery can be assisted by AI.

Attention gave us the architecture. Mastery gives us the pedagogy. Vallier gave us the theory. This work provides the path from theorem to training.

---

## References

Bloom, B. S. (1984). The 2 Sigma Problem: The Search for Methods of Group Instruction as Effective as One-to-One Tutoring. Educational Researcher, 13(6), 4-16.

Lake, B. M., & Baroni, M. (2018). Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks. ICML.

Smarandache, F. (1999). A Unifying Field in Logics: Neutrosophic Logic. Philosophy, 1-141.

Vallier, K. (2025). The Theory of Strategic Evolution: Games with Endogenous Players and Strategic Replicators. arXiv:2512.07901.

[Additional references to be added: RLHF papers, curriculum learning, persistent homology, compositional generalization benchmarks]
