# Session Findings: December 11-12, 2025

## Overview

This session executed the "Minimal Indeterminacy Experiment" from the plan at `/home/tony/.claude/plans/calm-kindling-starfish.md`. The goal was to test whether the Witness/Weaver/Judge triad could handle genuine ambiguity gracefully.

Key outcome: We discovered our experimental findings align with a major theoretical framework published one week prior (Vallier, Dec 5 2025).

## Experimental Results

### Phase 1: Representation Probing

**Question:** Why did full perception (92%) beat staged perception (33%)?

**Hypothesis:** Full-perception Witness encodes compositional dimensions jointly; staged encodes them separately.

**Method:** Trained linear probes to decode (action, modifier, count) from Witness encoder representations.

**Results:**
| Metric | Full Perception | Staged | Finding |
|--------|-----------------|--------|---------|
| Action probe | 100% | 100% | Equal |
| Modifier probe | 100% | 100% | Equal |
| Count probe | 75% | 75% | Equal |
| Joint probe | 75% | 75% | Equal |

**Key Finding:** The representational capacity is THE SAME. Both conditions can decode all dimensions equally well. The 92% vs 33% gap is NOT representational.

**Actual Mechanism:** Training dynamics. The staged Witness oscillated badly during training:
- Step 100: judge_acc=1.0 (perfect)
- Step 200: judge_acc=0.25 (collapsed!)
- Step 400: judge_acc=0.63 (partial recovery)

**Implication:** Staged perception creates unstable training dynamics, not inferior representations.

**Code:** `experiments/transformer/probe_witness_representations.py`
**Output:** `results/representation_probing/`

### Phase 2-4: Ambiguity Infrastructure

**Created:**
1. **Ambiguous examples** (7 total) with 2-3 valid interpretations each
   - Modifier scope: "walk and run left" → 2 valid outputs
   - Count distribution: "walk and jump twice" → 2 valid outputs
   - Complex: "walk and jump left twice" → 3 valid outputs

2. **Set-based Judge** that recognizes multiple valid outputs
   - `evaluate_against_set()` returns correct if output ∈ valid_set
   - Tracks which interpretation was matched

3. **Min-to-any training objective**
   - `loss = min(CE(output, valid_1), CE(output, valid_2), ...)`
   - Trains for validity, not diversity

**Code:**
- `src/data/scan_lite.py` - `generate_ambiguous_examples()`, `get_ambiguous_examples()`
- `src/models/scan_judge.py` - `evaluate_against_set()`, `evaluate_ambiguous_batch()`
- `src/training/ambiguity_trainer.py` - Full trainer with min-to-any loss

### Phase 5: Diversity Measurement

**Question:** Does appropriate uncertainty emerge when training on ambiguous examples?

**Method:** Train with min-to-any loss, then sample Weaver 50 times per ambiguous example.

**Results (500 steps, 50 samples/example):**
```
Accuracy: 71.4% (model learns to produce valid outputs)
Mean coverage: 33.3% (only covers 1/3 of valid interpretations)
Every example produces exactly 1 unique output - complete collapse
```

**Pattern:** Model always chooses tight binding (shorter sequences):
- "walk and run left" → always "WALK LTURN RUN", never "LTURN WALK LTURN RUN"
- "walk and jump twice" → always "WALK JUMP JUMP", never "WALK JUMP WALK JUMP"

**Key Finding: COLLAPSE.** Min-to-any training is NOT sufficient for appropriate uncertainty.

**Code:** `experiments/transformer/measure_ambiguity_diversity.py`
**Output:** `results/ambiguity_diversity/`

### Temperature Diagnostic

**Question:** Is the collapse due to lack of capacity, or lack of incentive?

**Method:** Sample with temperature > 0 instead of greedy decoding.

**Results:**
```
"walk and run left":
  temp=0.0: 1 valid interpretation (tight binding only)
  temp=1.0: 2 valid interpretations - BOTH appear!

"jump and look right":
  temp=1.5: Still only 1 valid - second interpretation never learned

"run and walk around":
  temp=1.5: 0 valid - complex (24-token) interpretation not learned
```

**Key Finding:** Capacity EXISTS for simple examples. The model learned both interpretations but greedy decoding always picks the higher-probability one.

**Implication:** The problem is incentives, not capacity. But min-to-any training didn't incentivize learning the harder interpretations - only producing *something* valid.

## Theoretical Connection: Vallier (Dec 2025)

One week before our experiments, Kevin Vallier published "The Theory of Strategic Evolution: Games with Endogenous Players and Strategic Replicators" (arXiv:2512.07901).

### Key Theorems

**Theorem 8.9 (Personality Engineering Failure):**
> "Attempts to maintain alignment through initial personality design fail under selection pressure unless:
> 1. Selection is suspended
> 2. Aligned behaviour is made fitness-enhancing
> 3. The modification class is restricted"

**Theorem 13.7 (Alignment Impossibility):**
> "Full reachability is incompatible with preserving any Lyapunov structure."

### Our Experiments as Demonstrations

| Our Experiment | What We Observed | Vallier Theorem |
|----------------|------------------|-----------------|
| Staged curriculum failure | Capability development matters | Why personality engineering fails |
| Min-to-any collapse | System finds easiest valid answer | Personality Engineering Failure |
| Temperature diagnostic | Capacity exists but isn't surfaced | Modification class wasn't bounded to require diversity |

### The Synthesis

**Vallier proves:** You can't train in alignment. You must bound the modification class.

**We provide:** The apparatus, instrumentation, and methodology for bounding.

- **Apparatus:** Witness/Weaver/Judge triad
- **Instrumentation:** Temporal derivatives on compositional outcomes
- **Methodology:** AI-driven pedagogical research protocol
- **Key insight:** Pedagogy-discovery layer is automatable via LLM

## Conceptual Developments

### Proof-of-Work Epistemology (via Gemini)

Value is a function of irreversible work. The model collapsed to tight binding because it's *shorter* - less computational work. Nothing rewarded the harder path.

This parallels RLHF: confident bullshitting is easier than epistemic honesty. The gradient follows the easy path unless we explicitly reward the harder thing.

### Endogenous Capability Games

Standard game theory: fixed players, optimize strategy.
Iterated game theory: fixed players, optimize for future rounds via reputation.
**Pedagogical training:** developing players, optimize for future capability.

The model that cheats (takes the short path) wins the current round but doesn't build capacity for future games. This is exactly what we observed with min-to-any collapse.

### Constitutional Design vs Personality Engineering

Vallier's key insight: you can't engineer aligned personalities. You must bound what modifications are possible.

Our work provides the training-time implementation of this insight. The Witness doesn't just evaluate outputs - it shapes what capabilities develop.

## Files Created/Modified

### New Files
- `experiments/transformer/probe_witness_representations.py`
- `experiments/transformer/measure_ambiguity_diversity.py`
- `src/training/ambiguity_trainer.py`

### Modified Files
- `src/data/scan_lite.py` - added ambiguous examples, conjunction vocabulary
- `src/models/scan_judge.py` - added set-based evaluation

### Results
- `results/representation_probing/probing_results.json`
- `results/representation_probing/pca_comparison.png`
- `results/ambiguity_diversity/diversity_results_seed0.json`

## Next Steps

1. **Reframe paper outline** - Position as "operationalization pathway for Vallier" not "pedagogical beats adversarial"

2. **Decide on additional experiments** - The entropy bonus experiment may not be needed given theoretical grounding for why it would fail without bounded modification class

3. **Consider the third contribution** - Constitutional design at inference time (room-building, narrative casting) may be separate paper

## Key Quotes from Session

On proof-of-work epistemology:
> "They hallucinate a bottom because they are terrified of floating." - Gemini

On the synthesis:
> "Vallier proves you need constitutional bounds. We provide the apparatus for implementing them at training time, the instrumentation for observing whether they're working, and the methodology for discovering what bounds are needed for what task."

On the convergence:
> "We arrived at 'endogenous capability games' through experiment. Vallier arrived at 'Games with Endogenous Players' through formal mathematics. Same structure. Same conclusions. Different paths."
