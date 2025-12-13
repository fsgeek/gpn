# SCAN-lite Transformer Experiment - Analysis

## Results Summary

| Seed | Pedagogical | Adversarial | Delta |
|------|-------------|-------------|-------|
| 0    | 75% (3/4)   | 75% (3/4)   | 0%    |
| 1    | 100% (4/4)  | 100% (4/4)  | 0%    |
| 2    | 50% (2/4)   | 100% (4/4)  | -50%  |
| **Mean** | **75.0%** | **91.7%** | **-16.7%** |

## Key Finding

**UNEXPECTED: Adversarial outperformed pedagogical by ~17 percentage points.**

This is the opposite of our hypothesis. We expected pedagogical ~95% and adversarial ~80%.

## Training Performance

Both conditions achieved 100% training accuracy on all seeds, demonstrating:
- Models can learn the SCAN-lite grammar perfectly
- Generalization gap is the distinguishing factor

## Analysis of Pedagogical Failure (Seed 2)

Pedagogical Seed 2 achieved only 50% on held-out compositions despite:
- 100% training accuracy
- Clean phase transitions (Phase 1→2 at step 500, Phase 2→3 at step 1500)
- High alignment loss in Phase 3 (4-6 range)

The high alignment loss suggests Witness perception (v_seen) diverged significantly from Weaver prediction (v_pred), potentially indicating instability in the cooperative dynamic for seq2seq tasks.

## Possible Explanations

### 1. Task is Too Easy
SCAN-lite has only 64 total examples, 4 held-out. Both conditions can memorize most of the grammar. Differences may be noise.

### 2. GPN Triad Doesn't Transfer to Seq2Seq
The Weaver/Witness/Judge dynamic was designed for image generation where:
- Witness evaluates rendered output quality
- Judge provides ground truth classification

In seq2seq:
- Witness receives variable-length action sequences
- Judge is deterministic (no neural uncertainty signal)
- The cooperative value signaling (v_pred ↔ v_seen) may not provide the same benefit

### 3. Adversarial Baseline Uses Supervised Learning
Our AdversarialSCANTrainer uses teacher-forcing for generation loss, which is essentially supervised learning. This is stronger than pure adversarial (would need REINFORCE-style training).

### 4. Small Test Set Variance
With only 4 held-out examples, each counts for 25%. One wrong prediction = 75%. Two wrong = 50%.

## Implications for Paper

### Option A: Accept as Negative Result
"GPN advantage does not generalize to seq2seq on small compositional tasks. This may indicate the mechanism is specific to continuous output modalities (images) where the Witness can extract meaningful perceptual signals."

### Option B: Run Harder Benchmark
Full SCAN or COGS benchmark with thousands of held-out compositions would reduce variance and test true compositional generalization.

### Option C: Revise Hypothesis
The pedagogical advantage may require different conditions:
- Longer training (more phase 2 time for cooperative learning)
- Different architecture (decoder-only, not encoder-decoder)
- Different task (generation, not translation)

## Recommendation

Document as exploratory negative result. The primary claims (100% vs 81% on images, topology signature, temporal derivatives) remain supported by MNIST/Fashion-MNIST evidence.

SCAN-lite shows that transferring GPN to fundamentally different architectures (Transformers) and modalities (seq2seq) requires additional research.

Proceed to CIFAR-10, which is closer to the original image domain and may show the expected pattern.
