# SCAN-lite Experiment: Comprehensive Analysis

## Overview

We ran the SCAN-lite experiment twice:
1. **Original (flawed)**: Grounding loss predicted action type only (4 classes)
2. **Fixed**: Grounding loss predicted full compositional command (64 classes)

## Results Comparison

### Original Implementation (Flawed Pedagogy)

| Seed | Pedagogical | Adversarial | Delta |
|------|-------------|-------------|-------|
| 0 | 75% (3/4) | 75% (3/4) | 0% |
| 1 | 100% (4/4) | 100% (4/4) | 0% |
| 2 | 50% (2/4) | 100% (4/4) | -50% |
| **Mean** | **75.0%** | **91.7%** | **-16.7%** |

### Fixed Implementation (Full Compositional Signal)

| Seed | Pedagogical | Adversarial | Delta |
|------|-------------|-------------|-------|
| 0 | 75% (3/4) | 75% (3/4) | 0% |
| 1 | 50% (2/4) | 100% (4/4) | -50% |
| 2 | 75% (3/4) | 100% (4/4) | -25% |
| **Mean** | **66.7%** | **91.7%** | **-25.0%** |

**Cohen's d = -1.73** (large effect in favor of adversarial)

## Key Findings

### 1. The Fix Made Things Worse

Counter to hypothesis, providing a richer grounding signal (64 classes vs 4) made pedagogical training WORSE:
- Mean dropped from 75.0% to 66.7%
- Variance increased (50-100% became 50-75%)

### 2. Adversarial Unchanged

Adversarial remained at 91.7% mean across both experiments - it doesn't depend on the Witness at all.

### 3. High Seed Variance Persists

With only 3 seeds and 4 test examples, variance is inherently high. Both pedagogical runs show one seed at 50%, one at 75%, and in the original a third at 100%.

## Why Didn't the Fix Help?

### Hypothesis 1: Harder Task, Less Signal

- 64-class classification is harder than 4-class
- During Phase 2, the Witness struggles more to learn the grounding target
- This creates a noisier v_seen signal for alignment
- Loss curves show higher values during Phase 2 with the fix (up to 4.58 vs ~0.5)

### Hypothesis 2: Compositional Structure in Output, Not Input

- In MNIST: Witness learns digit → v_seen represents digit quality
- In SCAN-lite: Witness learns command ID → v_seen represents... what?
- The output (action sequence) is what matters for composition, not the input command
- Maybe Witness should predict something about the OUTPUT structure

### Hypothesis 3: GPN Triad Architecture Mismatch

The GPN triad was designed for:
- **Continuous output** (images): v_pred predicts perceivable quality
- **Discrete output** (sequences): token probabilities don't have same "quality" notion

Alignment between Weaver's v_pred and Witness's v_seen may not transfer meaning in seq2seq:
- Image: "Will this look like digit 7?"
- Seq2seq: "Will this sequence be interpreted as command 42?" (not meaningful)

### Hypothesis 4: Adversarial Uses Supervised Signal

The "adversarial" baseline uses teacher forcing, which is essentially supervised learning. It's not a fair comparison to pure adversarial (REINFORCE with no ground truth).

## What This Tells Us

1. **Pedagogical advantage may be modality-specific**: Works for continuous outputs (images), not discrete sequences

2. **v_pred/v_seen alignment may not transfer**: The cooperative signaling that works in image generation doesn't have a natural analog in seq2seq

3. **The technique flaw was real but not the root cause**: Yes, predicting only action type was impoverished, but the fix revealed a deeper issue

4. **Domain generalization requires rethinking**: CIFAR-10 (images) is now higher priority than further SCAN-lite iteration

## Recommendations

1. **Proceed to CIFAR-10**: Image domain where GPN architecture was designed to work
2. **SCAN-lite is a valid negative result**: Document it as such - pedagogical doesn't transfer to seq2seq
3. **If revisiting SCAN**: Consider predicting OUTPUT structure, not input command
4. **Paper framing**: Position GPN as image-specific, not architecture-agnostic

## Files

- Original results: `results/scan_experiment/pedagogical_seed_*.json` (overwritten)
- Analysis: `results/scan_experiment/analysis.md` (original)
- This analysis: `results/scan_experiment/analysis_v2.md`
