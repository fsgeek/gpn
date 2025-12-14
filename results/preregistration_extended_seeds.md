# Pre-Registration: Extended Seed Experiment

**Date**: December 13, 2025
**Time**: (to be filled by git commit timestamp)
**Authors**: Tony Mason, Claude Code

## Initial Finding

From 5 seeds per condition:

| Condition | Mean | Std | Range |
|-----------|------|-----|-------|
| AC-GAN (frozen) | 80.1% | 2.8% | 75.2% - 82.9% |
| Architecture-only/Random (frozen) | 100.0% | 0.0% | 100% - 100% |
| Pedagogical (frozen) | 96.7% | 6.5% | 83.7% - 100% |

All conditions use identical architecture: frozen primitives + trainable latent splitter.

**Initial observation**: AC-GAN-trained primitives compose worse than untrained (random) primitives. Zero overlap between distributions.

## Script Verification (Completed)

Before pre-registration, we verified:

1. **Freezing**: PASSED - All three conditions correctly freeze primitives (requires_grad=False) and train only the splitter. Verified by checking that primitive weights don't change during training step.

2. **Checkpoint loading**: PASSED - Checkpoints are distinct. AC-GAN has different architecture (2M params). Pedagogical and Architecture-only have same architecture (4.7M params) but different weights (all 23 layers differ).

## Prediction

We predict that with additional seeds:

1. **AC-GAN will remain below 85%** - The 80% ceiling is robust, not a sampling artifact.

2. **Architecture-only (random) will remain at or near 100%** - Untrained primitives compose perfectly.

3. **Zero overlap will persist** - The worst random seed will exceed the best AC-GAN seed.

## Extended Experiment Design

- **Conditions**: AC-GAN, Architecture-only (random)
- **Additional seeds**: 10 per condition (total 15 per condition)
- **Metric**: Holdout relation accuracy on pairs [(7,3), (8,2), (9,1), (6,4)]
- **Training**: 5000 steps per seed
- **Evaluation**: 1000 samples per condition

## Commitment

We will report results regardless of outcome:

- **If prediction holds**: AC-GAN training actively hurts composition compared to random initialization.
- **If overlap emerges**: Our initial finding was a sampling artifact; more nuanced interpretation needed.
- **If random degrades**: Untrained primitive advantage was fragile; may depend on specific seeds.

## Analysis Plan

After running extended seeds:
1. Compute mean, std, min, max for each condition
2. Mann-Whitney U test for difference
3. Report effect size (Cohen's d)
4. Visualize distributions
5. Update findings document with honest interpretation

---

**This document will be committed to git before running extended experiment.**
