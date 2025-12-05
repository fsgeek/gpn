# Data Model: Neutrosophic Tracker

**Feature**: Neutrosophic Relationship Metrics (GPN-2)
**Date**: 2025-12-04

## Core Entity: NeutrosophicTracker

Tracks {T, I, F} state for Weaver-Witness relationship across training.

### Fields

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `T` | float | Truth: genuine synchronization (0-1) | 0 ≤ T ≤ 1 |
| `I` | float | Indeterminacy: honest uncertainty (0-1) | 0 ≤ I ≤ 1 |
| `F` | float | Falsity: gaming/collusion (0-1) | 0 ≤ F ≤ 1 |
| `T_ema` | float | Exponential moving average of T | 0 ≤ T_ema ≤ 1 |
| `I_ema` | float | Exponential moving average of I | 0 ≤ I_ema ≤ 1 |
| `F_ema` | float | Exponential moving average of F | 0 ≤ F_ema ≤ 1 |
| `judge_accuracy_ema` | float | EMA of Judge accuracy (for T computation) | 0 ≤ judge_accuracy_ema ≤ 1 |
| `ema_decay` | float | Decay rate for EMA (default 0.9) | 0 < ema_decay < 1 |

### State Transitions

```
Initialization (step 0):
  T = 0.0, I = 1.0, F = 0.0  # High uncertainty, no synchronization yet
  T_ema = 0.0, I_ema = 1.0, F_ema = 0.0

Each training step:
  1. Compute raw T, I, F from current batch observables
  2. Update EMAs: T_ema = ema_decay * T_ema + (1 - ema_decay) * T
  3. Log both raw and EMA values

Phase 1 (Scaffolding):
  Expected: T increases (0 → 0.6), I decreases (1.0 → 0.4), F stays low (<0.2)

Phase 2 (Relationship):
  Expected: T continues increasing (0.6 → 0.8), I continues decreasing (0.4 → 0.2), F stays low (<0.2)

Phase 3 (Drift Test):
  Healthy relationship: T stays high (>0.7), I stays low (<0.3), F stays low (<0.2)
  Fragile relationship: T drops, I increases, F may increase
```

## Dependent Entities

### NeutrosophicComponents

Internal breakdown of T, I, F into sub-components (for debugging/analysis).

| Field | Type | Description |
|-------|------|-------------|
| `alignment` | float | Weaver-Witness agreement (v_pred vs v_seen) |
| `judge_accuracy` | float | Judge correctness on generated images |
| `judge_improvement` | float | Rate of Judge accuracy increase |
| `weaver_uncertainty` | float | Variance in Weaver's v_pred |
| `witness_entropy` | float | Entropy in Witness's classification logits |
| `disagreement` | float | MSE between v_pred and v_seen |
| `collusion` | float | High alignment + low Judge accuracy |
| `mode_collapse` | float | Inverse of output diversity |
| `gaming` | float | Witness real acc > Witness gen acc |

**Note**: These components are logged for analysis but not required for T/I/F computation (which uses weighted combinations).

## Relationships

```
GPNTrainer
  ├── has one NeutrosophicTracker
  └── provides observables to tracker at each step

NeutrosophicTracker
  ├── receives v_pred from Weaver
  ├── receives v_seen from Witness
  ├── receives judge_logits from Judge
  ├── receives generated_images from Weaver
  └── computes T, I, F from observables

MetricsLogger
  └── logs T, I, F values to TensorBoard
```

## Data Flow

```
Training Step:
  1. Weaver generates images, outputs v_pred
  2. Witness classifies images, outputs v_seen and logits
  3. Judge evaluates images, outputs predictions
  4. NeutrosophicTracker.update(v_pred, v_seen, judge_logits, generated_images, labels)
  5. Tracker computes T, I, F
  6. Trainer logs T, I, F to MetricsLogger
```

## Invariants

1. **Range**: 0 ≤ T, I, F ≤ 1 at all times
2. **No constraint on sum**: T + I + F need not equal 1 (unlike classical probability)
3. **EMA stability**: EMA values should not oscillate wildly (use sufficient decay)
4. **Computational cost**: Metric computation should take <10% of training step time

