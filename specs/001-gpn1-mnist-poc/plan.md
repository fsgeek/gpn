# Implementation Plan: GPN-1 MNIST Proof-of-Concept

**Branch**: `001-gpn1-mnist-poc` | **Date**: 2025-12-02 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-gpn1-mnist-poc/spec.md`

## Summary

Implement Generative Pedagogical Networks proof-of-concept on MNIST to test whether architecturally-encoded cooperation produces qualitatively different training dynamics than adversarial framing. The architecture uses Weaver (generator with attribute prediction), Witness (classifier with attribute estimation), and frozen Judge (external grounding) in a three-phase training curriculum: Scaffolding (0-5000 steps), Relationship (5000-10000), and Drift Test (10000+). Includes GAN baseline for comparison.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: PyTorch 2.x, torchvision (MNIST), numpy, matplotlib (visualization), tensorboard (logging)
**Storage**: File-based checkpoints (.pt files), TensorBoard logs, generated image samples (PNG)
**Testing**: pytest with torch fixtures for reproducibility
**Target Platform**: Linux/macOS with CUDA optional (CPU training viable for MNIST scale)
**Project Type**: Single research project
**Performance Goals**: Complete 12,000 training steps in <2 hours on consumer GPU; CPU training acceptable
**Constraints**: Reproducibility primary constraint—same seed must produce identical trajectories
**Scale/Scope**: Single-task MNIST (60k training images), batch size 64, 12,000+ training steps

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. No Theater | ✅ Pass | Spec includes falsification criteria; negative results documented with same rigor as positive |
| II. Adversarial to Our Own Ideas | ✅ Pass | GAN baseline comparison required; mode collapse detection explicit; Phase 3 tests whether relationship holds without grounding |
| III. Extension Over Consensus | ✅ Pass | Research design captures both outcomes (relationship holds vs drift) as informative |
| IV. Process Embodies Claim | ✅ Pass | Specification developed through pedagogical dialogue, not extraction |
| V. Between-Instance Memory | ✅ Pass | All clarifications documented; reproducibility explicit requirement (FR-009, SC-006) |
| VI. Knowing When to Stop | ✅ Pass | Falsification criteria pre-specified: 5 hyperparameter variations before declaring falsified |

## Project Structure

### Documentation (this feature)

```text
specs/001-gpn1-mnist-poc/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Technology decisions (Phase 0)
├── data-model.md        # Entity definitions and relationships (Phase 1)
├── quickstart.md        # Getting started guide (Phase 1)
├── contracts/           # Interface contracts (Phase 1)
│   ├── models.md        # Weaver, Witness, Judge interfaces
│   ├── training.md      # Training loop contracts
│   └── metrics.md       # Metrics and logging contracts
├── checklists/
│   └── requirements.md  # Specification quality checklist
└── tasks.md             # Implementation tasks (69 tasks across 6 phases)
```

### Source Code (repository root)

```text
src/
├── models/
│   ├── __init__.py
│   ├── weaver.py          # Generator with v_pred output
│   ├── witness.py         # Classifier with v_seen output
│   ├── judge.py           # Frozen CNN classifier
│   └── baseline_gan.py    # Standard GAN for comparison
├── training/
│   ├── __init__.py
│   ├── gpn_trainer.py     # Three-phase GPN training loop
│   ├── gan_trainer.py     # Baseline GAN training loop
│   ├── losses.py          # Grounding, Alignment, Empowerment losses
│   ├── curriculum.py      # Phase transition logic
│   └── ema.py             # EMA state tracking for Witness
├── metrics/
│   ├── __init__.py
│   ├── mode_diversity.py  # Digit class coverage metrics
│   ├── quality.py         # Judge classification accuracy
│   └── convergence.py     # Convergence speed tracking
├── utils/
│   ├── __init__.py
│   ├── reproducibility.py # Seed management, deterministic ops
│   ├── logging.py         # TensorBoard integration
│   └── checkpointing.py   # State save/restore
└── cli/
    ├── __init__.py
    ├── train.py           # Training CLI entry point
    ├── evaluate.py        # Evaluation CLI
    └── visualize.py       # Sample generation CLI

scripts/
├── train_judge.py         # Pre-train Judge on MNIST
└── run_experiment.py      # Full experiment orchestration

tests/
├── unit/
│   ├── test_models.py
│   ├── test_losses.py
│   └── test_metrics.py
├── integration/
│   ├── test_training_loop.py
│   └── test_phase_transitions.py
└── contract/
    └── test_reproducibility.py

experiments/
└── [training runs stored here with configs and outputs]
```

**Structure Decision**: Single research project structure. Models, training, and metrics are separated for clarity. Scripts directory holds standalone tools (Judge training, experiment orchestration). Experiments directory stores training run outputs for reproducibility.

## Complexity Tracking

> **No violations requiring justification.** All constitution principles pass.

## Design Artifacts

| Artifact | Status | Description |
|----------|--------|-------------|
| [research.md](research.md) | ✅ Complete | Technology decisions: PyTorch, CNN architectures, loss implementations |
| [data-model.md](data-model.md) | ✅ Complete | Entity definitions: Weaver, Witness, Judge, EMAState, TrainingRun |
| [contracts/models.md](contracts/models.md) | ✅ Complete | Model interfaces and dimension constraints |
| [contracts/training.md](contracts/training.md) | ✅ Complete | Training loop, EMA, phase management contracts |
| [contracts/metrics.md](contracts/metrics.md) | ✅ Complete | Mode diversity, quality, convergence metrics |
| [quickstart.md](quickstart.md) | ✅ Complete | Installation, training, verification steps |

## Next Steps

1. ~~Run `/speckit.tasks` to generate implementation task list~~ ✅ Complete (69 tasks)
2. ~~Run `/speckit.analyze` to validate cross-artifact consistency~~ ✅ Complete (0 critical issues)
3. Create branch `001-gpn1-mnist-poc`
4. Begin implementation with `/speckit.implement`
