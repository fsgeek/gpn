# Implementation Plan: Neutrosophic Relationship Metrics (GPN-2)

**Branch**: `002-gpn2-neutrosophic-metrics` | **Date**: 2025-12-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-gpn2-neutrosophic-metrics/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Replace scalar health metrics in GPN with neutrosophic {T, I, F} state tracking to distinguish genuine synchronization from collusion/gaming. This preserves epistemic texture that single-number metrics destroy, enabling us to measure the quality of the pedagogical relationship between Weaver and Witness.

## Technical Context

**Language/Version**: Python 3.14
**Primary Dependencies**: PyTorch 2.x, existing GPN-1 codebase
**Storage**: TensorBoard logs, checkpoint files
**Testing**: pytest
**Target Platform**: Linux (CPU: Apple Silicon M4, GPU: NVIDIA RTX 4090)
**Project Type**: single (research ML project)
**Performance Goals**: <10% training overhead for neutrosophic computation
**Constraints**: Must integrate with existing GPN-1 without full rewrite
**Scale/Scope**: Extends GPN-1 (5000 step training runs, single-digit MNIST)

**Key Technical Questions (NEEDS CLARIFICATION)**:
1. How to compute T (truth) from observables?
2. How to compute I (indeterminacy) from observables?
3. How to compute F (falsity) from observables?
4. What are appropriate value ranges for T, I, F? (normalized [0,1]? raw counts?)
5. How to validate that T/I/F actually detect gaming vs genuine learning?

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. No Theater
✅ **PASS**: Neutrosophic metrics explicitly address theater risk. T/I/F tracking is designed to detect gaming and collusion that scalar metrics hide. The spec includes falsification criteria for validating that these metrics actually work.

### II. Adversarial to Our Own Ideas
✅ **PASS**: The spec includes explicit falsification criteria:
- If T/I/F cannot distinguish induced mode collapse from healthy learning → FALSIFIED
- If collusion doesn't increase F → FALSIFIED
- Validation experiments planned before deploying to real training

### III. Extension Over Consensus
✅ **PASS**: This extends GPN-1 by adding new measurement capability. Doesn't replace existing metrics, adds epistemic texture.

### IV. Process Embodies Claim
✅ **PASS**: We're studying pedagogical relationships, and neutrosophic metrics measure the *quality* of those relationships (not just scalar performance). The measurement method aligns with the pedagogical claim.

### V. Between-Instance Memory
✅ **PASS**: All design decisions documented in spec.md and plan.md. Research findings will be documented in research.md. Future instances can build on this without needing to rediscover rationale.

### VI. Knowing When to Stop
✅ **PASS**: Falsification criteria explicitly defined in spec.md. We commit to abandoning T/I/F if it cannot operationalize "genuine synchronization" as a computable value, or if it fails validation experiments.

**Overall Assessment**: ALL GATES PASS. Proceed to Phase 0 research.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/
├── models/
│   └── neutrosophic_tracker.py      # NEW: T/I/F computation and tracking
├── training/
│   └── gpn_trainer.py                # MODIFIED: Integrate neutrosophic tracking
├── utils/
│   ├── logging.py                    # MODIFIED: Log T/I/F values
│   └── visualization.py              # NEW: Visualize T/I/F trajectories
└── cli/
    └── train.py                      # MODIFIED: Add neutrosophic logging flags

tests/
├── unit/
│   └── test_neutrosophic_tracker.py  # NEW: Unit tests for T/I/F computation
└── integration/
    └── test_neutrosophic_validation.py  # NEW: Validate T/I/F detect gaming

experiments/
└── neutrosophic_validation/          # NEW: Synthetic failure mode experiments
    ├── mode_collapse.py
    ├── collusion.py
    └── healthy_baseline.py
```

**Structure Decision**: Single project structure (existing GPN codebase). Neutrosophic tracking is added as a new module (`neutrosophic_tracker.py`) that integrates with existing trainer infrastructure. Validation experiments are added to `experiments/` to test that T/I/F metrics actually detect gaming.

## Complexity Tracking

No constitution violations. This table is not needed.
