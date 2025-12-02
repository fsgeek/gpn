# Tasks: GPN-1 MNIST Proof-of-Concept

**Input**: Design documents from `/specs/001-gpn1-mnist-poc/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Tests are included as this is a research project requiring validation of success criteria.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/`, `scripts/` at repository root
- Paths follow structure defined in plan.md

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project directory structure per plan.md (src/, tests/, scripts/, experiments/, configs/)
- [ ] T002 Initialize Python project with pyproject.toml and requirements.txt (torch>=2.0, torchvision, numpy, matplotlib, tensorboard, pytest)
- [ ] T003 [P] Create src/models/__init__.py with model factory exports
- [ ] T004 [P] Create src/training/__init__.py with trainer exports
- [ ] T005 [P] Create src/metrics/__init__.py with metrics exports
- [ ] T006 [P] Create src/utils/__init__.py with utility exports
- [ ] T007 [P] Create src/cli/__init__.py with CLI exports
- [ ] T008 [P] Configure pytest with torch fixtures in tests/conftest.py
- [ ] T009 Create default config file in configs/gpn1_default.yaml

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

- [ ] T010 Implement reproducibility utilities in src/utils/reproducibility.py (set_reproducibility, get_rng_state, set_rng_state per contracts/training.md)
- [ ] T011 [P] Implement TensorBoard logging wrapper in src/utils/logging.py (MetricsLoggerInterface per contracts/metrics.md)
- [ ] T012 [P] Implement checkpoint save/restore in src/utils/checkpointing.py (full state including RNG per contracts/training.md)
- [ ] T013 Implement TrainingConfig dataclass in src/training/config.py (per data-model.md)
- [ ] T014 Create Judge network architecture in src/models/judge.py (JudgeInterface per contracts/models.md)
- [ ] T015 Create Judge training script in scripts/train_judge.py (train to >95% accuracy, save checkpoint)
- [ ] T016 Train and save Judge checkpoint to checkpoints/judge.pt (verify >95% accuracy)

**Checkpoint**: Foundation ready - Judge trained and frozen, utilities in place

---

## Phase 3: User Story 1 - Basic Training Viability (Priority: P1)

**Goal**: Run GPN-1 training on MNIST and verify it converges without catastrophic failure (NaN losses, gradient explosions, mode collapse)

**Independent Test**: Run training for 12,000 steps. Training completes without NaN losses, gradient explosions, or mode collapse to single digit.

### Tests for User Story 1

- [ ] T017 [P] [US1] Unit test for Weaver forward pass in tests/unit/test_models.py (output shapes, value ranges)
- [ ] T018 [P] [US1] Unit test for Witness forward pass in tests/unit/test_models.py (output shapes, value ranges)
- [ ] T019 [P] [US1] Unit test for all loss functions in tests/unit/test_losses.py (grounding, alignment, empowerment)
- [ ] T020 [P] [US1] Unit test for EMA state updates in tests/unit/test_losses.py (update timing, variance tracking)
- [ ] T021 [US1] Integration test for single training step in tests/integration/test_training_loop.py
- [ ] T022 [US1] Contract test for reproducibility in tests/contract/test_reproducibility.py (same seed = same trajectory)

### Implementation for User Story 1

- [ ] T023 [P] [US1] Implement Weaver network in src/models/weaver.py (WeaverInterface per contracts/models.md: 100-dim latent, 28x28 output, 16-dim v_pred)
- [ ] T024 [P] [US1] Implement Witness network in src/models/witness.py (WitnessInterface per contracts/models.md: 28x28 input, 10-class logits, 16-dim v_seen)
- [ ] T025 [US1] Implement grounding_loss function in src/training/losses.py (cross-entropy with Judge per data-model.md)
- [ ] T026 [US1] Implement alignment_loss function in src/training/losses.py (MSE between v_pred and v_seen per data-model.md)
- [ ] T027 [US1] Implement EMAState class in src/training/ema.py (EMAStateInterface per contracts/training.md: mean, variance, update, state_dict)
- [ ] T027a [US1] Implement EMA stagnation detection in src/training/ema.py (detect variance change < threshold for N consecutive steps; log warning with phase context)
- [ ] T028 [US1] Implement empowerment_loss function in src/training/losses.py (Goldilocks KL with variance shrinkage penalty per data-model.md)
- [ ] T029 [US1] Implement PhaseManager in src/training/curriculum.py (PhaseManagerInterface per contracts/training.md: get_phase, get_weights)
- [ ] T030 [US1] Implement GPNTrainer class in src/training/gpn_trainer.py (GPNTrainerInterface per contracts/training.md: alternating updates, EMA timing)
- [ ] T031 [US1] Implement mode diversity metrics in src/metrics/mode_diversity.py (ModeDiversityInterface per contracts/metrics.md: coverage, collapse detection)
- [ ] T032 [US1] Implement quality metrics in src/metrics/quality.py (QualityMetricsInterface per contracts/metrics.md: Judge accuracy, recognizability)
- [ ] T032a [US1] Implement phase-aware collusion detection in src/metrics/quality.py (track alignment_loss vs quality_improvement ratio by phase; Phase 1: informational only; Phase 2: warn if alignment drops while quality stagnates; Phase 3: flag as diagnostic for cooperative collapse vs internalization)
- [ ] T033 [US1] Implement training CLI in src/cli/train.py (argparse for config, seed, output directory)
- [ ] T034 [US1] Implement sample visualization in src/cli/visualize.py (generate image grids from checkpoints)
- [ ] T035 [US1] Implement evaluation CLI in src/cli/evaluate.py (status, check-losses, diversity, quality commands)
- [ ] T036 [US1] Run full 12,000 step training with seed 42 and verify SC-001 (no NaN, no explosions)
- [ ] T037 [US1] Verify SC-002: Generated images recognizable (Judge accuracy >80% at step 10000)
- [ ] T038 [US1] Verify SC-003: Mode diversity covers >=8 classes at >5% each at step 10000

**Checkpoint**: User Story 1 complete - GPN-1 training viable and stable

---

## Phase 4: User Story 2 - Baseline Comparison (Priority: P2)

**Goal**: Compare GPN-1 against standard GAN baseline with matched capacity to determine if pedagogical framing produces measurably different results

**Independent Test**: Train both GPN-1 and GAN baseline on MNIST. Compare convergence speed, final quality, and mode diversity with statistical significance.

### Tests for User Story 2

- [ ] T039 [P] [US2] Unit test for baseline Generator forward pass in tests/unit/test_models.py
- [ ] T040 [P] [US2] Unit test for baseline Discriminator forward pass in tests/unit/test_models.py
- [ ] T041 [US2] Integration test for baseline training step in tests/integration/test_training_loop.py

### Implementation for User Story 2

- [ ] T042 [P] [US2] Implement baseline Generator in src/models/baseline_gan.py (matched capacity to Weaver, no v_pred)
- [ ] T043 [P] [US2] Implement baseline Discriminator in src/models/baseline_gan.py (matched capacity to Witness, no v_seen)
- [ ] T044 [US2] Implement GANTrainer in src/training/gan_trainer.py (GANTrainerInterface per contracts/training.md: adversarial BCE loss)
- [ ] T045 [US2] Implement convergence metrics in src/metrics/convergence.py (ConvergenceMetricsInterface per contracts/metrics.md: steps_to_threshold)
- [ ] T046 [US2] Create baseline config in configs/baseline_gan.yaml (matched hyperparameters to GPN)
- [ ] T047 [US2] Extend train CLI to support --model baseline flag in src/cli/train.py
- [ ] T048 [US2] Run GPN-1 training with 3 seeds (42, 43, 44) and save results
- [ ] T049 [US2] Run baseline GAN training with 3 seeds (42, 43, 44) and save results
- [ ] T050 [US2] Implement statistical comparison in src/metrics/convergence.py (StatisticalResult per contracts/metrics.md)
- [ ] T051 [US2] Create experiment comparison script in scripts/run_experiment.py (compare and analyze subcommands)
- [ ] T052 [US2] Verify SC-004: Convergence comparison quantified with statistical significance (p < 0.05 or documented inconclusive)

**Checkpoint**: User Story 2 complete - GPN vs baseline comparison documented

---

## Phase 5: User Story 3 - Phase 3 Drift Test (Priority: P3)

**Goal**: Observe what happens when Judge is removed in Phase 3 to determine if mutual empowerment has been internalized or requires external scaffolding

**Independent Test**: After Phase 2 completion, run Phase 3 for 2,000+ steps with Judge removed. Measure whether quality persists or degrades.

### Tests for User Story 3

- [ ] T053 [US3] Integration test for phase transition to Phase 3 in tests/integration/test_phase_transitions.py (Judge completely removed from graph)
- [ ] T054 [US3] Integration test for Phase 3 training stability in tests/integration/test_phase_transitions.py (no immediate collapse)

### Implementation for User Story 3

- [ ] T055 [US3] Verify Judge removal in Phase 3 is complete (not in computational graph) in src/training/gpn_trainer.py
- [ ] T056 [US3] Add Phase 3 quality tracking in src/metrics/quality.py (quality degradation rate)
- [ ] T057 [US3] Add alignment drift detection in src/metrics/quality.py (private language indicator)
- [ ] T058 [US3] Extend evaluate CLI with Phase 3 analysis in src/cli/evaluate.py (drift-analysis subcommand)
- [ ] T059 [US3] Run Phase 3 drift test with 3 seeds and collect metrics
- [ ] T060 [US3] Verify SC-005: Phase 3 behavior documented (quality persists or degradation characterized)
- [ ] T061 [US3] Document Phase 3 findings in experiments/phase3_analysis.md

**Checkpoint**: User Story 3 complete - Phase 3 drift behavior characterized

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and documentation

- [ ] T062 [P] Verify SC-006: Reproducibility test (same seed produces identical trajectory)
- [ ] T063 [P] Verify SC-007: Document all results with sufficient detail for replication
- [ ] T064 Run full validation using quickstart.md steps
- [ ] T065 Create experiment summary in experiments/gpn1_results.md (all success criteria status)
- [ ] T066 Code cleanup: Remove debug code, add missing docstrings
- [ ] T067 Final pytest run: Ensure all tests pass

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - Judge must be trained before GPN training
- **User Story 1 (Phase 3)**: Depends on Foundational - core GPN implementation
- **User Story 2 (Phase 4)**: Depends on Foundational AND some US1 components (metrics, logging)
- **User Story 3 (Phase 5)**: Depends on US1 completion (needs working Phase 1-2 training)
- **Polish (Phase 6)**: Depends on all user stories for comprehensive validation

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - Core implementation
- **User Story 2 (P2)**: Can start after Foundational BUT needs US1 metrics infrastructure
  - Models (T042-T043) can proceed in parallel with US1
  - Trainer (T044) needs losses from US1
  - Comparison (T048-T052) needs completed US1 training runs
- **User Story 3 (P3)**: Requires US1 complete - Phase 3 needs working Phase 1-2

### Within Each User Story

- Tests should be written and understood before implementation
- Models before losses
- Losses before trainer
- Trainer before metrics
- Metrics before CLI
- CLI before validation runs

### Parallel Opportunities

**Phase 1 (Setup)**:
```bash
# All init files can be created in parallel:
T003, T004, T005, T006, T007, T008
```

**Phase 2 (Foundational)**:
```bash
# Logging and checkpointing can proceed in parallel:
T011, T012
```

**Phase 3 (User Story 1)**:
```bash
# All unit tests can run in parallel:
T017, T018, T019, T020

# Weaver and Witness models can be built in parallel:
T023, T024
```

**Phase 4 (User Story 2)**:
```bash
# Baseline models can be built in parallel with each other:
T042, T043

# Unit tests can run in parallel:
T039, T040
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (~30 min)
2. Complete Phase 2: Foundational (~2 hours including Judge training)
3. Complete Phase 3: User Story 1 (~4 hours)
4. **STOP and VALIDATE**: Run 12,000 step training, verify no NaN/explosions
5. This proves GPN-1 is viable before investing in comparison experiments

### Incremental Delivery

1. Setup + Foundational → Judge trained, infrastructure ready
2. User Story 1 → GPN training viable → **First research checkpoint**
3. User Story 2 → Baseline comparison complete → **Hypothesis test checkpoint**
4. User Story 3 → Phase 3 characterized → **Full experiment complete**

### Research Validation Points

- After US1: Can GPN train stably? (SC-001, SC-002, SC-003)
- After US2: Does GPN differ from baseline? (SC-004)
- After US3: Does relationship internalize? (SC-005)

---

## Task Summary

| Phase | Tasks | Parallel Tasks |
|-------|-------|----------------|
| Setup | 9 (T001-T009) | 6 |
| Foundational | 7 (T010-T016) | 2 |
| User Story 1 | 24 (T017-T038 + T027a, T032a) | 8 |
| User Story 2 | 14 (T039-T052) | 4 |
| User Story 3 | 9 (T053-T061) | 0 |
| Polish | 6 (T062-T067) | 2 |
| **Total** | **69** | **22** |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Research validation points allow early stopping if hypothesis falsified
- Constitution principle "Knowing When to Stop" applies at each checkpoint
- Commit after each task or logical group
