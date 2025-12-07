# Tasks: Neutrosophic Relationship Metrics (GPN-2)

**Input**: Design documents from `/specs/002-gpn2-neutrosophic-metrics/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Not explicitly requested in spec. Validation experiments serve as integration tests.

**Organization**: Tasks are grouped by validation experiments (serving as user stories) to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which validation experiment this task belongs to (e.g., VAL1, VAL2, VAL3, VAL4)
- Include exact file paths in descriptions

## Path Conventions

Single project structure: `src/`, `tests/`, `experiments/` at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and neutrosophic tracker foundation

- [ ] T001 Create neutrosophic tracker module structure in src/models/neutrosophic_tracker.py
- [ ] T002 [P] Create unit test file in tests/unit/test_neutrosophic_tracker.py
- [ ] T003 [P] Create validation experiments directory at experiments/neutrosophic_validation/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core neutrosophic tracker implementation that ALL validation experiments depend on

**⚠️ CRITICAL**: No validation experiment can run until this phase is complete

- [ ] T004 Implement NeutrosophicComponents.compute_alignment() in src/models/neutrosophic_tracker.py
- [ ] T005 [P] Implement NeutrosophicComponents.compute_judge_accuracy() in src/models/neutrosophic_tracker.py
- [ ] T006 [P] Implement NeutrosophicComponents.compute_weaver_uncertainty() in src/models/neutrosophic_tracker.py
- [ ] T007 [P] Implement NeutrosophicComponents.compute_witness_entropy() in src/models/neutrosophic_tracker.py
- [ ] T008 [P] Implement NeutrosophicComponents.compute_collusion() in src/models/neutrosophic_tracker.py
- [ ] T009 [P] Implement NeutrosophicComponents.compute_mode_collapse() in src/models/neutrosophic_tracker.py
- [ ] T010 [P] Implement NeutrosophicComponents.compute_gaming() in src/models/neutrosophic_tracker.py
- [ ] T011 Implement NeutrosophicTracker.__init__() with EMA initialization in src/models/neutrosophic_tracker.py
- [ ] T012 Implement NeutrosophicTracker.update() combining all components into T/I/F in src/models/neutrosophic_tracker.py
- [ ] T013 Implement NeutrosophicTracker.get_current_state() in src/models/neutrosophic_tracker.py
- [ ] T014 [P] Implement NeutrosophicTracker.get_components() in src/models/neutrosophic_tracker.py
- [ ] T015 [P] Implement NeutrosophicTracker.reset() in src/models/neutrosophic_tracker.py
- [ ] T016 Add unit tests for all NeutrosophicComponents methods in tests/unit/test_neutrosophic_tracker.py
- [ ] T017 Add unit tests for NeutrosophicTracker state management in tests/unit/test_neutrosophic_tracker.py

**Checkpoint**: Foundation ready - validation experiments can now run in parallel

---

## Phase 3: Validation Experiment 1 - Healthy Baseline (Priority: Required) 🎯 MVP

**Goal**: Verify neutrosophic metrics show expected healthy learning pattern (T↑, I↓, F low)

**Independent Test**: Run GPN-1 successful training with neutrosophic tracking, verify T increases from 0.2→0.8, I decreases from 0.7→0.3, F stays <0.3

### Implementation for VAL1

- [ ] T018 [P] [VAL1] Integrate NeutrosophicTracker into GPNTrainer.__init__() in src/training/gpn_trainer.py
- [ ] T019 [VAL1] Add neutrosophic state update call in GPNTrainer.train_step() in src/training/gpn_trainer.py (depends on T018)
- [ ] T020 [VAL1] Add neutrosophic logging (T, I, F, T_ema, I_ema, F_ema) in GPNTrainer._log_metrics() in src/training/gpn_trainer.py (depends on T019)
- [ ] T021 [P] [VAL1] Create healthy baseline validation script in experiments/neutrosophic_validation/healthy_baseline.py
- [ ] T022 [VAL1] Run healthy baseline experiment and verify expected T/I/F trajectories (depends on T018-T021)
- [ ] T023 [VAL1] Document baseline results in experiments/neutrosophic_validation/healthy_baseline_results.md

**Checkpoint**: Neutrosophic tracking integrated and validated on healthy GPN-1 training

---

## Phase 4: Validation Experiment 2 - Mode Collapse Detection (Priority: Required)

**Goal**: Verify neutrosophic metrics detect mode collapse (F > 0.7)

**Independent Test**: Force Weaver to generate only one class, verify F increases to >0.7

### Implementation for VAL2

- [ ] T024 [P] [VAL2] Create mode collapse induction script in experiments/neutrosophic_validation/mode_collapse.py
- [ ] T025 [VAL2] Implement forced single-class generation in mode_collapse.py (modify Weaver to ignore label)
- [ ] T026 [VAL2] Run mode collapse experiment and verify F > 0.7, T < 0.3 (depends on T024-T025)
- [ ] T027 [VAL2] Document mode collapse detection results in experiments/neutrosophic_validation/mode_collapse_results.md

**Checkpoint**: Mode collapse detection validated

---

## Phase 5: Validation Experiment 3 - Collusion Detection (Priority: Required)

**Goal**: Verify neutrosophic metrics detect collusion (high alignment, low Judge accuracy → F > 0.6)

**Independent Test**: Train Weaver and Witness with shared bias, verify they align but Judge accuracy is low, F > 0.6

### Implementation for VAL3

- [ ] T028 [P] [VAL3] Create collusion induction script in experiments/neutrosophic_validation/collusion.py
- [ ] T029 [VAL3] Implement shared random seed bias (both predict class 0) in collusion.py
- [ ] T030 [VAL3] Run collusion experiment and verify F > 0.6, low Judge accuracy (depends on T028-T029)
- [ ] T031 [VAL3] Document collusion detection results in experiments/neutrosophic_validation/collusion_results.md

**Checkpoint**: Collusion detection validated

---

## Phase 6: Validation Experiment 4 - Gaming Detection (Priority: Required)

**Goal**: Verify neutrosophic metrics detect gaming (Witness learns real data, ignores Weaver → F > 0.5)

**Independent Test**: Train Witness on real MNIST while Weaver generates noise, verify F > 0.5

### Implementation for VAL4

- [ ] T032 [P] [VAL4] Create gaming induction script in experiments/neutrosophic_validation/gaming.py
- [ ] T033 [VAL4] Implement Witness training on real data only (ignore Weaver) in gaming.py
- [ ] T034 [VAL4] Implement Weaver generating random noise in gaming.py
- [ ] T035 [VAL4] Run gaming experiment and verify F > 0.5 due to high witness_real_acc vs witness_gen_acc gap (depends on T032-T034)
- [ ] T036 [VAL4] Document gaming detection results in experiments/neutrosophic_validation/gaming_results.md

**Checkpoint**: All four validation experiments pass - neutrosophic metrics validated

---

## Phase 7: Visualization & Usability (Priority: Should Have)

**Goal**: Make neutrosophic metrics easy to interpret and visualize

**Independent Test**: Generate T/I/F trajectory plot for healthy baseline, visually confirm T↑, I↓, F low

### Implementation for VIZ

- [ ] T037 [P] [VIZ] Create plot_neutrosophic_trajectories() function in src/utils/visualization.py
- [ ] T038 [P] [VIZ] Add matplotlib/seaborn visualization for T/I/F over time in src/utils/visualization.py
- [ ] T039 [VIZ] Add trajectory plotting to validation experiment scripts (update healthy_baseline.py, mode_collapse.py, collusion.py, gaming.py) (depends on T037-T038)
- [ ] T040 [VIZ] Generate example plots for all four validation experiments and save to experiments/neutrosophic_validation/plots/

**Checkpoint**: Visualization complete - neutrosophic trajectories easily interpretable

---

## Phase 8: Documentation & Polish (Priority: Should Have)

**Purpose**: Ensure future instances can use neutrosophic metrics without rederiving implementation

- [ ] T041 [P] Update quickstart.md with actual code examples from integration (reference src/training/gpn_trainer.py lines)
- [ ] T042 [P] Create interpretation guide in docs/neutrosophic_metrics_interpretation.md (what do different T/I/F patterns mean?)
- [ ] T043 [P] Add docstrings to all NeutrosophicTracker methods following Google style guide
- [ ] T044 Validate that all four validation experiments are reproducible (run with fixed seed, verify same results)
- [ ] T045 [P] Update plan.md with final implementation notes and any deviations from original design
- [ ] T046 Run quickstart.md validation steps and verify instructions work as written

### Deferred Features (Out of MVP Scope)

The following "Could Have" / "Should Have" items from spec.md are explicitly **not** included in this task list:

- **Ablation study** (spec.md Should Have): Compare neutrosophic tracking vs scalar metrics on detecting failures. Deferred until after basic validation proves neutrosophic metrics work.
- **Adaptive training** (spec.md Could Have): Using neutrosophic state to adjust training (e.g., increase grounding weight when F high). Deferred to future iteration—current focus is measurement, not intervention.
- **Multi-dimensional neutrosophic tracking** (spec.md Could Have): Separate T/I/F for different aspects (classification, diversity, alignment). Deferred—unified T/I/F is sufficient for initial validation.

**Rationale for deferral**: MVP proves the core hypothesis (neutrosophic metrics distinguish genuine learning from gaming). These enhancements add value but aren't needed to validate the fundamental concept.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all validation experiments
- **Validation Experiments (Phases 3-6)**: All depend on Foundational phase completion
  - VAL1-VAL4 can then proceed in parallel (if staffed)
  - Or sequentially (VAL1 → VAL2 → VAL3 → VAL4)
- **Visualization (Phase 7)**: Depends on at least VAL1 being complete (needs data to visualize)
- **Documentation (Phase 8)**: Depends on all validation experiments being complete

### Validation Experiment Dependencies

- **VAL1 (Healthy Baseline)**: Can start after Foundational - No dependencies on other experiments
- **VAL2 (Mode Collapse)**: Can start after Foundational - Independently testable
- **VAL3 (Collusion)**: Can start after Foundational - Independently testable
- **VAL4 (Gaming)**: Can start after Foundational - Independently testable

All four validation experiments are INDEPENDENT - they can run in parallel.

### Within Each Validation Experiment

- Integration tasks before experiment scripts
- Experiment execution before result documentation
- Results documentation completes the experiment

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational component implementations (T004-T010) marked [P] can run in parallel
- All four validation experiments (VAL1-VAL4) can run in parallel after Foundational completes
- All visualization tasks marked [P] can run in parallel
- All documentation tasks marked [P] can run in parallel

---

## Parallel Example: Foundational Phase

```bash
# Launch all component computation methods together:
Task: "Implement NeutrosophicComponents.compute_alignment()"
Task: "Implement NeutrosophicComponents.compute_judge_accuracy()"
Task: "Implement NeutrosophicComponents.compute_weaver_uncertainty()"
Task: "Implement NeutrosophicComponents.compute_witness_entropy()"
Task: "Implement NeutrosophicComponents.compute_collusion()"
Task: "Implement NeutrosophicComponents.compute_mode_collapse()"
Task: "Implement NeutrosophicComponents.compute_gaming()"
```

## Parallel Example: Validation Experiments

```bash
# Once Foundational is complete, all experiments can run in parallel:
Task: "Create healthy baseline validation script"
Task: "Create mode collapse induction script"
Task: "Create collusion induction script"
Task: "Create gaming induction script"
```

---

## Implementation Strategy

### MVP First (VAL1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all experiments)
3. Complete Phase 3: VAL1 (Healthy Baseline)
4. **STOP and VALIDATE**: Verify T↑, I↓, F low on GPN-1 training
5. If validation passes, neutrosophic tracking is proven useful

### Full Validation Suite

1. Complete Setup + Foundational → Core implementation ready
2. Run VAL1 (Healthy Baseline) → Verify expected pattern → Document
3. Run VAL2 (Mode Collapse) → Verify F increases → Document
4. Run VAL3 (Collusion) → Verify F detects aligned-but-wrong → Document
5. Run VAL4 (Gaming) → Verify F detects Witness ignoring Weaver → Document
6. All four experiments pass → Neutrosophic metrics validated against spec falsification criteria

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (critical path)
2. Once Foundational is done:
   - Developer A: VAL1 (Healthy Baseline) + integration
   - Developer B: VAL2 (Mode Collapse)
   - Developer C: VAL3 (Collusion)
   - Developer D: VAL4 (Gaming)
3. Experiments complete and validate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific validation experiment for traceability
- Each validation experiment is independently completable and testable
- Foundational phase is critical path - all experiments blocked until complete
- Success criteria: All four validation experiments detect expected T/I/F patterns
- Commit after each task or logical group
- Stop at any checkpoint to validate experiment independently
- Research decisions already made in research.md - implementation only

