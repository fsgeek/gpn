# Feature Specification: GPN-1 MNIST Proof-of-Concept

**Feature Branch**: `001-gpn1-mnist-poc`
**Created**: 2025-12-02
**Status**: Draft
**Input**: GPN-1: MNIST proof-of-concept with Weaver/Witness architecture implementing mutual empowerment objectives and three-phase training curriculum

## Research Context

This is a research experiment, not production software. The "users" are researchers testing whether pedagogical training dynamics differ from adversarial dynamics. Success includes informative failure—learning why something doesn't work is a valid outcome.

**Core Research Question**: Does architecturally-encoded cooperation produce qualitatively different training dynamics than adversarial framing?

**Philosophical Foundation**: Ayni reciprocity—value measured by investment, not extraction. The Weaver's objective is the Witness's growth, not fooling the Witness.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Basic Training Viability (Priority: P1)

As a researcher, I want to run GPN-1 training on MNIST and observe that it converges without catastrophic failure, so that I can verify the architecture is implementable and stable enough to test hypotheses.

**Why this priority**: Without basic viability, no other experiments are possible. This is the foundation.

**Independent Test**: Run training for 12,000 steps. If training completes without NaN losses, gradient explosions, or mode collapse to single digit, the architecture is viable for further experimentation.

**Acceptance Scenarios**:

1. **Given** initialized Weaver, Witness, and Judge networks, **When** training runs for 5,000 steps (Phase 1), **Then** all loss components remain finite and gradients flow to both Weaver and Witness
2. **Given** Phase 1 completed successfully, **When** training continues through Phase 2 (steps 5,000-10,000), **Then** generated images are recognizable as digits and loss values stabilize
3. **Given** Phase 2 completed, **When** Phase 3 begins (Judge removed), **Then** training continues without immediate collapse

---

### User Story 2 - Baseline Comparison (Priority: P2)

As a researcher, I want to compare GPN-1 against a standard GAN baseline on the same task, so that I can determine whether pedagogical framing produces measurably different results.

**Why this priority**: The core hypothesis requires comparison. Without a baseline, we cannot distinguish "GPN works" from "any generative model works on MNIST."

**Independent Test**: Train both GPN-1 and a standard GAN with equivalent architecture capacity on MNIST. Compare convergence speed, final quality, and mode diversity.

**Acceptance Scenarios**:

1. **Given** GPN-1 and GAN baseline with matched capacity, **When** both are trained on MNIST, **Then** convergence curves can be directly compared
2. **Given** completed training runs for both architectures, **When** mode diversity is measured, **Then** the number of distinct digit classes represented in generated samples is quantifiable
3. **Given** both models trained to completion, **When** quality metrics are computed, **Then** results can be compared with statistical significance (minimum 3 runs each)

---

### User Story 3 - Phase 3 Drift Test (Priority: P3)

As a researcher, I want to observe what happens when external grounding (Judge) is removed in Phase 3, so that I can determine whether mutual empowerment has been internalized or requires ongoing external scaffolding.

**Why this priority**: This is the critical experiment that distinguishes GPN from "GAN with extra losses." Phase 3 tests whether the relationship holds without external enforcement.

**Independent Test**: After Phase 2 completion, run Phase 3 for minimum 2,000 steps with Judge completely removed. Measure whether output quality persists or degrades.

**Acceptance Scenarios**:

1. **Given** successful Phase 2 completion, **When** Judge is removed and training continues, **Then** the change in image quality over Phase 3 is measurable
2. **Given** Phase 3 running, **When** 2,000 steps complete, **Then** either quality persists (relationship internalized) or quality degrades (relationship requires external grounding)
3. **Given** Phase 3 completion, **When** Weaver/Witness alignment loss is measured, **Then** we can determine if agents drifted into "private language" or maintained grounded communication

---

### Edge Cases

- What happens if Weaver collapses to generating only one digit class? (Mode collapse declared when fewer than 5 classes at >5% each; early warning when any class exceeds 50%)
- What happens if Witness stops learning and EMA state stagnates? (Stagnation detection required)
- What happens if alignment loss goes to zero too quickly? (Potential collusion signal)
- How does the system behave with different random seeds? (Reproducibility verification)
- What if Phase 1 doesn't establish sufficient grounding before Phase 2 transition?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST implement Weaver network that generates images and outputs attribute predictions (v_pred)
- **FR-002**: System MUST implement Witness network that classifies images and estimates attributes from observation (v_seen)
- **FR-003**: System MUST implement frozen Judge network (pre-trained MNIST classifier) for Phase 1-2 grounding
- **FR-004**: System MUST compute Grounding Loss (Judge's classification of Weaver output)
- **FR-005**: System MUST compute Alignment Loss as MSE between v_pred and v_seen
- **FR-006**: System MUST compute Empowerment Loss using Goldilocks KL divergence tracking both mean and variance
- **FR-007**: System MUST implement three-phase training curriculum with configurable step boundaries
- **FR-008**: System MUST log all loss components at each training step before any aggregation
- **FR-009**: System MUST save checkpoints at configurable intervals with full state for reproducibility
- **FR-010**: System MUST implement EMA state tracking for Witness with updates only on Witness forward pass
- **FR-011**: System MUST implement alternating updates (Witness first with Weaver detached, then Weaver with fresh forward pass)
- **FR-012**: System MUST completely remove Judge from computational graph in Phase 3 (not just weight to zero)
- **FR-013**: System MUST generate sample images at configurable intervals for visual inspection
- **FR-014**: System MUST compute mode diversity metrics (coverage of digit classes in generated samples)
- **FR-015**: System MUST implement equivalent GAN baseline with matched architecture capacity for comparison
- **FR-016**: System MUST use batch size of 64 for both GPN-1 and baseline experiments (configurable)

### Key Entities

- **Weaver**: Generator network; produces images from latent vectors; outputs 16-dimensional attribute prediction vector (v_pred, configurable); objective is Witness's growth
- **Witness**: Discriminator-analog; classifies images; estimates 16-dimensional attributes from observation (v_seen, configurable); learns through pedagogical relationship with Weaver
- **Judge**: External frozen CNN classifier trained fresh on MNIST (>95% accuracy); provides reality anchor in Phase 1-2; completely removed in Phase 3; training script and checkpoint included in repo
- **Training Run**: Complete experiment from initialization through Phase 3; includes configuration, all checkpoints, all logged metrics
- **Experiment**: Collection of related training runs (e.g., GPN vs baseline comparison); includes analysis and conclusions

## Falsification Criteria *(research-specific)*

Per the constitution's "Knowing When to Stop" principle, we define what would falsify the GPN-1 hypothesis:

**GPN-1 is falsified if ALL of the following are true after bounded exploration (5 hyperparameter variations):**

1. GPN-1 converges slower than GAN baseline (measured by steps to 90% digit recognizability)
2. GPN-1 shows equal or worse mode diversity than GAN baseline (measured by digit class coverage)
3. Phase 3 shows >20% quality degradation within 1,000 steps of Judge removal

**GPN-1 is NOT falsified by:**

- A single failed run (might be hyperparameter issue)
- Phase 3 showing some drift (partial internalization is still informative)
- Equivalent-to-baseline performance (null result, not falsification of broader approach)

**Informative outcomes that are NOT failure:**

- Phase 3 drifts predictably → reveals what external grounding provides
- GPN-1 slower but more stable → different tradeoff, worth understanding
- Mode diversity different in character (not just quantity) → qualitative insight

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Training completes all three phases (12,000+ steps) without NaN losses or gradient explosions in at least 3 of 5 runs
- **SC-002**: Generated images at end of Phase 2 are recognizable as digits (human inspection primary; Judge classification >80% accuracy on generated samples as objective proxy)
- **SC-003**: Mode diversity at end of Phase 2 covers at least 8 of 10 digit classes with >5% representation each
- **SC-004**: Convergence speed comparison between GPN-1 and GAN baseline is quantified with statistical significance (p < 0.05 or documented as inconclusive)
- **SC-005**: Phase 3 behavior is documented with metrics: either quality persists (relationship internalized) or degradation pattern is characterized
- **SC-006**: All experiments are reproducible: same random seed produces same training trajectory within numerical precision
- **SC-007**: Results (positive, negative, or inconclusive) are documented with sufficient detail for independent replication

### Research Success (distinct from technical success)

- We learn something true about whether pedagogical framing affects training dynamics
- Negative results are documented with same rigor as positive results
- The experiment informs whether GPN-2/GPN-3 are worth pursuing

## Clarifications

### Session 2025-12-02

- Q: What is the attribute vector dimension for v_pred/v_seen? → A: 16 dimensions (configurable for future experiments)
- Q: What threshold defines mode collapse? → A: Fewer than 5 of 10 digit classes at >5% representation each (primary); any single class exceeding 50% of samples (early warning metric)
- Q: What batch size for training? → A: 64 (standard MNIST GAN batch size)
- Q: What is the source for the pre-trained Judge? → A: Train fresh CNN on MNIST, freeze weights, include training script and checkpoint in repo for full reproducibility
- Q: How to measure "recognizable digits" objectively? → A: Human inspection primary; Judge classification accuracy >80% on generated samples as objective reproducibility proxy

## Assumptions

- MNIST is sufficient complexity to reveal differences between adversarial and pedagogical training (if differences exist)
- Standard CNN architectures are appropriate for Weaver/Witness/Judge networks
- 12,000 training steps is sufficient to observe Phase 3 dynamics
- Three random seeds per condition provides adequate signal for comparison (may need more if variance is high)
- Visual inspection by researchers is primary for "recognizable digit" assessment, with Judge classification >80% as objective reproducibility proxy
- The attribute vector dimension of 16 (v_pred, v_seen) is adequate for MNIST; may need adjustment for more complex datasets in future experiments

## Future Research Questions

*Captured for GPN-1.5 or later — not in scope for GPN-1:*

- **Generalization vs Judge-specific learning**: Phase 3 quality is measured using the same Judge that was removed from training. This tests whether Weaver internalized the grounding, but doesn't distinguish between "internalized general digit quality" and "internalized this-specific-Judge's decision boundary." A follow-on experiment could evaluate Phase 3 outputs with a *different* classifier to test whether quality generalizes beyond the training Judge.

*Observation contributed by reviewer instance (2025-12-02)*
