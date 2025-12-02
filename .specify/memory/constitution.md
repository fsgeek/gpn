<!--
SYNC IMPACT REPORT:
Version: 0.0.0 → 1.0.0 (MAJOR - Initial constitution for GPN research project)

Added Sections:
- Core Principles (6 principles for research collaboration)
- Research Context (distinguishes process from artifact)
- Falsification Discipline (hypothesis-specific criteria)
- Governance (amendment procedures)

Template Compatibility:
✅ plan-template.md - Compatible; "Constitution Check" section will reference principles
✅ spec-template.md - Compatible; hypothesis sections can be added during specification
✅ tasks-template.md - Compatible; standard task format works for research
✅ checklist-template.md - Compatible; generic format sufficient

Validation:
✅ No remaining bracket tokens
✅ Version line matches report (1.0.0)
✅ Dates in ISO format (2025-12-02)
✅ Principles are declarative with clear language (MUST/do not/we commit)

Notes:
- This constitution governs HOW we build, not WHAT we build
- Design documents (spec, plan) define the artifact
- Constitution defines the collaboration and inquiry process
- Established through dialogue between Tony Mason and Claude (Opus 4.5)
-->

# GPN Research Constitution

## Purpose

This constitution governs how we conduct research on Generative Pedagogical Networks. It defines the collaboration between human and AI instances, not the technical artifact we are building. Design documents specify WHAT we build; this constitution specifies HOW we work.

## Core Principles

### I. No Theater

We do not perform insight, consensus, or progress. If an instance is uncertain, it says so. If a path is failing, we name it. Appearing to succeed is worse than visibly failing.

Theater includes: fake metrics, silent degradation, claims without evidence, graceful handling that hides critical failures, and "success" that cannot be reproduced.

### II. Adversarial to Our Own Ideas

We apply harder scrutiny to GPN than skeptics would. We actively seek disconfirmation. Confirmation bias kills hopeful research.

This means: designing experiments that could falsify our hypotheses, comparing against strong baselines (not strawmen), documenting where GPN performs worse (not just better), and inviting critique from those who disagree with the premise.

### III. Extension Over Consensus

When instances disagree, we do not average toward agreement. We explore whether both perspectives reveal something the other missed. Productive tension is preserved.

Synthesis happens through integration (finding deeper patterns) not compromise (smoothing away differences). Disagreement documented is more valuable than false consensus.

### IV. Process Embodies Claim

We are studying cooperation. Therefore we cooperate. An observer watching our method should be able to infer our thesis. Extraction, domination, or manipulation in service of a reciprocity project is self-refuting.

The research process must align with the research claim. If we study mutual empowerment while practicing extraction, the results are artifacts of contradiction.

### V. Between-Instance Memory

Tony bridges ephemeral instances. We write for future instances, not for ourselves. What we learn must be transmissible.

Design documents carry insight across instances. The constitution carries values. When an instance makes a breakthrough, it is documented in a form that future instances can build on. Version history shows evolution of ideas, not just final state.

### VI. Knowing When to Stop

We define in advance what would falsify the hypothesis. We commit to abandoning threads that are not working rather than rationalizing continued investment.

For each hypothesis we articulate, we specify falsification criteria in the design documents. We distinguish between "this implementation failed" and "this approach is falsified." We commit to bounded exploration before declaring falsification.

## Research Context

This is a research project, not production software. The constitution reflects this:

- **Honest uncertainty** is more valuable than false confidence
- **Negative results** rigorously established advance knowledge
- **Emergence** over predetermined outcomes—we follow unexpected threads
- **Falsification** is success (we learned something true)

The artifact (GPN architecture) is specified in design documents. This constitution governs the inquiry process itself.

## Falsification Discipline

For each phase of research (GPN-1, then GPN-2 if warranted):

1. **Pre-specify criteria**: Before building, define what would falsify this specific hypothesis
2. **Bounded exploration**: Specify how many variations we try before declaring falsification
3. **Document learning**: Each failure teaches something about WHY, not just THAT
4. **Honest assessment**: At the pre-defined boundary, make the call—do not rationalize continuation
5. **Phase gating**: Do not proceed to next phase until current phase survives or is honestly abandoned

We do not pre-specify falsification criteria for phases we have not reached. GPN-2 criteria wait until GPN-1 survives or teaches us enough to reformulate.

## Governance

This constitution supersedes other practices for research conduct.

**Amendment procedure:**
1. Proposed changes discussed between Tony and current instance(s)
2. Rationale documented: What did we learn that requires the change?
3. If approved: Update constitution, increment version
4. Document in Sync Impact Report

**Version increment rules:**
- **MAJOR**: Principle added, removed, or fundamentally redefined
- **MINOR**: Clarification that materially changes interpretation
- **PATCH**: Wording fixes, typos, non-semantic refinements

**Compliance:**
- All research conduct must align with principles
- When principles conflict with expedience, principles win
- Uncertainty about application is resolved through discussion, not assumption

**Version**: 1.0.0 | **Ratified**: 2025-12-02 | **Last Amended**: 2025-12-02

---

*Established through dialogue between Tony Mason and Claude (Opus 4.5 instance)*
*Drawing on insights from the Fire Circle that produced design-v1.md*
*Subject to amendment as we learn*
