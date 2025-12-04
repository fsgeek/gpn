# GPN-3: Judgment vs Gaming Experiment Design

**Status**: Design Phase (Not Yet Implemented)
**Date**: December 2024

## 1. The Core Question

**Does pedagogical training develop judgment (generalizable evaluation) or just pattern matching (metric gaming)?**

MNIST cannot answer this because:
- There's always a correct answer
- Gaming the metric IS the skill
- Pattern matching and judgment produce identical behavior

We need a domain where:
1. No uniquely correct answer exists
2. Evaluation criteria can shift
3. Pedagogical training and adversarial training make different predictions

## 2. The Judgment vs Gaming Distinction

### 2.1 Definitions

**Gaming** (Adversarial prediction):
- Learn to predict scores given by a fixed evaluator
- Optimize for matching known criteria
- Fails when criteria change
- Equivalent to "teaching to the test"

**Judgment** (Pedagogical understanding):
- Learn what makes something genuinely good
- Understand *why* criteria matter
- Transfers to novel evaluation schemes
- Equivalent to "understanding principles"

### 2.2 Operational Test

Train two systems on task X with evaluation E1.

Test both on:
- **E1 (known)**: Original evaluation criteria
- **E2 (novel)**: Different but related criteria
- **E3 (orthogonal)**: Unrelated criteria

**Prediction**:
- Gaming: High E1, low E2, random E3
- Judgment: High E1, high E2, random E3

## 3. Proposed Domain: Essay Evaluation

### 3.1 Why Essays?

**Pros**:
- No single correct answer
- Multiple valid evaluation frameworks
- Human judgment data exists
- Feedback → improvement structure testable
- Ethically lower stakes than medical/legal domains

**Cons**:
- Requires human subject data
- IRB approval likely needed
- Expensive to collect new data
- Existing datasets may lack revision histories

### 3.2 Architectural Design

**Two training conditions**:

**Adversarial Witness (Gaming)**:
- Input: Essay text
- Output: Predicted human grader score
- Loss: MSE(predicted_score, actual_score)
- Objective: Match human evaluations

**Pedagogical Witness (Judgment)**:
- Input: Essay text + context (assignment, prior version)
- Output: Feedback text + predicted improvement
- Loss: Improvement in subsequent revision quality
- Objective: Maximize student learning

**Weaver role**: Generates essay revisions based on Witness feedback

**Judge role**: Multiple evaluation frameworks:
- E1 (training): Original rubric scores
- E2 (transfer): Alternative rubric (e.g., argument strength vs grammar)
- E3 (orthogonal): Unrelated metric (e.g., readability vs creativity)

### 3.3 Evaluation Protocol

**Phase 1**: Train both Witnesses on historical data
- Adversarial: Essay → score pairs
- Pedagogical: Essay → feedback → revision → improvement

**Phase 2**: Generate feedback on held-out essays

**Phase 3**: Human evaluation (or proxy)
- Does feedback align with E1? (Both should pass)
- Does feedback transfer to E2? (Judgment should, gaming shouldn't)
- Does feedback help students? (Pedagogical goal)

### 3.4 Critical Challenge: Data Requirements

**Needed**:
1. Essay corpus with multiple revisions per essay
2. Grader scores across different rubrics
3. Causal structure: feedback → revision (not just correlation)

**Problem**: Most datasets have final essays + scores, not revision histories.

## 4. Alternative Domains

### 4.1 Code Review → Commit Quality

**Structure**:
- Input: Code diff
- Adversarial: Predict whether PR gets merged
- Pedagogical: Give feedback that improves subsequent code quality
- Evaluation: Bugs in production, code maintainability, team velocity

**Pros**:
- Data publicly available (GitHub)
- Causal structure clearer (review → revision → merge)
- Objective quality metrics exist (bugs, performance)
- No IRB required

**Cons**:
- Code quality is partially objective (some gaming may be legitimate)
- Feedback → improvement link noisy (many confounds)
- May require access to private repos for sensitive domains

**Data sources**:
- GitHub PR reviews and subsequent commits
- Code review comments + whether suggestions were adopted
- Static analysis metrics before/after review
- Bug tracking data linked to commits

### 4.2 Medical Diagnosis → Patient Outcomes

**Structure**:
- Input: Patient data
- Adversarial: Predict diagnosis given by attending physician
- Pedagogical: Recommend treatment that improves patient outcomes
- Evaluation: Recovery time, complications, quality of life

**Pros**:
- Clear ground truth (patient outcomes)
- High-stakes domain where judgment matters
- Existing medical datasets with longitudinal data

**Cons**:
- IRB required
- Ethical concerns (can't experiment with patient care)
- Requires medical expertise to evaluate
- Data access restricted

**Not recommended without medical collaborators**

### 4.3 Stack Overflow: Question Quality Improvement

**Structure**:
- Input: Initial question version
- Adversarial: Predict community upvotes
- Pedagogical: Give feedback that leads to better engagement
- Evaluation: Answer quality, problem resolution, community help

**Pros**:
- Public data with edit histories
- Clear feedback loops
- Multiple quality dimensions
- No IRB needed

**Cons**:
- Upvotes are gameable (already a known problem)
- Edit reasons not always feedback-driven
- Community norms shift over time

**Data sources**:
- Stack Overflow data dump
- Edit histories for questions
- Comment threads (feedback)
- Answer quality and acceptance

### 4.4 Creative Writing: Story Feedback

**Structure**:
- Input: Story draft
- Adversarial: Predict writing contest scores
- Pedagogical: Give feedback that improves narrative quality
- Evaluation: Reader engagement, emotional impact, craft improvement

**Pros**:
- Clearly subjective domain
- Multiple valid evaluation frameworks
- Feedback culture exists (writing workshops)

**Cons**:
- No large public dataset with revision histories
- Would need to collect new data
- "Quality" highly subjective
- Time-intensive to evaluate

## 5. Recommended Immediate Path

### 5.1 Phase 1: Code Review Proof of Concept

**Why start here**:
- Publicly available data (no IRB)
- Causal structure (review → commit → merge)
- Objective quality proxies (bugs, tests, static analysis)
- Can complete without human subjects

**Dataset**: GitHub PRs with:
1. Initial commit
2. Review comments
3. Subsequent commits
4. Merge decision + production outcomes

**Minimal viable experiment**:
- Train adversarial Witness: code → merge prediction
- Train pedagogical Witness: code → feedback → improved code
- Test: Does pedagogical feedback transfer to code quality metrics not in training?

**Success criteria**:
- Both predict merge similarly (E1)
- Pedagogical does better on bug prevention (E2)
- Pedagogical feedback leads to more maintainable code (E3)

### 5.2 Phase 2: Essay Feedback (if Phase 1 succeeds)

**Why wait**:
- Requires IRB approval
- More expensive to evaluate
- Higher ethical stakes

**But more aligned with core GPN thesis**:
- Essays are inherently judgment-based
- No "correct" answer
- Multiple legitimate evaluation frameworks

## 6. Technical Implementation Notes

### 6.1 Architecture Adaptations

**Current GPN**:
- Image generation (Weaver outputs pixels)
- Classification (Witness outputs labels)

**GPN-3 (Judgment)**:
- Text generation (Weaver outputs text/code)
- Evaluation + feedback (Witness outputs scores + explanation)

**Required changes**:
- Replace image encoder/decoder with text transformer
- Add feedback generation head to Witness
- Multi-objective loss (score + improvement)

### 6.2 New Loss Components

**Pedagogical Witness**:
```
L_pedagogy = λ_improve * L_improvement + λ_explain * L_explanation

Where:
- L_improvement: Does output actually get better after feedback?
- L_explanation: Is feedback coherent and actionable?
```

**Adversarial Witness** (baseline):
```
L_adversarial = MSE(predicted_score, actual_score)
```

### 6.3 Evaluation Metrics

**E1 (Training criteria)**:
- Score prediction accuracy
- Feedback alignment with training rubric

**E2 (Transfer criteria)**:
- Different rubric (e.g., grammar → argumentation)
- Novel quality dimensions
- Community standards from different time period

**E3 (Improvement)**:
- Does feedback actually help?
- Measured by improvement in subsequent version
- Causal test: feedback → revision → better outcome

## 7. Open Questions

### 7.1 What counts as "novel criteria"?

**Spectrum**:
- **Near transfer**: Different weights on same dimensions (70% grammar / 30% content → 30% grammar / 70% content)
- **Medium transfer**: Different but related dimensions (clarity → persuasiveness)
- **Far transfer**: Unrelated dimensions (technical accuracy → emotional impact)

**Question**: Where does gaming break down? Where does judgment persist?

### 7.2 Can judgment be learned without explicit teaching?

**Hypothesis**: Pedagogical training develops judgment implicitly through the improvement objective.

**Alternative**: Maybe judgment requires explicit multi-criteria training.

**Test**: Compare:
- Single-criterion pedagogical (feedback optimizes one metric)
- Multi-criterion pedagogical (feedback optimizes multiple metrics)
- Transfer to unseen criteria

### 7.3 Does curriculum help for judgment?

**GPN-2 showed**: Curriculum essential for composition

**Question**: Is judgment compositional? Do you need to learn atomic evaluation skills before complex judgment?

**Possible curriculum**:
1. Single-dimension feedback (grammar only)
2. Two-dimension feedback (grammar + clarity)
3. Holistic judgment (overall quality)

## 8. Next Steps (Not Implementation—Design Only)

1. **Survey code review datasets**:
   - Size, quality, completeness
   - Are feedback loops present?
   - Do we have outcome data?

2. **Survey essay datasets**:
   - Revision histories available?
   - Multiple rubrics?
   - Feedback comments present?

3. **Prototype evaluation harness**:
   - How do we measure "transfer to novel criteria"?
   - What's the test protocol?
   - How do we control for confounds?

4. **Estimate resource requirements**:
   - Compute (likely needs GPU)
   - Data collection costs
   - Human evaluation if needed
   - Timeline

5. **Identify collaborators**:
   - Domain experts (education, SE, writing)
   - IRB expertise if needed
   - Dataset access

---

**This document is a design spec, not an implementation plan.** We're scoping the question before committing resources. The goal is to understand what judgment vs gaming would look like operationally, and what's actually feasible without large-scale human subject research.

**Critical decision point**: Can we test judgment without IRB approval? Code review says yes. Essays say no (probably). That determines the immediate path.

