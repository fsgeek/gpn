# Dataset Survey: Feedback → Improvement Structures

**Purpose**: Identify existing datasets with feedback loops suitable for testing judgment vs gaming in GPN-3.

**Requirements**:
1. Input → Feedback → Improved Output structure
2. Multiple evaluation criteria (for testing transfer)
3. Publicly accessible (no IRB)
4. Large enough for training (>10k examples)

## 1. Code Review Datasets

### 1.1 GitHub Pull Request Reviews

**Description**: Code changes + review comments + subsequent commits

**Sources**:
- **GH Archive**: Full GitHub event stream (public)
- **GHTorrent**: Curated GitHub data dump
- **CodeReview dataset** (Microsoft Research)

**Structure**:
```
PR submission → Review comments → Revised commits → Merge decision
```

**Evaluation dimensions**:
- E1 (training): Merge acceptance
- E2 (transfer): Static analysis metrics (complexity, bugs)
- E3 (transfer): Test coverage changes
- E4 (transfer): Documentation quality

**Pros**:
- ✓ Large-scale (millions of PRs)
- ✓ Public, no IRB
- ✓ Clear causal structure
- ✓ Objective quality metrics exist
- ✓ Temporal data (can track improvement)

**Cons**:
- Merge decision is multi-factor (not just code quality)
- Review quality varies wildly
- May need to filter for "good" repos
- Some feedback is social/procedural, not technical

**Data access**:
- GH Archive: https://www.gharchive.org/
- GHTorrent: http://ghtorrent.org/
- BigQuery GitHub public dataset

**Estimated size**: 10M+ PRs, but need filtering for quality

**Verdict**: **Strong candidate** - clear feedback loops, objective metrics, no IRB

### 1.2 Stack Overflow Edits

**Description**: Questions + comments + edit history + answer quality

**Structure**:
```
Initial question → Comments (feedback) → Edited question → Answer quality
```

**Evaluation dimensions**:
- E1 (training): Upvotes / answer acceptance
- E2 (transfer): Time to answer
- E3 (transfer): Answer thoroughness
- E4 (transfer): Similar question linking

**Pros**:
- ✓ Public data dump available
- ✓ Large-scale (millions of questions)
- ✓ Edit histories preserved
- ✓ Multiple quality signals
- ✓ No IRB needed

**Cons**:
- Upvotes are gameable (known problem)
- Comments not always improvement-focused
- Community norms shift over time
- Not all edits are feedback-driven

**Data access**:
- Stack Exchange Data Dump: https://archive.org/details/stackexchange
- Kaggle Stack Overflow datasets
- BigQuery Stack Overflow public dataset

**Estimated size**: 20M+ questions with edit histories

**Verdict**: **Moderate candidate** - gaming is already a problem, but large-scale and accessible

## 2. Writing/Essay Datasets

### 2.1 Writing Prompts (Reddit)

**Description**: Creative writing prompts + stories + community feedback

**Structure**:
```
Prompt → Initial story → Comments → Revised story (rare)
```

**Evaluation dimensions**:
- E1 (training): Upvotes
- E2 (transfer): Comment sentiment
- E3 (transfer): Narrative structure analysis
- E4 (transfer): Engagement (replies)

**Pros**:
- ✓ Public (Reddit API)
- ✓ Large volume
- ✓ Subjective domain (good for judgment test)

**Cons**:
- Revisions are rare (most stories are one-shot)
- Upvotes don't measure improvement
- Feedback often vague ("great job!")
- No ground truth quality

**Data access**:
- Reddit API
- Pushshift Reddit dumps
- Academic datasets (r/WritingPrompts)

**Estimated size**: 100k+ stories, but <1% have revisions

**Verdict**: **Weak candidate** - insufficient revision data

### 2.2 Peer Review Essays (Academic)

**Description**: Student essays + peer feedback + revisions

**Datasets**:
- **ASAP (Automated Student Assessment Prize)**: Kaggle, essays with scores
- **Feedback Prize** (Kaggle 2021): Essays with feedback effectiveness labels
- **ICLE (International Corpus of Learner English)**: Non-native essays with annotations

**Structure**:
```
Essay draft → Feedback → Revised essay → Grade
```

**Evaluation dimensions**:
- E1 (training): Rubric scores
- E2 (transfer): Different rubric dimensions
- E3 (transfer): Improvement magnitude

**Pros**:
- ✓ Designed for assessment research
- ✓ Multiple rubric dimensions
- ✓ Expert annotations
- ✓ Some revision data

**Cons**:
- Limited size (1k-10k essays)
- May require institutional access
- IRB unclear (depends on dataset)
- Not all have revision histories

**Data access**:
- ASAP: https://www.kaggle.com/c/asap-aes
- Feedback Prize: https://www.kaggle.com/c/feedback-prize-2021
- ICLE: Requires license

**Estimated size**: 1-10k essays per dataset

**Verdict**: **Moderate candidate** - designed for this, but small and may need IRB

### 2.3 CommonLit Readability (Kaggle)

**Description**: Text passages with readability scores + comprehension data

**Structure**:
```
Text → Readability score + Comprehension difficulty
```

**Evaluation dimensions**:
- E1 (training): Target grade level
- E2 (transfer): Comprehension scores
- E3 (transfer): Engagement metrics

**Pros**:
- ✓ Public (Kaggle)
- ✓ Multiple quality dimensions
- ✓ Educational domain

**Cons**:
- NO feedback loop (just scoring)
- NO revisions
- Static assessment only

**Verdict**: **Not suitable** - no feedback/improvement structure

## 3. Conversational Datasets

### 3.1 Reddit CMV (Change My View)

**Description**: Arguments + counterarguments + delta awards (view changes)

**Structure**:
```
Initial view → Counterarguments → OP responses → Delta award (if convinced)
```

**Evaluation dimensions**:
- E1 (training): Delta awards
- E2 (transfer): Argument quality (structure)
- E3 (transfer): Evidence use
- E4 (transfer): Civility / persuasiveness

**Pros**:
- ✓ Public (Reddit)
- ✓ Large-scale (10k+ threads)
- ✓ Clear success metric (delta = view changed)
- ✓ Judgment-heavy domain

**Cons**:
- Not exactly feedback → revision (more debate)
- Delta awards are sparse
- Persuasiveness is context-dependent
- Not traditional pedagogical structure

**Data access**:
- Reddit API
- Pushshift dumps
- r/changemyview archives

**Estimated size**: 10k+ threads with deltas

**Verdict**: **Interesting alternative** - not standard pedagogy, but tests persuasion judgment

### 3.2 Ubuntu Dialogue Corpus

**Description**: Technical support conversations

**Structure**:
```
Question → Attempted answer → Clarification → Better answer → Problem solved
```

**Evaluation dimensions**:
- E1 (training): Problem resolution (explicit marker)
- E2 (transfer): Response quality
- E3 (transfer): Efficiency (turns to solution)

**Pros**:
- ✓ Public
- ✓ Large-scale (1M+ dialogues)
- ✓ Clear success metric
- ✓ Iterative refinement structure

**Cons**:
- Technical domain (limited)
- Not always clear improvement
- Solution quality varies
- May need domain knowledge

**Data access**:
- http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/

**Estimated size**: 1M+ dialogues

**Verdict**: **Moderate candidate** - clear structure, but narrow domain

## 4. Educational Datasets

### 4.1 MathDial (Math Tutoring)

**Description**: Student-tutor dialogues solving math problems

**Structure**:
```
Problem → Student attempt → Tutor feedback → Improved attempt → Solution
```

**Evaluation dimensions**:
- E1 (training): Solution correctness
- E2 (transfer): Efficiency (steps needed)
- E3 (transfer): Explanation quality
- E4 (transfer): Conceptual understanding

**Pros**:
- ✓ Clear pedagogical structure
- ✓ Feedback → improvement visible
- ✓ Multiple quality dimensions
- ✓ Ground truth (correct answer)

**Cons**:
- Math has "correct" answers (less judgment)
- Small dataset (1k dialogues)
- May need IRB depending on source
- Limited domain

**Data access**:
- Research paper datasets (various)
- May require request to authors

**Estimated size**: 1-5k dialogues

**Verdict**: **Weak candidate** - too small, too objective

### 4.2 Quizlet Study Sets

**Description**: Flashcards + study sessions + performance

**Structure**:
```
Flashcard set → Study session → Performance → Adjusted set
```

**Evaluation dimensions**:
- E1 (training): Test performance
- E2 (transfer): Long-term retention
- E3 (transfer): Transfer to related concepts

**Pros**:
- ✓ Large-scale (millions of sets)
- ✓ Performance tracking
- ✓ Multiple domains

**Cons**:
- NO explicit feedback text
- Performance is individual-dependent
- Privacy concerns (user data)
- May require Quizlet partnership

**Verdict**: **Not suitable** - no feedback text, privacy issues

## 5. Recommended Dataset: GitHub PR Reviews

### 5.1 Why GitHub?

**Best fit for requirements**:
1. ✓ Feedback → improvement structure (review → revised commit)
2. ✓ Multiple evaluation criteria (merge, bugs, tests, complexity)
3. ✓ Publicly accessible, no IRB
4. ✓ Large-scale (millions of examples)
5. ✓ Objective quality metrics available

**Additional benefits**:
- Temporal structure (can track long-term code quality)
- Diverse feedback types (style, correctness, design)
- Community norms (different repos have different standards)
- Can filter for quality (repo stars, contributor reputation)

### 5.2 Proposed Data Pipeline

**Step 1: Filter repos**
- Stars > 100 (quality signal)
- Active development (commits in last year)
- Merged PRs > 50 (enough data)

**Step 2: Extract PR data**
- Initial commit diff
- Review comments (feedback)
- Subsequent commit diffs (revisions)
- Merge decision (binary outcome)
- Post-merge metrics (bugs, reverts)

**Step 3: Augment with static analysis**
- Run pylint / eslint / etc on code
- Compute complexity metrics
- Measure test coverage
- Check documentation completeness

**Step 4: Create splits**
- Training: Repos 1-1000
- Transfer (same language): Repos 1001-1200
- Transfer (different norms): Different language or domain

### 5.3 Example Experiment

**Training**:
- Adversarial Witness: Code → Merge prediction
- Pedagogical Witness: Code → Feedback → Improved code quality

**Evaluation**:
- **E1 (training)**: Merge prediction accuracy
- **E2 (transfer)**: Bug prediction on post-merge code
- **E3 (transfer)**: Code maintainability scores
- **E4 (transfer)**: Test coverage improvement

**Hypothesis**:
- Both predict merge similarly (E1: ~80%)
- Pedagogical better at bug prediction (E2: +10%)
- Pedagogical better at maintainability (E3: +15%)
- Pedagogical feedback increases test coverage (E4: +5%)

### 5.4 Data Access Plan

**Sources**:
1. **GHTorrent**: Historical data dump
2. **GitHub API**: Recent PRs
3. **BigQuery**: GitHub public dataset

**Tools**:
- PyGithub (Python API wrapper)
- PyDriller (git analysis)
- Static analysis: pylint, radon, coverage.py

**Storage estimate**:
- 10k PRs × 1MB per PR = 10GB
- Manageable on single machine

## 6. Alternative: Stack Overflow (Secondary)

### 6.1 Why SO as backup?

**If GitHub proves insufficient**:
- Easier to parse (structured Q&A)
- Clear quality signals (votes, acceptance)
- Multiple domains (not just code)
- Established research community

### 6.2 Proposed pipeline

**Step 1: Filter questions**
- Has edit history
- Has comments before edit
- Has accepted answer
- Score > 0

**Step 2: Extract structure**
- Initial question version
- Comments (presumed feedback)
- Edited question version
- Answer quality (votes, acceptance)

**Step 3: Create evaluation splits**
- E1 (training): Answer acceptance
- E2 (transfer): Time to answer
- E3 (transfer): Answer thoroughness

**Issue**: Gaming is already widespread. May not distinguish judgment from gaming.

## 7. Not Recommended

### 7.1 Student Essays
- Too small (<10k)
- IRB concerns
- Limited public availability

### 7.2 Medical Records
- IRB required
- Privacy concerns
- Needs domain experts

### 7.3 Creative Writing
- Insufficient revision data
- Purely subjective
- No ground truth

## 8. Next Steps

1. **Start with GitHub PR data**
   - Download sample (100 repos)
   - Verify feedback → revision structure
   - Test static analysis pipeline
   - Estimate data quality

2. **Build data processing pipeline**
   - Parse PR diffs
   - Extract review comments
   - Link comments to revisions
   - Compute quality metrics

3. **Create train/test splits**
   - By repo (avoid data leakage)
   - By time (test generalization)
   - By language (test transfer)

4. **Prototype evaluation harness**
   - How to measure "judgment"?
   - Baseline: random feedback
   - Upper bound: human expert feedback

5. **Estimate compute requirements**
   - Data processing: CPU-bound
   - Training: GPU (likely 4090)
   - Evaluation: Can be parallelized

---

**Recommendation**: **Proceed with GitHub PR review dataset** as primary target. Public, large-scale, no IRB, clear feedback loops, objective quality metrics.

**Backup**: Stack Overflow if GitHub proves too noisy.

**Do not pursue**: Essays (IRB + small), medical (IRB + expertise), creative writing (no revisions).

