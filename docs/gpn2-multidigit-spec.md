# GPN-2: Multi-Digit Number Generation

## Hypothesis

Curriculum-based training (single digits → composition) produces better multi-digit generators than training on multi-digit numbers from scratch.

This tests whether pedagogical structure matters when the task has compositional structure.

## Task Definition

Generate images of 2-digit or 3-digit numbers (e.g., "372") given:
- The target number as a label
- A latent vector for variation

### Success Criteria

1. **Atomic accuracy**: Each digit is recognizable
2. **Positional accuracy**: Digits are in correct positions
3. **Compositional accuracy**: The full number is correct
4. **Transfer**: A fresh classifier recognizes the generated numbers

## Architecture Changes

### Weaver (Generator)

**Option A: Sequential Generation**
```
Input: z (latent), label (e.g., 372)
→ Decompose label: [3, 7, 2]
→ Generate digit 0: Weaver(z_0, 3) → 28x28 image
→ Generate digit 1: Weaver(z_1, 7) → 28x28 image
→ Generate digit 2: Weaver(z_2, 2) → 28x28 image
→ Compose: concat horizontally → 28x84 image
Output: Multi-digit image
```

**Option B: Holistic Generation**
```
Input: z (latent), label (e.g., 372)
→ Weaver generates full 28x84 image directly
Output: Multi-digit image
```

**Recommendation**: Option A (sequential) because:
- Reuses existing single-digit Weaver
- Curriculum is natural: train single-digit first, then learn to compose
- Failure modes are diagnosable (which digit failed?)

### Witness (Evaluator)

**Per-digit evaluation**:
```
Input: 28x84 image
→ Split into 3 regions: [left, middle, right]
→ Evaluate each: Witness(region_i) → digit prediction
→ Aggregate: accuracy per position, full-number accuracy
```

**Compositional evaluation** (new):
```
→ Also evaluate spatial coherence
→ Are digits properly aligned?
→ Is spacing consistent?
```

### Judge (Ground Truth)

Pre-trained classifier that:
1. Recognizes individual digits (existing Judge)
2. Recognizes multi-digit numbers (new training needed)

## Training Curriculum

### Phase 0: Single-Digit Mastery (use existing GPN-1)
- Train Weaver on single digits until 100% Judge accuracy
- This is the GPN-1 we already have

### Phase 1: Composition Introduction
- Freeze single-digit Weaver weights
- Train only the composition mechanism
- Weaver learns: "put digit X in position Y"

### Phase 2: End-to-End Fine-Tuning
- Unfreeze all weights
- Train on full multi-digit numbers
- Weaver can adjust digit generation for compositional context

### Phase 3: Drift Test
- Remove grounding, test stability

## Ablations

### Ablation 1: No Curriculum
- Train directly on multi-digit from scratch
- Skip Phase 0/1, go directly to Phase 2
- Prediction: slower convergence, worse transfer

### Ablation 2: No Composition Phase
- Train single digits (Phase 0)
- Skip to end-to-end (Phase 2)
- Prediction: similar final accuracy, but less stable

### Ablation 3: Random Digit Order
- Instead of [hundreds, tens, ones], randomize positions
- Tests whether spatial structure matters

## Metrics

### Atomic Metrics (per digit)
- `digit_0_accuracy`: Is the first digit correct?
- `digit_1_accuracy`: Is the second digit correct?
- `digit_2_accuracy`: Is the third digit correct?

### Compositional Metrics
- `full_number_accuracy`: Are ALL digits correct?
- `position_accuracy`: Are digits in right positions?
- `partial_credit`: What fraction of digits are correct?

### Transfer Metrics
- `fresh_classifier_accuracy`: Does a new classifier recognize the numbers?
- `cross_architecture_gap`: V3 vs GAN difference on fresh classifier

## Implementation Plan

### Step 1: Multi-Digit Dataset
Create a dataset of multi-digit MNIST numbers:
- Concatenate 2-3 MNIST digits horizontally
- Label with the full number (0-999)
- ~60k training examples

### Step 2: Multi-Digit Judge
Train a classifier on the multi-digit dataset:
- Input: 28x84 image
- Output: 1000-class (or per-position) prediction

### Step 3: Sequential Weaver
Modify Weaver to generate multi-digit:
- Accept multi-digit label
- Generate each digit position
- Compose into full image

### Step 4: Multi-Position Witness
Modify Witness to evaluate multi-digit:
- Per-position classification
- Full-number classification
- Compositional coherence score

### Step 5: Curriculum Training
Implement phased training:
- Load pre-trained single-digit Weaver
- Add composition layers
- Phased unfreezing

## Success Criteria for Phase 2

1. **Curriculum helps**: Curriculum-trained version reaches 95%+ full-number accuracy faster than no-curriculum version
2. **Transfer holds**: Multi-digit V3 maintains transfer advantage over multi-digit GAN
3. **Meta-learning matters**: If improvement-based training helps more than grounding-only on this task

## Open Questions

1. Should we use 2-digit (100 classes) or 3-digit (1000 classes)?
   - 2-digit is simpler, might not stress the system enough
   - 3-digit has more compositional structure

2. How to handle composition mechanism?
   - Simple concatenation vs learned positioning
   - Fixed spacing vs adaptive spacing

3. Does the meta-learning inner loop help here where it didn't help on single-digit?
   - Key question: "improvement on composition" vs "correctness on composition"
