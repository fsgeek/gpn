# Domain Generalization Scoping Document

## Purpose

Validate that the pedagogical training advantage (100% vs ~80% compositional transfer) generalizes beyond MNIST/Fashion-MNIST to:
1. **Transformer architecture** - Critical for RLHF-alternative positioning
2. **CIFAR-10** - Real images with natural visual features
3. **SCAN/COGS** - Compositional language benchmarks (stretch goal)

---

## 1. Transformer Experiment

### Recommended Task: SCAN-lite (Simplified Command Parsing)

**Why SCAN-lite over alternatives:**

| Task | Compositional Structure | Judge Clarity | Literature Comparison | Recommendation |
|------|------------------------|---------------|----------------------|----------------|
| Counting | Weak (count × object) | Clear | Limited | No |
| Parity/Logic | Weak (operation × bits) | Clear | Limited | No |
| Arithmetic | Medium (operation × numbers) | Clear | Some | Maybe |
| SCAN-lite | Strong (action × modifier × count) | Clear | Strong (SCAN benchmark) | **YES** |

**SCAN-lite advantages:**
- Direct comparison to SCAN literature (established compositional benchmark)
- Clear compositional structure: "jump twice" → "JUMP JUMP"
- Deterministic ground truth (Judge is trivially correct)
- Natural train/test split: hold out novel combinations (e.g., "jump thrice" never seen)
- Small vocabulary, fast training

### Task Design

**Primitives:**
- Actions: `walk`, `run`, `jump`, `look` (4)
- Modifiers: `left`, `right`, `around` (3)
- Counts: `once`, `twice`, `thrice` (3)

**Compositions:**
- "walk left twice" → "LTURN WALK LTURN WALK"
- "jump around" → "LTURN JUMP LTURN JUMP LTURN JUMP LTURN JUMP"

**Holdout strategy:**
- Train: All combinations EXCEPT specific (action, modifier, count) triples
- Test: Held-out combinations only
- Example holdouts: ("jump", "around", "twice"), ("run", "left", "thrice")

### GPN Triad for Transformers

```
Weaver (Generator):
- Encoder-decoder Transformer
- Input: command tokens
- Output: action sequence tokens + v_pred (from [CLS] or pooled output)

Witness (Predictor):
- Encoder-only Transformer
- Input: generated action sequence
- Output: predicted command + v_seen

Judge (Oracle):
- Deterministic parser (not neural)
- Input: generated action sequence
- Output: correct/incorrect classification
```

### Adversarial vs Pedagogical Training

**Adversarial (Baseline):**
- Train Weaver to fool a discriminator that distinguishes real vs generated sequences
- Discriminator sees (command, action_sequence) pairs
- Standard seq2seq GAN objective

**Pedagogical (Ours):**
- Phase 1: Heavy grounding (Judge correctness signal)
- Phase 2: Balanced (Weaver predicts Witness perception via v_pred/v_seen alignment)
- Phase 3: Drift test (minimal supervision)
- Same three-phase curriculum as image experiments

### Architecture Sketch

```yaml
weaver:
  type: encoder_decoder_transformer
  d_model: 128
  n_heads: 4
  n_encoder_layers: 3
  n_decoder_layers: 3
  vocab_size: ~50 (small)
  v_pred_dim: 16

witness:
  type: encoder_transformer
  d_model: 128
  n_heads: 4
  n_layers: 3
  v_seen_dim: 16

judge:
  type: deterministic_parser
  # No neural network - direct rule evaluation
```

### Compositional Generalization Measurement

**Metrics:**
- **Sequence accuracy**: Exact match on full output sequence
- **Token accuracy**: Per-token correctness
- **Compositional accuracy**: Accuracy on held-out combinations only

**Expected Results:**
- Adversarial: ~80-85% on held-out (consistent with SCAN literature)
- Pedagogical: Target 95-100% (demonstrating the advantage)

### Training Estimates

- Vocabulary: ~50 tokens
- Sequence length: ~20 tokens max
- Training examples: ~1000 (small by design)
- Batch size: 32
- Steps: ~5000-10000

**Estimated time on 4090: 30-60 minutes per seed**

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Seq2seq GAN unstable | Medium | Use standard techniques (teacher forcing, scheduled sampling) |
| Pedagogical doesn't help | Low | Core mechanism is architecture-agnostic |
| Task too easy | Low | SCAN variants are known to be hard for transformers |
| Task too hard | Low | Using simplified version |

---

## 2. CIFAR-10 Experiment

### Compositional Task Design

**Challenge:** CIFAR-10 categories don't have natural ordering like digits.

**Proposed approach: Spatial Composition**
- Generate 32x64 images with [Object A][Object B] side by side
- Test: novel (A, B) combinations
- Example: Train on (cat, dog), (bird, frog), test on (cat, frog)

**Alternative: Semantic Relation**
- Generate [Animal][Vehicle] pairs
- Animals: bird, cat, deer, dog, frog, horse (6)
- Vehicles: airplane, automobile, ship, truck (4)
- 24 possible pairs, hold out 4-6 for testing

### Architecture Adaptation

**Current MNIST Weaver:**
- Input: 28x28 grayscale
- Latent dim: 64
- Conv layers: 4 blocks

**CIFAR-10 Weaver (proposed):**
- Input: 32x32 RGB (or 32x64 for composition)
- Latent dim: 128 (increased for complexity)
- Conv layers: 5-6 blocks (deeper for natural images)
- Use residual connections (ResNet-style)

```python
# Key changes from MNIST Weaver:
image_channels = 3  # was 1
init_size = 4  # adjusted for 32x32
latent_dim = 128  # was 64
# Add residual blocks
```

### Judge Design

**Option A: Pretrained ResNet-18**
- ~95% accuracy on CIFAR-10
- Frozen during training
- Well-characterized behavior

**Option B: Train our own**
- Consistent with MNIST approach
- More control over architecture
- ~3 epochs to 90%+ accuracy

**Recommendation:** Option A (pretrained ResNet-18) for speed and reliability.

### Adversarial Baseline

Use established CIFAR-10 GAN:
- DCGAN adapted for 32x32 RGB
- Or use publicly available pretrained weights
- Ensure same model capacity as pedagogical Weaver

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GAN mode collapse on CIFAR-10 | High | High | Use proven architectures, spectral norm |
| Pedagogical can't generate recognizable CIFAR-10 | Medium | Critical | Start with single-class generation first |
| Compositional task poorly defined | Medium | Medium | Clear holdout protocol, semantic groupings |
| Training too slow | Low | Medium | Use pretrained components where possible |

### Training Estimates

- Single-class Weaver: ~2-4 hours
- Compositional layer: ~1-2 hours
- Per seed total: ~4-6 hours

**Estimated time on 4090: 4-6 hours per seed, 20-30 hours for 5 seeds**

### Simpler Intermediate Step

If CIFAR-10 proves too hard, consider **SVHN** (Street View House Numbers):
- 32x32 color images of digits
- Harder than MNIST but with known compositional structure
- Multi-digit recognition is natural compositional task

---

## 3. SCAN/COGS (Stretch Goal)

### Why This Matters

SCAN and COGS are THE benchmarks for compositional generalization in NLP:
- SCAN: Command → action sequence
- COGS: Sentence → semantic parse

Success here would be high-impact, directly addressing the compositional generalization literature.

### Adaptation Challenge

SCAN/COGS use sequence-to-sequence models. The GPN triad needs rethinking:
- **Weaver**: Generates output sequence from input
- **Witness**: Predicts Weaver's behavior (but what does this mean for seq2seq?)
- **Judge**: Evaluates correctness (easy - ground truth available)

The key insight: **v_pred/v_seen alignment** needs to mean something for sequences.

**Possible approach:**
- v_pred: Weaver's confidence in its generation
- v_seen: Witness's assessment of generation quality
- Alignment: Weaver learns to predict how Witness will perceive the output

### Risk Level: HIGH

This requires the most architectural innovation. Recommend only after Transformer and CIFAR-10 succeed.

---

## Recommended Execution Order

1. **Transformer (SCAN-lite)** - 1-2 days
   - Critical for RLHF-alternative claim
   - Fastest to implement (small scale)
   - Cleanest compositional structure

2. **CIFAR-10** - 3-4 days
   - Validates real image generalization
   - Reuses existing CNN infrastructure
   - Moderate risk

3. **SCAN/COGS** - 5+ days (if needed)
   - Highest impact if successful
   - Highest implementation cost
   - Do only if 1 & 2 succeed and we need stronger evidence

---

## Success Criteria

| Experiment | Adversarial Baseline | Pedagogical Target | Pass |
|------------|---------------------|-------------------|------|
| Transformer (SCAN-lite) | ~80% | >95% | δ > 10% |
| CIFAR-10 | ~75-80% | >90% | δ > 10% |
| SCAN/COGS | ~80% (literature) | >95% | δ > 10% |

**Minimum for paper:** At least ONE non-toy domain showing the pattern.

**Strong paper:** TWO domains (Transformer + CIFAR-10) showing consistent pattern.

**Very strong paper:** All THREE domains confirming the mechanism is fundamental.

---

*Document created: 2025-12-09*
*Status: Ready for implementation*
