# Research: Neutrosophic Metrics Computation

**Date**: 2025-12-04
**Feature**: Neutrosophic Relationship Metrics (GPN-2)

## Research Questions

1. How to compute T (truth) from observables?
2. How to compute I (indeterminacy) from observables?
3. How to compute F (falsity) from observables?
4. What are appropriate value ranges for T, I, F?
5. How to validate that T/I/F actually detect gaming vs genuine learning?

## Decision: Observable-Based T/I/F Computation

### T (Truth): Genuine Synchronization

**Definition**: Evidence that Weaver and Witness are genuinely aligned AND both correct according to Judge.

**Computation**:
```python
# T measures three-way agreement: Weaver claims → Witness observations → Judge verdict
# All three must agree for high T

# 1. Alignment: Do Weaver's v_pred match Witness's v_seen?
alignment = 1 - torch.mean((v_pred - v_seen)**2)  # MSE distance, inverted

# 2. Correctness: Does Judge agree with their consensus?
judge_accuracy = (judge_predictions == labels).float().mean()

# 3. Learning progress: Is Judge accuracy improving?
judge_improvement = judge_accuracy - judge_accuracy_ema  # Positive = improving

# Combined T metric (weighted average)
T = 0.4 * alignment + 0.4 * judge_accuracy + 0.2 * max(0, judge_improvement)
```

**Rationale**:
- Pure alignment isn't enough (could be collusion)
- Judge accuracy alone isn't enough (could be lucky guessing)
- Learning progress ensures it's not static
- All three components normalized to [0,1]

**Alternatives considered**:
- Mutual information between Weaver/Witness: Too expensive to compute
- Cross-entropy based: Doesn't capture alignment vs correctness distinction

### I (Indeterminacy): Honest Uncertainty

**Definition**: Evidence of epistemic uncertainty in the relationship, distinct from noise.

**Computation**:
```python
# I measures variance/entropy in predictions

# 1. Weaver uncertainty: Variance in v_pred across batch
weaver_uncertainty = torch.var(v_pred, dim=0).mean()

# 2. Witness uncertainty: Entropy in classification logits
witness_entropy = -torch.sum(witness_probs * torch.log(witness_probs + 1e-8), dim=1).mean()

# 3. Disagreement: When Weaver and Witness don't align
disagreement = torch.mean((v_pred - v_seen)**2)

# Combined I metric
I = 0.3 * weaver_uncertainty + 0.3 * witness_entropy + 0.4 * disagreement
```

**Rationale**:
- Uncertainty is epistemic (we don't know) not aleatory (inherently random)
- High I early in training is expected and healthy
- High I late in training suggests relationship hasn't stabilized
- Normalized to [0,1] using dataset statistics

**Alternatives considered**:
- Predictive entropy alone: Doesn't capture disagreement
- Only measuring disagreement: Doesn't capture within-agent uncertainty

### F (Falsity): Gaming/Collusion

**Definition**: Evidence that Weaver and Witness are aligned but both wrong, or that the system is gaming metrics.

**Computation**:
```python
# F measures alignment WITHOUT correctness

# 1. Collusion: High alignment but low Judge accuracy
alignment = 1 - torch.mean((v_pred - v_seen)**2)
judge_accuracy = (judge_predictions == labels).float().mean()
collusion = alignment * (1 - judge_accuracy)  # High when aligned but wrong

# 2. Mode collapse: Low diversity in generated outputs
# Measure using inverse of output variance
diversity = torch.var(generated_images.reshape(batch_size, -1), dim=0).mean()
mode_collapse = 1 / (1 + diversity)  # High when low diversity

# 3. Gaming: Witness accuracy on real data is high, but on generated data is low
# (Witness learned to classify real data, not generated)
witness_real_acc = witness_accuracy_on_real_data
witness_gen_acc = witness_accuracy_on_generated_data
gaming = max(0, witness_real_acc - witness_gen_acc)

# Combined F metric
F = 0.4 * collusion + 0.3 * mode_collapse + 0.3 * gaming
```

**Rationale**:
- Collusion is the core failure mode we want to detect
- Mode collapse is a known GAN pathology
- Gaming (Witness learns real data distribution, not Weaver's) indicates failed pedagogy
- All components normalized to [0,1]

**Alternatives considered**:
- Binary collusion detection: Too coarse, loses information
- Discriminator-based gaming detection: Adds adversarial component we're avoiding

## Decision: Value Ranges

**Chosen**: Normalize all T, I, F to [0, 1] range.

**Rationale**:
- Interpretable: 0 = none, 1 = maximum
- Comparable across metrics
- Standard for probabilistic/fuzzy measures
- Easy to visualize and track over training

**Alternatives considered**:
- Raw counts: Not comparable, scales differ
- Neutrosophic sets [T, I, F] ∈ ]−0, 1+[: Mathematically rigorous but unnecessarily complex for our use case

## Decision: Validation Strategy

**Approach**: Synthetic failure modes + baseline comparison

### Validation Experiments:

**1. Mode Collapse (Induced)**
- Train GPN but force Weaver to always generate same class
- **Expected**: F > 0.7 (high mode collapse signal)
- **Expected**: T < 0.3 (low genuine synchronization)

**2. Collusion (Induced)**
- Train Weaver and Witness with shared random seed
- Both learn same biased pattern (e.g., always predict class 0)
- **Expected**: High alignment, low Judge accuracy
- **Expected**: F > 0.6 (collusion signal)

**3. Healthy Baseline (GPN-1)**
- Use existing GPN-1 successful training run
- **Expected**: T increases over training (0.2 → 0.8)
- **Expected**: I decreases over training (0.7 → 0.3)
- **Expected**: F stays low (<0.3)

**4. Gaming (Induced)**
- Train Witness on real MNIST, Weaver on random noise
- Witness learns real distribution, ignores Weaver
- **Expected**: F > 0.5 (gaming signal: high witness_real_acc, low witness_gen_acc)

### Success Criteria:
- All four validation experiments produce expected T/I/F patterns
- T/I/F trajectories are reproducible (same seed → same metrics)
- Visualization clearly shows differences between failure modes

## Best Practices

### PyTorch Implementation:
- Use `torch.no_grad()` for metric computation (don't backprop through metrics)
- Compute metrics on same batch used for training (representative sample)
- Use EMA smoothing for stability: `metric_ema = 0.9 * metric_ema + 0.1 * metric`

### Logging:
- Log T, I, F at same frequency as other metrics (every 100 steps)
- Use TensorBoard scalar logging for time series
- Create custom TensorBoard tab for T/I/F visualization

### Performance:
- Batch all metric computations together (avoid multiple forward passes)
- Reuse existing computations (e.g., Judge accuracy already computed for grounding loss)
- Profile to ensure <10% overhead

## Open Questions Resolved

✅ **Q1: How to compute T from observables?**
→ Weighted combination of alignment + Judge accuracy + learning progress

✅ **Q2: How to compute I from observables?**
→ Weighted combination of weaver uncertainty + witness entropy + disagreement

✅ **Q3: How to compute F from observables?**
→ Weighted combination of collusion + mode collapse + gaming

✅ **Q4: What are appropriate value ranges?**
→ Normalize all to [0, 1] for interpretability

✅ **Q5: How to validate T/I/F?**
→ Synthetic failure mode experiments before real deployment

## Next Steps (Phase 1)

With research complete, proceed to Phase 1:
1. Design data model for `NeutrosophicTracker` class
2. Generate API contracts (class interface)
3. Create quickstart guide for using neutrosophic metrics

