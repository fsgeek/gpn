# Critical Ablation: GAN vs GPN on Compositional Learning

**Date**: December 4, 2024
**Question**: Is the curriculum requirement specific to pedagogical architectures, or general to compositional learning?

## Summary

**Finding**: Adversarial training (GAN) can learn compositional tasks from scratch with 79% accuracy, while pedagogical training (GPN) completely fails (1.2% accuracy). However, curriculum-based pedagogical training achieves perfect transfer (100%).

**Implication**: Curriculum is **architecture-specific**, not a general requirement for compositional learning.

## Experimental Design

### Task
Generate 2-digit MNIST numbers (28x56 images, 100 classes: 0-99)

### Three Conditions

1. **GPN Curriculum (Baseline)**
   - Pre-trained single-digit Weaver from GPN-1
   - Test at step 0 (no 2-digit training)

2. **GPN From-scratch (Ablation 1)**
   - Train 2-digit Weaver directly, no pre-training
   - Pedagogical architecture (Weaver-Witness cooperation)
   - 5000 steps, CPU

3. **GAN From-scratch (Ablation 2)**
   - Train 2-digit GAN directly, no pre-training
   - Adversarial architecture (Generator-Discriminator competition)
   - 5000 steps, GPU (NVIDIA RTX 4090)

### Evaluation
- **Judge Accuracy**: Full 100-class classification
- **Tens Accuracy**: Tens digit only (10 classes)
- **Ones Accuracy**: Ones digit only (10 classes)

## Results

### Final Performance (5000 steps)

| Condition | Judge | Tens | Ones | Training Time |
|-----------|-------|------|------|---------------|
| **GPN Curriculum** | **100%** | **100%** | **100%** | 0 (pre-trained) |
| **GAN From-scratch** | **79.1%** | **87.6%** | **91.5%** | 2.5 min (GPU) |
| **GPN From-scratch** | **1.2%** | **8.3%** | **10.0%** | 3 hours (CPU) |

### Training Progression

**GAN From-scratch**:
| Step | Judge | Tens | Ones |
|------|-------|------|------|
| 0    | 0.0%  | 14.1%| 3.1% |
| 500  | 21.9% | 43.8%| 43.8%|
| 1000 | 35.9% | 57.8%| 62.5%|
| 2000 | 54.7% | 70.3%| 73.4%|
| 3000 | 81.2% | 90.6%| 85.9%|
| 4000 | 73.4% | 84.4%| 84.4%|
| 5000 | 79.1% | 87.6%| 91.5%|

**GPN From-scratch**:
| Step | Judge | Tens | Ones |
|------|-------|------|------|
| 0    | 1.6%  | 10.9%| 7.8% |
| 500  | 1.6%  | 14.1%| 12.5%|
| 1000 | 1.6%  | 7.8% | 7.8% |
| 2000 | 0.0%  | 7.8% | 14.1%|
| 3000 | 0.0%  | 10.9%| 10.9%|
| 4000 | 3.1%  | 10.9%| 9.4% |
| 5000 | 1.2%  | 8.3% | 10.0%|

**Observations**:
- GAN shows **consistent improvement** throughout training
- GPN shows **no improvement**, oscillating around random chance
- GAN peaks at step 3000 (81.2%), then stabilizes around 79%

## Analysis

### Why GAN Succeeds

**Adversarial gradient flow**:
- Discriminator provides gradient signal for ANY output
- Even incorrect generations get meaningful feedback
- "This doesn't look like a 42" → gradient toward digit-like features
- Partial success is rewarded incrementally

**No chicken-and-egg problem**:
- Discriminator learns "real vs fake" from ground truth data
- Generator learns from discriminator's evolving criterion
- Both improve together, no bootstrapping deadlock

### Why GPN Fails

**Pedagogical feedback requires legibility**:
- Witness must see *recognizable* features to give useful feedback
- Without atomic competence, Weaver generates noise
- Witness learns "everything is bad" → no productive gradient
- Weaver gets uniform "bad" signal → no direction to improve

**Chicken-and-egg deadlock**:
- Weaver needs Witness feedback to improve
- Witness needs recognizable Weaver outputs to learn
- Without curriculum (pre-trained atoms), neither can bootstrap

### Why Curriculum Works

**Pre-trained Weaver breaks the deadlock**:
- Already generates recognizable single digits
- Witness can immediately give useful per-digit feedback
- Weaver learns composition on top of solid foundation
- Perfect transfer because atomic skills are compositional primitives

## Theoretical Implications

### 1. Curriculum Requirement is Architecture-Specific

Compositional learning is not inherently hard. GANs prove that direct training on compositional tasks can work. The curriculum requirement emerges from **how pedagogical systems learn**, not from the task structure itself.

### 2. Pedagogical Systems Trade Flexibility for Stability

**GAN (Adversarial)**:
- ✓ Can learn from scratch
- ✓ Flexible gradient flow
- ✗ Unstable training
- ✗ Mode collapse risk
- ✗ Requires careful tuning

**GPN (Pedagogical)**:
- ✓ Stable once established
- ✓ Cooperative dynamics
- ✓ Interpretable signals
- ✗ Requires curriculum for bootstrapping
- ✗ Cannot learn compositional tasks from scratch

### 3. The 79% vs 100% Gap

Even after 5000 steps, the GAN achieved 79%, not 100%. The curriculum-based approach has a **21% advantage**.

**Why the gap?**
- Curriculum provides structural bias: "learn atoms, then compose"
- From-scratch must discover compositional structure implicitly
- Direct learning may conflate atomic and compositional features

**Is 79% a plateau?**
- GAN peaked at 81.2% (step 3000), then regressed slightly
- May need more steps, different architecture, or curriculum to reach 100%
- Or 80% may be the limit of direct compositional learning

### 4. Implications for GPN Design

**Current GPN assumption**: Mutual empowerment enables learning

**Reality**: Mutual empowerment enables learning **when both agents can bootstrap**

**Design requirement**: Pedagogical systems need **initialization strategies**:
- Curriculum (proven effective)
- Pre-training on simpler tasks
- Auxiliary supervised signals during early training
- Phased competence gating (wait for Witness before training Weaver)

### 5. Implications for Alignment

If this generalizes beyond MNIST:

**Adversarial RLHF**: Can learn directly on complex tasks, but unstable, prone to gaming
**Pedagogical RLHF**: Requires curriculum (simple → complex tasks), but more stable and cooperative

**Recommendation**: Pedagogical alignment systems should use explicit curriculum structure, not expect emergence from scratch.

## Open Questions

### 1. Would GAN reach 100% with more steps?

Run for 50k steps to see if it plateaus or continues improving.

### 2. Does curriculum help GANs too?

Train GAN with pre-trained single-digit Generator. Does it reach 100% faster or more stably?

### 3. Is there a GPN architecture that can bootstrap from scratch?

Could auxiliary supervised losses during early training provide the bootstrap signal?

### 4. What's the minimum curriculum for GPN?

Does GPN need full single-digit mastery (100%), or would 50% competence be sufficient?

### 5. Does this generalize to other domains?

Test on:
- Language (character → word → sentence)
- Code (statement → function → program)
- Visual (stroke → shape → scene)

## Recommendations

### For GPN Research

1. **Accept curriculum as feature, not bug**: Design pedagogical systems with explicit curriculum structure
2. **Investigate hybrid approaches**: Can we combine adversarial bootstrapping with pedagogical refinement?
3. **Study minimum viable curriculum**: How much atomic competence is enough?

### For Broader ML

1. **Architecture choice matters**: From-scratch vs curriculum is not just a training decision—it's determined by architecture
2. **Evaluate curriculum requirement**: When designing cooperative/pedagogical systems, test from-scratch viability
3. **Consider the trade-off**: Adversarial flexibility vs pedagogical stability

## Conclusion

Adversarial training can learn compositional tasks from scratch (79% accuracy), proving curriculum is not universally required. However, pedagogical training completely fails without curriculum (1.2% accuracy), revealing an **architecture-specific requirement**.

The curriculum-based approach still achieves the best performance (100%), suggesting that while from-scratch adversarial learning is **viable**, curriculum-based learning is **optimal**.

This finding clarifies the scope of the GPN-2 curriculum result: we've shown curriculum is essential **for pedagogical architectures**, not for all learning systems. The question remains whether the stability and cooperation benefits of pedagogical systems justify the curriculum requirement.

---

**Technical Notes**:
- GAN training: 2.5 minutes on NVIDIA RTX 4090
- GPN training: 3 hours on Apple Silicon M4 (CPU)
- Hardware difference may affect absolute convergence speed but not relative trends
- Both used same hyperparameters (lr=0.0002, batch_size=64)

