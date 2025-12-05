# Quickstart: Neutrosophic Metrics

**Feature**: Neutrosophic Relationship Metrics (GPN-2)
**Last Updated**: 2025-12-04

## What Are Neutrosophic Metrics?

Neutrosophic metrics replace scalar "health" scores with three-dimensional {T, I, F} tracking:

- **T (Truth)**: Evidence of genuine synchronization between Weaver and Witness
- **I (Indeterminacy)**: Evidence of honest uncertainty in the relationship
- **F (Falsity)**: Evidence of gaming, collusion, or mode collapse

This preserves epistemic texture that single-number metrics destroy.

## Quick Usage

### 1. Add Neutrosophic Tracker to Training

```python
from src.models.neutrosophic_tracker import NeutrosophicTracker

# In GPNTrainer.__init__:
self.neutrosophic_tracker = NeutrosophicTracker(ema_decay=0.9)
```

### 2. Update Tracker Each Step

```python
# In GPNTrainer.train_step(), after computing losses:
neutro_state = self.neutrosophic_tracker.update(
    v_pred=weaver_attributes,           # [batch, attribute_dim]
    v_seen=witness_attributes,          # [batch, attribute_dim]
    judge_logits=judge_logits,          # [batch, num_classes]
    generated_images=generated_images,  # [batch, C, H, W]
    labels=labels,                      # [batch]
    witness_logits=witness_logits,      # [batch, num_classes]
    witness_real_accuracy=witness_real_acc,  # float
    witness_gen_accuracy=witness_gen_acc,    # float
)
```

### 3. Log Neutrosophic Values

```python
# Log to TensorBoard
self.logger.log_scalar('neutrosophic/T', neutro_state['T'], step)
self.logger.log_scalar('neutrosophic/I', neutro_state['I'], step)
self.logger.log_scalar('neutrosophic/F', neutro_state['F'], step)
self.logger.log_scalar('neutrosophic/T_ema', neutro_state['T_ema'], step)
self.logger.log_scalar('neutrosophic/I_ema', neutro_state['I_ema'], step)
self.logger.log_scalar('neutrosophic/F_ema', neutro_state['F_ema'], step)
```

### 4. Visualize Trajectories

```python
# After training, visualize T/I/F over time
from src.utils.visualization import plot_neutrosophic_trajectories

plot_neutrosophic_trajectories(
    log_dir='experiments/gpn1_neutrosophic',
    save_path='neutrosophic_plot.png'
)
```

## Interpreting Values

### Healthy Training Pattern

```
Step 0:    T=0.0, I=1.0, F=0.0  (High uncertainty, no sync yet)
Step 1000: T=0.4, I=0.6, F=0.1  (Starting to synchronize)
Step 3000: T=0.7, I=0.3, F=0.1  (Strong sync, low uncertainty)
Step 5000: T=0.8, I=0.2, F=0.1  (Mature relationship)
```

**Expected trajectory**:
- T increases (0 → 0.8)
- I decreases (1.0 → 0.2)
- F stays low (<0.2)

### Mode Collapse Pattern

```
Step 3000: T=0.3, I=0.4, F=0.7  (High F! Mode collapse detected)
```

**Diagnosis**: F > 0.5 indicates:
- Low diversity in generated outputs
- Possible collusion (aligned but wrong)
- Gaming (Witness learns real data, ignores Weaver)

### Collusion Pattern

```
Step 3000: T=0.2, I=0.3, F=0.6  (High F, low T)
```

**Diagnosis**: High F with low T indicates:
- Weaver and Witness are aligned (agreeing)
- But Judge accuracy is low (they're wrong)
- This is collusion, not genuine learning

## Validation Experiments

Before trusting neutrosophic metrics on real training, run validation:

```bash
# 1. Healthy baseline (should show T↑, I↓, F low)
python -m experiments.neutrosophic_validation.healthy_baseline

# 2. Mode collapse (should show F↑)
python -m experiments.neutrosophic_validation.mode_collapse

# 3. Collusion (should show F↑, T low)
python -m experiments.neutrosophic_validation.collusion

# 4. Gaming (should show F↑)
python -m experiments.neutrosophic_validation.gaming
```

**Success criteria**: All four experiments produce expected T/I/F patterns.

## Advanced Usage

### Access Sub-Components

```python
# Get breakdown of T, I, F into sub-components
components = self.neutrosophic_tracker.get_components()

# Inspect individual components
print(f"Alignment: {components['alignment']:.3f}")
print(f"Judge accuracy: {components['judge_accuracy']:.3f}")
print(f"Collusion: {components['collusion']:.3f}")
print(f"Mode collapse: {components['mode_collapse']:.3f}")
```

### Adaptive Training

```python
# Example: Increase grounding weight if F is high
if neutro_state['F'] > 0.5:
    # Gaming/collusion detected, increase Judge supervision
    grounding_weight *= 1.5
    logger.warning(f"High F ({neutro_state['F']:.2f}), increasing grounding")
```

### Custom EMA Decay

```python
# Slower EMA for more stable tracking
tracker = NeutrosophicTracker(ema_decay=0.95)

# Faster EMA for responsive tracking
tracker = NeutrosophicTracker(ema_decay=0.8)
```

## Troubleshooting

### Problem: All values are NaN

**Cause**: Likely division by zero or log of zero in entropy computation.

**Fix**: Check that witness_logits has no zeros. The implementation adds 1e-8 epsilon for numerical stability.

### Problem: F is always high, even for healthy training

**Cause**: Thresholds may need calibration for your domain.

**Fix**: Run validation experiments to establish baselines. Adjust component weights in neutrosophic_tracker.py if needed.

### Problem: T never increases above 0.3

**Cause**: Judge may not be learning, or Weaver/Witness are not synchronizing.

**Fix**:
1. Check Judge accuracy is increasing
2. Check alignment loss is decreasing
3. Verify v_pred and v_seen are actually being compared

## Next Steps

1. Run validation experiments to verify T/I/F work as expected
2. Train GPN-1 with neutrosophic tracking enabled
3. Compare neutrosophic trajectories for successful vs failed runs
4. Use insights to design Phase 3 drift experiments

## Further Reading

- [spec.md](./spec.md): Full feature specification
- [research.md](./research.md): Research on how T/I/F are computed
- [data-model.md](./data-model.md): Entity relationships and state transitions
- [contracts/neutrosophic_tracker_interface.py](./contracts/neutrosophic_tracker_interface.py): API contract

