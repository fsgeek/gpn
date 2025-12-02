# Quickstart: GPN-1 MNIST Proof-of-Concept

**Date**: 2025-12-02
**Spec**: [spec.md](spec.md)

## Prerequisites

- Python 3.11+
- CUDA-capable GPU (optional, CPU training viable for MNIST)
- ~2GB disk space for experiments

## Installation

```bash
# Clone repository
git clone <repo-url>
cd gpn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Verification

```bash
# Run unit tests
pytest tests/unit -v

# Verify MNIST download
python -c "from torchvision.datasets import MNIST; MNIST('./data', download=True)"
```

## Training the Judge

Before GPN training, pre-train the Judge classifier:

```bash
# Train Judge to >95% accuracy
python scripts/train_judge.py --output checkpoints/judge.pt

# Verify Judge accuracy
python scripts/train_judge.py --verify checkpoints/judge.pt
# Expected output: "Judge accuracy: 97.5% (passes >95% threshold)"
```

## Running GPN-1 Training

### Basic Training Run

```bash
# Train GPN-1 with default config
python -m src.cli.train \
    --config configs/gpn1_default.yaml \
    --seed 42 \
    --output experiments/run_001
```

### Monitor Training

```bash
# In separate terminal
tensorboard --logdir experiments/run_001/logs
# Open http://localhost:6006
```

### Expected Timeline

| Phase | Steps | Duration (GPU) | Duration (CPU) |
|-------|-------|----------------|----------------|
| Phase 1 | 0-5000 | ~15 min | ~60 min |
| Phase 2 | 5000-10000 | ~15 min | ~60 min |
| Phase 3 | 10000-12000 | ~6 min | ~24 min |
| **Total** | 12000 | ~36 min | ~2.5 hrs |

## Running Baseline Comparison

```bash
# Train GAN baseline
python -m src.cli.train \
    --model baseline \
    --config configs/baseline_gan.yaml \
    --seed 42 \
    --output experiments/baseline_001

# Compare results
python scripts/run_experiment.py compare \
    --gpn experiments/run_001 \
    --baseline experiments/baseline_001
```

## Verification Steps

### 1. Check Training Completion

```bash
# View final step
python -m src.cli.evaluate status experiments/run_001
# Expected: "Training completed: 12000/12000 steps, Phase 3"
```

### 2. Verify No Catastrophic Failure

```bash
# Check for NaN losses
python -m src.cli.evaluate check-losses experiments/run_001
# Expected: "All losses finite. No gradient explosions detected."
```

### 3. Check Mode Diversity

```bash
# Compute mode diversity at end of Phase 2
python -m src.cli.evaluate diversity \
    experiments/run_001/checkpoints/step_10000.pt
# Expected: "Mode coverage: 9/10 classes (passes >=8 threshold)"
```

### 4. Check Image Quality

```bash
# Generate sample grid
python -m src.cli.visualize samples \
    experiments/run_001/checkpoints/step_10000.pt \
    --output samples.png

# Compute Judge accuracy
python -m src.cli.evaluate quality experiments/run_001
# Expected: "Judge accuracy on generated: 85.2% (passes >80% threshold)"
```

### 5. Verify Reproducibility

```bash
# Run with same seed
python -m src.cli.train \
    --config configs/gpn1_default.yaml \
    --seed 42 \
    --output experiments/run_001_verify

# Compare trajectories
python -m src.cli.evaluate compare-runs \
    experiments/run_001 \
    experiments/run_001_verify
# Expected: "Trajectories match within numerical precision"
```

## Common Issues

### CUDA Out of Memory

```bash
# Reduce batch size
python -m src.cli.train --batch-size 32 ...
```

### Training Diverges

Check TensorBoard for:
- Loss spikes (may indicate learning rate too high)
- Variance collapse (empowerment loss not preventing mode collapse)

### Phase 3 Collapse

Expected possible outcome. Document:
- Step at which quality degrades
- Rate of degradation
- Whether alignment loss increases

## Full Experiment Suite

For the complete comparison required by the spec:

```bash
# Run 3 GPN seeds
for seed in 42 43 44; do
    python -m src.cli.train \
        --config configs/gpn1_default.yaml \
        --seed $seed \
        --output experiments/gpn_seed_$seed
done

# Run 3 baseline seeds
for seed in 42 43 44; do
    python -m src.cli.train \
        --model baseline \
        --config configs/baseline_gan.yaml \
        --seed $seed \
        --output experiments/baseline_seed_$seed
done

# Statistical comparison
python scripts/run_experiment.py analyze \
    --gpn-runs experiments/gpn_seed_* \
    --baseline-runs experiments/baseline_seed_* \
    --output results/comparison.md
```

## Output Structure

After training:

```
experiments/run_001/
├── config.yaml          # Full configuration used
├── checkpoints/
│   ├── step_5000.pt     # Phase 1 end
│   ├── step_10000.pt    # Phase 2 end
│   └── step_12000.pt    # Final
├── logs/
│   └── events.out.*     # TensorBoard logs
├── samples/
│   ├── step_0500.png
│   ├── step_1000.png
│   └── ...
└── metrics.json         # Summary metrics
```

## Next Steps

After successful training:

1. Review TensorBoard logs for training dynamics
2. Visually inspect sample images at each phase
3. Run statistical comparison if baseline also trained
4. Document findings in experiment notes
