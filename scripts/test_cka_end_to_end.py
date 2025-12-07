#!/usr/bin/env python3
"""
End-to-End CKA Analysis Test

Prove the complete pipeline:
  Load → Extract → Analyze → Result

Using the proven foundation from test_direct_checkpoint_load.py
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.weaver import create_weaver
from models.acgan import ACGANGenerator
from analysis.similarity import compute_cka

print("=" * 80)
print("END-TO-END TEST: CKA Analysis on GPN vs GAN Features")
print("=" * 80)

device = torch.device('cpu')

# ==============================================================================
# STEP 1: LOAD MODELS (proven to work)
# ==============================================================================
print("\nSTEP 1: Loading models...")

# GPN
gpn_ckpt = torch.load('checkpoints/checkpoint_final.pt', map_location=device, weights_only=False)
gpn_weaver = create_weaver(latent_dim=64, num_classes=10)
gpn_weaver.load_state_dict(gpn_ckpt['models']['weaver'])
gpn_weaver.eval()
print("  ✓ GPN loaded")

# GAN
gan_ckpt = torch.load('checkpoints/acgan_final.pt', map_location=device, weights_only=False)
gan_generator = ACGANGenerator(latent_dim=64, num_classes=10)
gan_generator.load_state_dict(gan_ckpt['models']['generator'])
gan_generator.eval()
print("  ✓ GAN loaded")

# ==============================================================================
# STEP 2: EXTRACT FEATURES (proven to work)
# ==============================================================================
print("\nSTEP 2: Extracting features from multiple samples...")

# Generate more samples for meaningful statistics
num_samples = 100
batch_size = 10
num_batches = num_samples // batch_size

gpn_fc_features = []
gan_fc_features = []

def get_activation(feature_list):
    def hook(model, input, output):
        feature_list.append(output.detach())
    return hook

# Register hooks
gpn_hook = gpn_weaver.fc.register_forward_hook(get_activation(gpn_fc_features))
gan_hook = gan_generator.fc[0].register_forward_hook(get_activation(gan_fc_features))

# Extract features over multiple batches
print(f"  Generating {num_samples} samples across {num_batches} batches...")
for i in range(num_batches):
    z = torch.randn(batch_size, 64)
    labels = torch.randint(0, 10, (batch_size,))

    with torch.no_grad():
        _ = gpn_weaver(z, labels)
        _ = gan_generator(z, labels)

    if (i + 1) % 5 == 0:
        print(f"    Batch {i+1}/{num_batches} complete")

# Remove hooks
gpn_hook.remove()
gan_hook.remove()

# Concatenate all batches
gpn_features = torch.cat(gpn_fc_features, dim=0).numpy()  # [100, 25088]
gan_features = torch.cat(gan_fc_features, dim=0).numpy()  # [100, 12544]

print(f"\n  ✓ Extracted features:")
print(f"    GPN: {gpn_features.shape}")
print(f"    GAN: {gan_features.shape}")

# ==============================================================================
# STEP 3: RUN CKA ANALYSIS
# ==============================================================================
print("\nSTEP 3: Computing CKA (Centered Kernel Alignment)...")
print("  This measures similarity of representation geometry")
print("  CKA = 1.0: identical geometry (up to rotation/scaling)")
print("  CKA = 0.0: completely uncorrelated representations")

# Compute CKA
cka_score = compute_cka(gpn_features, gan_features, kernel='linear')

print(f"\n  ✓ CKA computed: {cka_score:.6f}")

# ==============================================================================
# STEP 4: INTERPRET RESULT
# ==============================================================================
print("\nSTEP 4: Interpreting result...")

if cka_score > 0.9:
    interpretation = "VERY SIMILAR - Nearly identical representational geometry"
elif cka_score > 0.7:
    interpretation = "SIMILAR - Shared representational structure"
elif cka_score > 0.5:
    interpretation = "MODERATELY SIMILAR - Some shared structure"
elif cka_score > 0.3:
    interpretation = "WEAKLY SIMILAR - Limited shared structure"
else:
    interpretation = "DISSIMILAR - Fundamentally different representations"

print(f"\n  CKA Score: {cka_score:.6f}")
print(f"  Interpretation: {interpretation}")

# Additional context
print("\n  Context:")
print(f"    - Comparing fc layer features (after latent projection)")
print(f"    - GPN trained with pedagogical objective (cooperative)")
print(f"    - GAN trained with adversarial objective (competitive)")
print(f"    - Both achieve similar visual quality on MNIST")
print(f"    - But GPN achieves 100% vs GAN's 81% on novel combinations")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("✓✓✓ END-TO-END ANALYSIS COMPLETE ✓✓✓")
print("=" * 80)
print("\nPipeline proven:")
print("  1. ✓ Load models from checkpoints")
print("  2. ✓ Extract features from 100 samples")
print("  3. ✓ Compute CKA between representations")
print("  4. ✓ Produce interpretable result")
print(f"\nRESULT: GPN vs GAN fc layer CKA = {cka_score:.6f}")
print(f"        {interpretation}")
print("\nThis is the pattern all other analyses follow.")
print("Proven ground for Checkpoint 1.")
print("=" * 80)
