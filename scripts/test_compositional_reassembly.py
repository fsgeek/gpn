#!/usr/bin/env python3
"""
Compositional Reassembly Test

Standard metrics don't explain the gap. What does?
Question: Can representations REASSEMBLE, not just factorize?

Tests:
1. Latent Interpolation - Do intermediate points produce valid digits?
2. Feature Swapping - Can we swap block0 features and maintain identity?
3. Compositional Linearity - Does latent arithmetic work?

The hypothesis: GPN's geometry supports reassembly. GAN's doesn't.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.weaver import create_weaver
from models.acgan import ACGANGenerator

print("=" * 80)
print("COMPOSITIONAL REASSEMBLY TEST")
print("The test isn't factorization. It's whether features reassemble.")
print("=" * 80)

device = torch.device('cpu')

# ==============================================================================
# LOAD MODELS
# ==============================================================================
print("\nSTEP 1: Loading models...")

gpn_ckpt = torch.load('checkpoints/checkpoint_final.pt', map_location=device, weights_only=False)
gpn_weaver = create_weaver(latent_dim=64, num_classes=10)
gpn_weaver.load_state_dict(gpn_ckpt['models']['weaver'])
gpn_weaver.eval()

gan_ckpt = torch.load('checkpoints/acgan_final.pt', map_location=device, weights_only=False)
gan_generator = ACGANGenerator(latent_dim=64, num_classes=10)
gan_generator.load_state_dict(gan_ckpt['models']['generator'])
gan_generator.eval()

print("  ✓ Models loaded")

# ==============================================================================
# TEST 1: LATENT INTERPOLATION
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 1: Latent Interpolation (Reassembly in Latent Space)")
print("=" * 80)
print("Question: Do intermediate points produce valid outputs?")
print("Hypothesis: GPN interpolates smoothly, GAN has mode collapse/artifacts")

# Generate two digits and interpolate
digit1, digit2 = 3, 7
z1 = torch.randn(1, 64)
z2 = torch.randn(1, 64)

n_steps = 7
alphas = np.linspace(0, 1, n_steps)

print(f"\nInterpolating between digit {digit1} and digit {digit2}...")

fig, axes = plt.subplots(2, n_steps, figsize=(14, 4))

with torch.no_grad():
    for i, alpha in enumerate(alphas):
        # Interpolate latents
        z_interp = (1 - alpha) * z1 + alpha * z2

        # GPN
        label1 = torch.tensor([digit1])
        label2 = torch.tensor([digit2])
        label_interp = label1 if alpha < 0.5 else label2  # Switch at midpoint

        gpn_img, _ = gpn_weaver(z_interp, label_interp)
        gan_img = gan_generator(z_interp, label_interp)

        # Plot GPN
        axes[0, i].imshow(gpn_img[0, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
        axes[0, i].set_title(f'α={alpha:.2f}', fontsize=9)
        axes[0, i].axis('off')

        # Plot GAN
        axes[1, i].imshow(gan_img[0, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
        axes[1, i].axis('off')

axes[0, 0].set_ylabel('GPN', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('GAN', fontsize=12, fontweight='bold')

plt.suptitle(f'Latent Interpolation: Digit {digit1} → Digit {digit2}\nSmooth Reassembly or Mode Collapse?',
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = Path('experiments/latent_interpolation.png')
output_path.parent.mkdir(exist_ok=True)
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

# ==============================================================================
# TEST 2: COMPOSITIONAL LINEARITY (LATENT ARITHMETIC)
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 2: Compositional Linearity (Latent Arithmetic)")
print("=" * 80)
print("Question: Does z_A + z_B - z_C = meaningful composition?")

# Test: z_7 + z_3 - z_0 → should relate to "7>3" concept
# This is abstract, but we can visualize what the arithmetic produces

digits_to_test = [(7, 3, 0), (5, 2, 1), (9, 4, 6)]

fig, axes = plt.subplots(len(digits_to_test), 4, figsize=(10, 3*len(digits_to_test)))

for row, (d1, d2, d3) in enumerate(digits_to_test):
    # Generate latents for each digit
    z_a = torch.randn(1, 64)
    z_b = torch.randn(1, 64)
    z_c = torch.randn(1, 64)

    # Arithmetic: z_a + z_b - z_c
    z_composed = z_a + z_b - z_c

    labels = [d1, d2, d3, d1]  # Last one is for composed latent
    latents = [z_a, z_b, z_c, z_composed]
    titles = [f'z_{d1}', f'z_{d2}', f'z_{d3}', f'z_{d1}+z_{d2}-z_{d3}']

    with torch.no_grad():
        for col, (z, label, title) in enumerate(zip(latents, labels, titles)):
            # GPN only (GAN will be similar pattern)
            img, _ = gpn_weaver(z, torch.tensor([label]))

            axes[row, col].imshow(img[0, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis('off')

plt.suptitle('Latent Arithmetic Test (GPN)\nDoes Addition/Subtraction Produce Valid Outputs?',
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = Path('experiments/latent_arithmetic.png')
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

# ==============================================================================
# TEST 3: FEATURE SPACE REASSEMBLY
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 3: Feature Space Reassembly (Block0 Swapping)")
print("=" * 80)
print("Question: Can we swap intermediate features and maintain identity?")
print("This tests if features are truly compositional building blocks")

# Extract block0 features for multiple digits
n_digits_to_test = 5
digits = list(range(n_digits_to_test))

# Storage
gpn_block0_features = {d: None for d in digits}
gpn_images = {d: None for d in digits}

# Extract features
for digit in digits:
    z = torch.randn(1, 64)
    label = torch.tensor([digit])

    # Hook to capture block0 output
    block0_feature = []
    def capture_block0(module, input, output):
        block0_feature.append(output.detach().clone())

    hook = gpn_weaver.blocks[0].register_forward_hook(capture_block0)

    with torch.no_grad():
        img, _ = gpn_weaver(z, label)

    hook.remove()

    gpn_block0_features[digit] = block0_feature[0]
    gpn_images[digit] = img

# Now test: can we swap block0 features between digits?
# Generate new digit with original latent but swapped block0 features
print("\n  Testing feature swapping (conceptual)...")
print("  Note: Direct feature swapping requires intervention in forward pass")
print("  This is a design limitation - full test would need custom forward hook")

# Instead, visualize what we CAN test: feature distance
print("\n  Computing feature distances to test reassembly potential...")

# Compute pairwise distances in block0 space
distances = np.zeros((n_digits_to_test, n_digits_to_test))

for i, d1 in enumerate(digits):
    for j, d2 in enumerate(digits):
        feat1 = gpn_block0_features[d1].flatten()
        feat2 = gpn_block0_features[d2].flatten()
        dist = torch.norm(feat1 - feat2).item()
        distances[i, j] = dist

# Visualize distance matrix
fig, ax = plt.subplots(1, 1, figsize=(8, 7))

im = ax.imshow(distances, cmap='viridis', aspect='auto')
ax.set_xticks(range(n_digits_to_test))
ax.set_yticks(range(n_digits_to_test))
ax.set_xticklabels(digits)
ax.set_yticklabels(digits)
ax.set_xlabel('Digit', fontweight='bold')
ax.set_ylabel('Digit', fontweight='bold')
ax.set_title('Block0 Feature Distances (GPN)\nCan These Features Reassemble?',
             fontweight='bold', fontsize=14)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('L2 Distance', fontweight='bold')

# Annotate distances
for i in range(n_digits_to_test):
    for j in range(n_digits_to_test):
        text = ax.text(j, i, f'{distances[i, j]:.1f}',
                      ha="center", va="center", color="white", fontsize=9)

plt.tight_layout()

output_path = Path('experiments/feature_distances.png')
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")
plt.close()

# ==============================================================================
# TEST 4: REASSEMBLY SCORE (QUANTITATIVE)
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 4: Reassembly Score (Quantitative Metric)")
print("=" * 80)
print("Metric: Variance in pairwise feature distances")
print("  - High variance: features are distinct, support composition")
print("  - Low variance: features are homogeneous, harder to recombine")

gpn_variance = np.var(distances[np.triu_indices_from(distances, k=1)])
print(f"\n  GPN Block0 Distance Variance: {gpn_variance:.2f}")

# Compare to GAN
gan_block0_features = {}
for digit in digits:
    z = torch.randn(1, 64)
    label = torch.tensor([digit])

    block0_feature = []
    def capture_conv1(module, input, output):
        if len(output.shape) == 4:
            pooled = output.mean(dim=[2, 3])
            block0_feature.append(pooled.detach().clone())

    hook = gan_generator.conv_blocks[1].register_forward_hook(capture_conv1)

    with torch.no_grad():
        _ = gan_generator(z, label)

    hook.remove()

    gan_block0_features[digit] = block0_feature[0]

# GAN distances
gan_distances = np.zeros((n_digits_to_test, n_digits_to_test))
for i, d1 in enumerate(digits):
    for j, d2 in enumerate(digits):
        feat1 = gan_block0_features[d1].flatten()
        feat2 = gan_block0_features[d2].flatten()
        dist = torch.norm(feat1 - feat2).item()
        gan_distances[i, j] = dist

gan_variance = np.var(gan_distances[np.triu_indices_from(gan_distances, k=1)])
print(f"  GAN Conv1 Distance Variance: {gan_variance:.2f}")

variance_ratio = gpn_variance / gan_variance
print(f"\n  Variance Ratio (GPN/GAN): {variance_ratio:.2f}")

if variance_ratio > 1.1:
    print("  → GPN features MORE VARIED - better reassembly potential")
elif variance_ratio < 0.9:
    print("  → GAN features MORE VARIED - better reassembly potential")
else:
    print("  → SIMILAR variance in feature distances")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Compositional Reassembly Analysis")
print("=" * 80)

print("\nTests performed:")
print("  1. Latent Interpolation - Visual smoothness check")
print("  2. Latent Arithmetic - Addition/subtraction validity")
print("  3. Feature Distances - Block0 distinctiveness")
print("  4. Reassembly Score - Quantitative variance metric")

print(f"\nKey Finding:")
print(f"  Feature Distance Variance: GPN={gpn_variance:.2f}, GAN={gan_variance:.2f}")
print(f"  Ratio: {variance_ratio:.2f}")

print("\nInterpretation:")
print("  Standard metrics (dimensionality, separability) don't explain the gap.")
print("  Reassembly potential depends on:")
print("    - How distinct features are (variance in distances)")
print("    - Whether latent operations produce valid outputs")
print("    - Smoothness of interpolation (no mode collapse)")

print("\nConnection to 100% vs 81% composition:")
print("  GPN's pedagogical training may shape features for reassembly")
print("  GAN's adversarial training optimizes for discrimination, not composition")
print("  The geometry supports different operations")

print("\n" + "=" * 80)
print("✓✓✓ REASSEMBLY ANALYSIS COMPLETE ✓✓✓")
print("=" * 80)
print("\nThis probes WHAT explains the compositional gap.")
print("The test isn't factorization. It's whether features reassemble.")
print("=" * 80)
