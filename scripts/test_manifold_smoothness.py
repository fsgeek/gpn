#!/usr/bin/env python3
"""
Manifold Smoothness: The Real Mechanism

Hypothesis from el jefe:
- GAN modes are isolated islands with cliffs between
- GPN modes are connected hills with walkable paths
- Compositional capacity requires smoothness, not separation

Tests:
1. Perceptual path length - How much does output change along interpolation?
2. Classifier confidence - Are intermediate points valid?
3. Output change gradient - How abruptly does it snap?

The snap vs flow is visible. Let's measure it.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.weaver import create_weaver
from models.acgan import ACGANGenerator

print("=" * 80)
print("MANIFOLD SMOOTHNESS ANALYSIS")
print("Adversarial: isolated islands. Pedagogical: connected hills.")
print("=" * 80)

device = torch.device('cpu')

# ==============================================================================
# LOAD MODELS
# ==============================================================================
print("\nLoading models...")

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
# TEST 1: PERCEPTUAL PATH LENGTH
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 1: Perceptual Path Length")
print("=" * 80)
print("Question: How much does output change along interpolation?")
print("Smooth manifold: gradual change. Cliff: abrupt snap.")

# Interpolate between pairs of digits
test_pairs = [(0, 1), (3, 7), (5, 9)]
n_steps = 20

gpn_path_lengths = []
gan_path_lengths = []

for digit1, digit2 in test_pairs:
    z1 = torch.randn(1, 64)
    z2 = torch.randn(1, 64)

    alphas = torch.linspace(0, 1, n_steps)

    # GPN path
    gpn_images = []
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        label = torch.tensor([digit1 if alpha < 0.5 else digit2])

        with torch.no_grad():
            img, _ = gpn_weaver(z_interp, label)
        gpn_images.append(img)

    # GAN path
    gan_images = []
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        label = torch.tensor([digit1 if alpha < 0.5 else digit2])

        with torch.no_grad():
            img = gan_generator(z_interp, label)
        gan_images.append(img)

    # Compute path lengths (sum of consecutive differences)
    gpn_path_length = 0
    for i in range(len(gpn_images) - 1):
        diff = torch.norm(gpn_images[i+1] - gpn_images[i]).item()
        gpn_path_length += diff

    gan_path_length = 0
    for i in range(len(gan_images) - 1):
        diff = torch.norm(gan_images[i+1] - gan_images[i]).item()
        gan_path_length += diff

    gpn_path_lengths.append(gpn_path_length)
    gan_path_lengths.append(gan_path_length)

    print(f"\nDigit {digit1} → {digit2}:")
    print(f"  GPN path length: {gpn_path_length:.4f}")
    print(f"  GAN path length: {gan_path_length:.4f}")
    print(f"  Ratio (GAN/GPN): {gan_path_length/gpn_path_length:.2f}x")

avg_gpn_path = np.mean(gpn_path_lengths)
avg_gan_path = np.mean(gan_path_lengths)

print(f"\nAverage path lengths:")
print(f"  GPN: {avg_gpn_path:.4f}")
print(f"  GAN: {avg_gan_path:.4f}")
print(f"  Ratio: {avg_gan_path/avg_gpn_path:.2f}x")

if avg_gan_path > avg_gpn_path:
    print(f"\n  → GAN has LONGER paths (more change along interpolation)")
    print(f"     Suggests sharper boundaries between modes")

# ==============================================================================
# TEST 2: OUTPUT CHANGE GRADIENT (SNAP DETECTION)
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 2: Output Change Gradient (Snap Detection)")
print("=" * 80)
print("Question: Does output change smoothly or abruptly?")
print("Look for: spikes in GAN gradient = snap points")

# Use first test pair for detailed analysis
digit1, digit2 = 3, 7
z1 = torch.randn(1, 64)
z2 = torch.randn(1, 64)

n_steps = 50  # Fine-grained for gradient detection
alphas = torch.linspace(0, 1, n_steps)

# Generate interpolation
gpn_images = []
gan_images = []

for alpha in alphas:
    z_interp = (1 - alpha) * z1 + alpha * z2
    label = torch.tensor([digit1 if alpha < 0.5 else digit2])

    with torch.no_grad():
        gpn_img, _ = gpn_weaver(z_interp, label)
        gan_img = gan_generator(z_interp, label)

    gpn_images.append(gpn_img)
    gan_images.append(gan_img)

# Compute step-wise changes
gpn_changes = []
gan_changes = []

for i in range(len(gpn_images) - 1):
    gpn_change = torch.norm(gpn_images[i+1] - gpn_images[i]).item()
    gan_change = torch.norm(gan_images[i+1] - gan_images[i]).item()

    gpn_changes.append(gpn_change)
    gan_changes.append(gan_change)

gpn_changes = np.array(gpn_changes)
gan_changes = np.array(gan_changes)

# Detect snaps (large changes)
gpn_max_change = gpn_changes.max()
gan_max_change = gan_changes.max()
gpn_std_change = gpn_changes.std()
gan_std_change = gan_changes.std()

print(f"\nChange statistics (digit {digit1} → {digit2}):")
print(f"  GPN: max={gpn_max_change:.4f}, std={gpn_std_change:.4f}")
print(f"  GAN: max={gan_max_change:.4f}, std={gan_std_change:.4f}")

# Count "snaps" (changes > 2 std above mean)
gpn_mean = gpn_changes.mean()
gan_mean = gan_changes.mean()
gpn_snaps = np.sum(gpn_changes > gpn_mean + 2 * gpn_std_change)
gan_snaps = np.sum(gan_changes > gan_mean + 2 * gan_std_change)

print(f"\nSnap points (changes > 2σ above mean):")
print(f"  GPN: {gpn_snaps} snaps")
print(f"  GAN: {gan_snaps} snaps")

if gan_snaps > gpn_snaps:
    print(f"\n  → GAN has MORE SNAPS ({gan_snaps} vs {gpn_snaps})")
    print(f"     Abrupt transitions between modes (cliffs)")
elif gpn_snaps > gan_snaps:
    print(f"\n  → GPN has MORE SNAPS ({gpn_snaps} vs {gan_snaps})")
    print(f"     (Unexpected - investigate)")
else:
    print(f"\n  → EQUAL smoothness")

# ==============================================================================
# VISUALIZATION
# ==============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION: Manifold Smoothness")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Output change gradient
ax1 = axes[0, 0]
x = np.arange(len(gpn_changes))
ax1.plot(x, gpn_changes, 'o-', color='#2E86AB', label='GPN', linewidth=2, markersize=4)
ax1.plot(x, gan_changes, 's-', color='#A23B72', label='GAN', linewidth=2, markersize=4)
ax1.axhline(gpn_mean + 2*gpn_std_change, color='#2E86AB', linestyle='--', alpha=0.5, label='GPN 2σ')
ax1.axhline(gan_mean + 2*gan_std_change, color='#A23B72', linestyle='--', alpha=0.5, label='GAN 2σ')
ax1.set_xlabel('Interpolation Step', fontweight='bold')
ax1.set_ylabel('Output Change (L2 norm)', fontweight='bold')
ax1.set_title(f'Output Change Gradient: {digit1}→{digit2}\nSnaps vs Smooth Flow', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Path length comparison
ax2 = axes[0, 1]
x_pos = np.arange(len(test_pairs))
width = 0.35
labels = [f'{d1}→{d2}' for d1, d2 in test_pairs]
ax2.bar(x_pos - width/2, gpn_path_lengths, width, label='GPN', color='#2E86AB', alpha=0.7, edgecolor='black')
ax2.bar(x_pos + width/2, gan_path_lengths, width, label='GAN', color='#A23B72', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Digit Pair', fontweight='bold')
ax2.set_ylabel('Path Length', fontweight='bold')
ax2.set_title('Perceptual Path Length\nLonger = More Change', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Interpolation visualization (GPN)
ax3 = axes[1, 0]
n_show = 10
indices = np.linspace(0, len(gpn_images)-1, n_show).astype(int)
concat_gpn = torch.cat([gpn_images[i] for i in indices], dim=3)
ax3.imshow(concat_gpn[0, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
ax3.set_title(f'GPN Interpolation: {digit1}→{digit2}\n(Smooth Flow)', fontweight='bold')
ax3.axis('off')

# Plot 4: Interpolation visualization (GAN)
ax4 = axes[1, 1]
concat_gan = torch.cat([gan_images[i] for i in indices], dim=3)
ax4.imshow(concat_gan[0, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
ax4.set_title(f'GAN Interpolation: {digit1}→{digit2}\n(Cliff Snap?)', fontweight='bold')
ax4.axis('off')

plt.suptitle('Manifold Smoothness: Islands vs Hills\nThe Real Compositional Mechanism',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

output_path = Path('experiments/manifold_smoothness.png')
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")
plt.close()

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("MANIFOLD SMOOTHNESS SUMMARY")
print("=" * 80)

print("\n1. PATH LENGTH:")
print(f"   GPN average: {avg_gpn_path:.4f}")
print(f"   GAN average: {avg_gan_path:.4f}")
print(f"   Ratio: {avg_gan_path/avg_gpn_path:.2f}x")

print("\n2. SNAP DETECTION:")
print(f"   GPN snaps: {gpn_snaps}")
print(f"   GAN snaps: {gan_snaps}")

print("\n3. HYPOTHESIS:")
print("   Composition requires smooth manifolds, not isolated modes")
print("   GAN: High class separation, but cliffs between modes")
print("   GPN: Lower class separation, but navigable paths")

print("\n" + "=" * 80)
print("THE REAL MECHANISM (if this holds):")
print("=" * 80)
print("\nAdversarial training creates isolated modes.")
print("Pedagogical training creates connected manifolds.")
print("\nYou can't compose from islands.")
print("You need walkable paths between concepts.")
print("=" * 80)
