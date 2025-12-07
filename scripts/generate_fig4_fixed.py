#!/usr/bin/env python3
"""
Generate Figure 4: Fidelity Comparison (Fixed Layout)

Fix the overlapping text boxes at the bottom.
"""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.weaver import create_weaver
from models.acgan import ACGANGenerator

print("Generating Figure 4: Fidelity Comparison (Fixed)")

device = torch.device('cpu')

# Load models
gpn_ckpt = torch.load('checkpoints/checkpoint_final.pt', map_location=device, weights_only=False)
gpn_weaver = create_weaver(latent_dim=64, num_classes=10)
gpn_weaver.load_state_dict(gpn_ckpt['models']['weaver'])
gpn_weaver.eval()

gan_ckpt = torch.load('checkpoints/acgan_final.pt', map_location=device, weights_only=False)
gan_generator = ACGANGenerator(latent_dim=64, num_classes=10)
gan_generator.load_state_dict(gan_ckpt['models']['generator'])
gan_generator.eval()

# Generate 10 samples per digit for both models
n_samples = 10
digits = list(range(10))

gpn_samples = []
gan_samples = []

with torch.no_grad():
    for digit in digits:
        digit_gpn = []
        digit_gan = []

        for _ in range(n_samples):
            z = torch.randn(1, 64)
            label = torch.tensor([digit])

            # GPN
            gpn_img, _ = gpn_weaver(z, label)
            digit_gpn.append(gpn_img)

            # GAN
            gan_img = gan_generator(z, label)
            digit_gan.append(gan_img)

        gpn_samples.append(digit_gpn)
        gan_samples.append(digit_gan)

print(f"Generated {len(digits)} x {n_samples} samples for each model")

# Create figure with better spacing
fig = plt.figure(figsize=(16, 10))

# Title at top
fig.suptitle('The Fidelity Trap: Visual Quality ≠ Compositional Capacity',
             fontsize=20, fontweight='bold', y=0.98)

# Create grid: left (GAN), right (Pedagogical)
# Each side: 10 rows (digits) x 10 cols (samples)

# GAN side
ax_gan_title = plt.subplot2grid((14, 2), (0, 0), colspan=1)
ax_gan_title.text(0.5, 0.5, 'Adversarial (GAN) - High Fidelity',
                  ha='center', va='center', fontsize=16, fontweight='bold')
ax_gan_title.axis('off')

# GPN side
ax_gpn_title = plt.subplot2grid((14, 2), (0, 1), colspan=1)
ax_gpn_title.text(0.5, 0.5, 'Pedagogical (Ours) - Topological',
                  ha='center', va='center', fontsize=16, fontweight='bold')
ax_gpn_title.axis('off')

# GAN grid (rows 1-10)
ax_gan = plt.subplot2grid((14, 2), (1, 0), rowspan=10, colspan=1)

gan_grid = []
for digit_samples in gan_samples:
    row = torch.cat(digit_samples, dim=3)  # Concatenate horizontally
    gan_grid.append(row)
gan_grid = torch.cat(gan_grid, dim=2)  # Stack vertically

ax_gan.imshow(gan_grid[0, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
ax_gan.axis('off')

# GPN grid (rows 1-10)
ax_gpn = plt.subplot2grid((14, 2), (1, 1), rowspan=10, colspan=1)

gpn_grid = []
for digit_samples in gpn_samples:
    row = torch.cat(digit_samples, dim=3)
    gpn_grid.append(row)
gpn_grid = torch.cat(gpn_grid, dim=2)

ax_gpn.imshow(gpn_grid[0, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
ax_gpn.axis('off')

# Description boxes (row 11) - with more vertical space
ax_gan_desc = plt.subplot2grid((14, 2), (11, 0), colspan=1)
ax_gan_desc.text(0.5, 0.5,
                 'Sharp strokes, realistic variation\nOptimized for visual quality',
                 ha='center', va='center', fontsize=11, style='italic',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black', linewidth=2))
ax_gan_desc.set_xlim(0, 1)
ax_gan_desc.set_ylim(0, 1)
ax_gan_desc.axis('off')

ax_gpn_desc = plt.subplot2grid((14, 2), (11, 1), colspan=1)
ax_gpn_desc.text(0.5, 0.5,
                 'Blobby features, simplified structure\nOptimized for compositional transfer',
                 ha='center', va='center', fontsize=11, style='italic',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black', linewidth=2))
ax_gpn_desc.set_xlim(0, 1)
ax_gpn_desc.set_ylim(0, 1)
ax_gpn_desc.axis('off')

# Empty row for spacing (row 12)
ax_space = plt.subplot2grid((14, 2), (12, 0), colspan=2)
ax_space.axis('off')

# Results box (row 13) - now with clear separation
ax_results = plt.subplot2grid((14, 2), (13, 0), colspan=2)
ax_results.text(0.5, 0.5,
                'Phase 1.6 Hold-out Transfer Results: GAN 81.1% | Ours 100.0% (Δ = +18.9%)',
                ha='center', va='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='yellow', edgecolor='black', linewidth=2))
ax_results.set_xlim(0, 1)
ax_results.set_ylim(0, 1)
ax_results.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
output_path = Path('docs/external/blog/assets/img/fig4_fidelity_comparison.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')

print(f"✓ Saved: {output_path}")
print("Fixed: No more overlap between description boxes and results box")
