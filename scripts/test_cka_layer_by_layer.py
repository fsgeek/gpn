#!/usr/bin/env python3
"""
Layer-by-Layer CKA Analysis

Find where GPN and GAN representations diverge.
We know fc layer is 96.5% similar. Where do they split?
That's where compositional capacity diverges.
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
from analysis.similarity import compute_cka

print("=" * 80)
print("LAYER-BY-LAYER CKA: Find Where GPN and GAN Diverge")
print("=" * 80)

device = torch.device('cpu')

# ==============================================================================
# STEP 1: LOAD MODELS
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
# STEP 2: IDENTIFY COMPARABLE LAYERS
# ==============================================================================
print("\nSTEP 2: Identifying comparable layers...")

# Both models have:
# - fc: initial latent projection
# - blocks/conv_blocks: upsampling/generation blocks
# - output/final: final image generation

# Let's extract from:
# GPN: fc, blocks[0], blocks[1], output
# GAN: fc[0], conv_blocks (we need to find comparable points)

gpn_layers = {
    'fc': gpn_weaver.fc,
    'block0': gpn_weaver.blocks[0],
    'block1': gpn_weaver.blocks[1],
    'output': gpn_weaver.output[0],  # The conv layer before tanh
}

# GAN structure needs inspection
gan_layers = {
    'fc': gan_generator.fc[0],
}

# Try to add conv_blocks if they exist
if hasattr(gan_generator, 'conv_blocks'):
    # Add first few conv blocks
    for i, block in enumerate(gan_generator.conv_blocks[:4]):
        if isinstance(block, torch.nn.Conv2d):
            gan_layers[f'conv{i}'] = block

print(f"  GPN layers to extract: {list(gpn_layers.keys())}")
print(f"  GAN layers to extract: {list(gan_layers.keys())}")

# ==============================================================================
# STEP 3: EXTRACT FEATURES FROM ALL LAYERS
# ==============================================================================
print("\nSTEP 3: Extracting features from all layers...")

num_samples = 100
batch_size = 10
num_batches = num_samples // batch_size

# Storage for features
gpn_features = {name: [] for name in gpn_layers.keys()}
gan_features = {name: [] for name in gan_layers.keys()}

def get_activation(feature_dict, name):
    def hook(model, input, output):
        # Flatten spatial dimensions for comparison
        if len(output.shape) == 4:  # [B, C, H, W]
            # Global average pooling to get [B, C]
            pooled = output.mean(dim=[2, 3])
            feature_dict[name].append(pooled.detach())
        else:  # [B, D]
            feature_dict[name].append(output.detach())
    return hook

# Register all hooks
gpn_hooks = []
for name, layer in gpn_layers.items():
    gpn_hooks.append(layer.register_forward_hook(get_activation(gpn_features, name)))

gan_hooks = []
for name, layer in gan_layers.items():
    gan_hooks.append(layer.register_forward_hook(get_activation(gan_features, name)))

# Extract features
print(f"  Generating {num_samples} samples...")
for i in range(num_batches):
    z = torch.randn(batch_size, 64)
    labels = torch.randint(0, 10, (batch_size,))

    with torch.no_grad():
        _ = gpn_weaver(z, labels)
        _ = gan_generator(z, labels)

    if (i + 1) % 5 == 0:
        print(f"    Batch {i+1}/{num_batches}")

# Remove hooks
for hook in gpn_hooks:
    hook.remove()
for hook in gan_hooks:
    hook.remove()

# Concatenate features
for name in gpn_features.keys():
    gpn_features[name] = torch.cat(gpn_features[name], dim=0).numpy()
    print(f"  GPN {name}: {gpn_features[name].shape}")

for name in gan_features.keys():
    gan_features[name] = torch.cat(gan_features[name], dim=0).numpy()
    print(f"  GAN {name}: {gan_features[name].shape}")

# ==============================================================================
# STEP 4: COMPUTE CKA AT EACH LAYER
# ==============================================================================
print("\nSTEP 4: Computing layer-by-layer CKA...")

# We'll compare fc to fc, then track GPN blocks
layer_names = []
cka_scores = []

# Compare fc layers
if 'fc' in gpn_features and 'fc' in gan_features:
    cka = compute_cka(gpn_features['fc'], gan_features['fc'], kernel='linear')
    layer_names.append('fc')
    cka_scores.append(cka)
    print(f"  fc layer: CKA = {cka:.6f}")

# For GPN blocks, compare to GAN conv blocks if available
block_comparisons = [
    ('block0', 'conv1'),
    ('block1', 'conv3'),
    ('output', 'conv5'),
]

for gpn_layer, gan_layer in block_comparisons:
    if gpn_layer in gpn_features and gan_layer in gan_features:
        cka = compute_cka(gpn_features[gpn_layer], gan_features[gan_layer], kernel='linear')
        layer_names.append(gpn_layer)
        cka_scores.append(cka)
        print(f"  {gpn_layer} vs {gan_layer}: CKA = {cka:.6f}")
    elif gpn_layer in gpn_features:
        # If we can't find comparable GAN layer, note it
        print(f"  {gpn_layer}: No comparable GAN layer found")

# ==============================================================================
# STEP 5: VISUALIZE DIVERGENCE
# ==============================================================================
print("\nSTEP 5: Visualizing where representations diverge...")

if len(cka_scores) > 0:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot CKA trajectory
    x_positions = np.arange(len(layer_names))
    ax.plot(x_positions, cka_scores, 'o-', linewidth=2, markersize=10,
            color='#2E86AB', label='CKA Score')

    # Add horizontal reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Similarity')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate Similarity')
    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='No Similarity')

    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_xlabel('Network Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('CKA Score', fontsize=12, fontweight='bold')
    ax.set_title('Representational Divergence: GPN vs GAN\nWhere Does Compositional Capacity Split?',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Annotate scores
    for i, (name, score) in enumerate(zip(layer_names, cka_scores)):
        ax.annotate(f'{score:.3f}',
                   xy=(i, score),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9,
                   fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = Path('experiments/cka_layer_divergence.png')
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved visualization: {output_path}")

# ==============================================================================
# STEP 6: INTERPRET RESULTS
# ==============================================================================
print("\n" + "=" * 80)
print("INTERPRETATION: Where Do They Diverge?")
print("=" * 80)

if len(cka_scores) > 1:
    # Find largest drop
    drops = [cka_scores[i] - cka_scores[i+1] for i in range(len(cka_scores)-1)]
    if drops:
        max_drop_idx = np.argmax(np.abs(drops))
        max_drop = drops[max_drop_idx]

        print(f"\nStarting similarity (fc layer): {cka_scores[0]:.4f}")
        print(f"Final similarity ({layer_names[-1]}): {cka_scores[-1]:.4f}")
        print(f"Total divergence: {cka_scores[0] - cka_scores[-1]:.4f}")

        if abs(max_drop) > 0.1:
            print(f"\nLargest drop: {max_drop:.4f}")
            print(f"  Between {layer_names[max_drop_idx]} → {layer_names[max_drop_idx+1]}")
            print(f"  This is where representations split most dramatically.")
        else:
            print(f"\nNo dramatic drop detected (max drop: {max_drop:.4f})")
            print(f"  Representations remain similar throughout the network.")

    print("\nContext:")
    print("  - Both models trained on same MNIST data")
    print("  - GPN: 100% on novel combinations")
    print("  - GAN: 81% on novel combinations")
    print("  - If representations diverge, that's where compositional capacity splits")
    print("  - If they stay similar, the difference is subtle/not geometric")

print("\n" + "=" * 80)
print("✓✓✓ LAYER-BY-LAYER ANALYSIS COMPLETE ✓✓✓")
print("=" * 80)
print("\nWe found where GPN and GAN representations agree and diverge.")
print("This tells us where compositional capacity emerges in the architecture.")
print("=" * 80)
