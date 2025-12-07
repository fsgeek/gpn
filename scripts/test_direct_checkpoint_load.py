#!/usr/bin/env python3
"""
Test loading checkpoints directly using the pattern from test_digit_primitives.py
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.weaver import create_weaver
from models.acgan import ACGANGenerator

print("=" * 80)
print("TEST: Direct Checkpoint Loading (using create_weaver pattern)")
print("=" * 80)

device = torch.device('cpu')

# Test GPN
print("\n1. Loading GPN checkpoint...")
checkpoint_path = Path('checkpoints/checkpoint_final.pt')
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

weaver = create_weaver(latent_dim=64, num_classes=10)
weaver.load_state_dict(ckpt['models']['weaver'])
weaver.eval()

print("   ✓ GPN Weaver loaded")

# Test generation
z = torch.randn(4, 64)
labels = torch.tensor([0, 1, 2, 3])

with torch.no_grad():
    images, v_pred = weaver(z, labels)

print(f"   Generated images: {images.shape}")
print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")

# Test AC-GAN
print("\n2. Loading AC-GAN checkpoint...")
acgan_path = Path('checkpoints/acgan_final.pt')
acgan_ckpt = torch.load(acgan_path, map_location=device, weights_only=False)

generator = ACGANGenerator(latent_dim=64, num_classes=10)
generator.load_state_dict(acgan_ckpt['models']['generator'])
generator.eval()

print("   ✓ AC-GAN Generator loaded")

# Test generation
with torch.no_grad():
    images_gan = generator(z, labels)

print(f"   Generated images: {images_gan.shape}")
print(f"   Image range: [{images_gan.min():.3f}, {images_gan.max():.3f}]")

# Now test feature extraction
print("\n3. Testing feature extraction...")

gpn_features = {}
gan_features = {}

def get_activation(feature_dict, name):
    def hook(model, input, output):
        feature_dict[name] = output.detach()
    return hook

# Register hooks on GPN
gpn_hooks = []
gpn_hooks.append(weaver.fc.register_forward_hook(get_activation(gpn_features, 'fc')))
gpn_hooks.append(weaver.blocks[0].register_forward_hook(get_activation(gpn_features, 'block0')))

# Register hooks on GAN
gan_hooks = []
gan_hooks.append(generator.fc[0].register_forward_hook(get_activation(gan_features, 'fc')))

# Forward passes
with torch.no_grad():
    _ = weaver(z, labels)
    _ = generator(z, labels)

# Remove hooks
for hook in gpn_hooks:
    hook.remove()
for hook in gan_hooks:
    hook.remove()

print("\n   GPN features:")
for name, feat in gpn_features.items():
    print(f"     {name}: {feat.shape} | mean={feat.mean():.3f}, std={feat.std():.3f}")

print("\n   GAN features:")
for name, feat in gan_features.items():
    print(f"     {name}: {feat.shape} | mean={feat.mean():.3f}, std={feat.std():.3f}")

print("\n" + "=" * 80)
print("✓✓✓ FOUNDATION COMPLETELY PROVEN ✓✓✓")
print("=" * 80)
print("\nWe can:")
print("  1. ✓ Load GPN weaver from checkpoint")
print("  2. ✓ Load AC-GAN generator from checkpoint")
print("  3. ✓ Generate images from both")
print("  4. ✓ Extract intermediate features from both")
print("\nEverything needed for Checkpoint 1 analysis works.")
print("Ready to build the full analysis pipeline on this proven ground.")
print("=" * 80)
