#!/usr/bin/env python3
"""
Even Simpler POC: Can we extract features from a model AT ALL?

Don't worry about loading checkpoints yet.
Just prove we can extract features.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

print("=" * 80)
print("SIMPLEST POSSIBLE TEST: Feature Extraction")
print("=" * 80)

# Create a trivial model
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.layer3 = nn.Linear(30, 5)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = TinyModel()
print("\n✓ Created tiny model")

# Generate input
x = torch.randn(8, 10)
print(f"✓ Created input: {x.shape}")

# Extract features using hooks
features = {}

def get_activation(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

# Register hooks
hooks = []
hooks.append(model.layer1.register_forward_hook(get_activation('layer1')))
hooks.append(model.layer2.register_forward_hook(get_activation('layer2')))
hooks.append(model.layer3.register_forward_hook(get_activation('layer3')))

print("✓ Registered hooks")

# Forward pass
with torch.no_grad():
    output = model(x)

print(f"✓ Forward pass: output shape {output.shape}")

# Remove hooks
for hook in hooks:
    hook.remove()

# Check features
print("\nExtracted features:")
for name, feat in features.items():
    print(f"  {name}: {feat.shape} | mean={feat.mean():.3f}, std={feat.std():.3f}")

print("\n" + "=" * 80)
print("✓✓✓ FEATURE EXTRACTION WORKS ✓✓✓")
print("=" * 80)
print("\nNow we know we can extract features from ANY PyTorch model.")
print("Next step: Load actual GPN/GAN checkpoints.")
print("=" * 80)
