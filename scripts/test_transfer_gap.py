#!/usr/bin/env python3
"""
Test the 10% transfer gap hypothesis:
Does classification grounding produce more transferable representations than discrimination grounding?

Tests V3, V3-no-meta, and GAN against a fresh classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.models.weaver import create_weaver
from src.models.baseline_gan import create_baseline_gan
from src.training.config import TrainingConfig


class FreshClassifier(nn.Module):
    """A different architecture than Judge - simple LeNet-style."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_fresh_classifier(epochs: int = 3, seed: int = 42):
    """Train a fresh classifier on MNIST."""
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    model = FreshClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

    # Final test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return model, 100*correct/total


def test_generator(classifier, generator, config, num_samples: int = 1000):
    """Test a generator against the classifier."""
    classifier.eval()
    generator.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, config.latent_dim)
        labels = torch.randint(0, 10, (num_samples,))

        output = generator(z, labels)
        if isinstance(output, tuple):
            images = output[0]
        else:
            images = output

        logits = classifier(images)
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()

    return 100 * correct / num_samples


def load_weaver(checkpoint_path: str, config: TrainingConfig):
    """Load a Weaver from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    weaver = create_weaver(latent_dim=config.latent_dim, v_pred_dim=config.weaver.v_pred_dim, device='cpu')
    weaver.load_state_dict(checkpoint['models']['weaver'])
    return weaver


def load_gan(checkpoint_path: str, config: TrainingConfig):
    """Load a GAN generator from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    generator, _ = create_baseline_gan(latent_dim=config.latent_dim, device='cpu')
    generator.load_state_dict(checkpoint['models']['generator'])
    return generator


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=5, help='Number of random seeds to test')
    parser.add_argument('--samples', type=int, default=1000, help='Samples per test')
    args = parser.parse_args()

    config = TrainingConfig()

    # Try to load models
    models = {}

    # V3 with meta-learning
    try:
        models['V3 (meta-learning)'] = load_weaver('checkpoints/checkpoint_v3_final.pt', config)
    except FileNotFoundError:
        try:
            models['V3 (meta-learning)'] = load_weaver('checkpoints/checkpoint_v3_step2000.pt', config)
        except FileNotFoundError:
            print("V3 checkpoint not found")

    # V3 without meta-learning
    try:
        models['V3-no-meta'] = load_weaver('checkpoints/checkpoint_v3nometa_step1000.pt', config)
    except FileNotFoundError:
        print("V3-no-meta checkpoint not found")

    # GAN
    try:
        models['GAN'] = load_gan('checkpoints/gan_checkpoint_gan_final.pt', config)
    except FileNotFoundError:
        try:
            models['GAN'] = load_gan('checkpoints/gan_checkpoint_step5000.pt', config)
        except FileNotFoundError:
            print("GAN checkpoint not found")

    if not models:
        print("No models found to test!")
        return

    print(f"\nTesting {len(models)} models across {args.seeds} random seeds")
    print("="*60)

    results = {name: [] for name in models}

    for seed in range(args.seeds):
        print(f"\n--- Seed {seed} ---")

        # Train fresh classifier with this seed
        classifier, mnist_acc = train_fresh_classifier(epochs=3, seed=seed)
        print(f"Fresh classifier MNIST accuracy: {mnist_acc:.1f}%")

        # Test each model
        for name, model in models.items():
            torch.manual_seed(seed + 1000)  # Different seed for generation
            acc = test_generator(classifier, model, config, num_samples=args.samples)
            results[name].append(acc)
            print(f"  {name}: {acc:.1f}%")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Cross-Architecture Transfer")
    print("="*60)

    for name, accs in results.items():
        mean = sum(accs) / len(accs)
        std = (sum((a - mean)**2 for a in accs) / len(accs)) ** 0.5
        print(f"{name:20s}: {mean:.1f}% +/- {std:.1f}%")

    # Compare classification-grounded vs discrimination-grounded
    classification_grounded = [name for name in results if 'V3' in name]
    discrimination_grounded = [name for name in results if 'GAN' in name]

    if classification_grounded and discrimination_grounded:
        class_mean = sum(sum(results[n]) for n in classification_grounded) / sum(len(results[n]) for n in classification_grounded)
        disc_mean = sum(sum(results[n]) for n in discrimination_grounded) / sum(len(results[n]) for n in discrimination_grounded)

        print("\n" + "-"*60)
        print(f"Classification-grounded (V3 variants): {class_mean:.1f}%")
        print(f"Discrimination-grounded (GAN):         {disc_mean:.1f}%")
        print(f"Transfer gap:                          {class_mean - disc_mean:.1f}%")
        print("-"*60)

        if class_mean - disc_mean > 5:
            print("\nFINDING: Classification grounding produces more transferable")
            print("representations than discrimination grounding.")
        else:
            print("\nNo significant transfer gap detected.")


if __name__ == "__main__":
    main()
