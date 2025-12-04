#!/usr/bin/env python3
"""
Test generators against a fresh classifier that wasn't in anyone's training loop.

This tests whether V3's advantage is "learned what Judge likes" vs "learned to be clear."
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
        # Different architecture than Judge (which uses larger hidden dims)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28->14
        x = self.pool(F.relu(self.conv2(x)))  # 14->7
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_fresh_classifier(epochs: int = 3):
    """Train a fresh classifier on MNIST."""
    print("Training fresh classifier on MNIST...")

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
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        # Test accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        print(f"Epoch {epoch+1}: Test accuracy = {100*correct/total:.2f}%")

    return model


def test_generator(classifier, generator, config, num_samples: int = 1000, name: str = "Generator"):
    """Test a generator against the classifier."""
    classifier.eval()
    generator.eval()

    correct = 0
    total = num_samples

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

    accuracy = 100 * correct / total
    print(f"{name}: {accuracy:.2f}% accuracy on fresh classifier")
    return accuracy


def main():
    config = TrainingConfig()

    # Train fresh classifier
    fresh_classifier = train_fresh_classifier(epochs=3)

    # Load V3 Weaver
    print("\nLoading V3 (Meta-Learning) model...")
    v3_checkpoint = torch.load('checkpoints/checkpoint_v3_final.pt', map_location='cpu', weights_only=False)
    v3_weaver = create_weaver(latent_dim=config.latent_dim, v_pred_dim=config.weaver.v_pred_dim, device='cpu')
    v3_weaver.load_state_dict(v3_checkpoint['models']['weaver'])

    # Load GAN Generator
    print("Loading GAN (Adversarial) model...")
    gan_checkpoint = torch.load('checkpoints/gan_checkpoint_gan_final.pt', map_location='cpu', weights_only=False)
    gan_generator, _ = create_baseline_gan(latent_dim=config.latent_dim, device='cpu')
    gan_generator.load_state_dict(gan_checkpoint['models']['generator'])

    # Test both
    print("\n" + "="*50)
    print("TESTING ON FRESH CLASSIFIER (not in training loop)")
    print("="*50)

    v3_acc = test_generator(fresh_classifier, v3_weaver, config, num_samples=1000, name="V3 (Meta-Learning)")
    gan_acc = test_generator(fresh_classifier, gan_generator, config, num_samples=1000, name="GAN (Adversarial)")

    print("\n" + "="*50)
    print(f"RESULT: V3={v3_acc:.2f}%, GAN={gan_acc:.2f}%, Difference={v3_acc-gan_acc:.2f}%")
    print("="*50)

    if v3_acc > gan_acc + 5:
        print("V3's clarity advantage GENERALIZES to fresh classifier")
    elif gan_acc > v3_acc + 5:
        print("V3's advantage was specific to Judge - doesn't generalize")
    else:
        print("Results are similar - no clear winner on fresh classifier")


if __name__ == "__main__":
    main()
