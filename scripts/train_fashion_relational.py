"""
Train relational model on Fashion-MNIST primitives.

Tests compositional capacity: Can pedagogical primitives compose better than adversarial?

Relation: Arbitrary ordering by label index (0 < 1 < 2 < ... < 9)
Same structure as MNIST relational task for apples-to-apples comparison.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.models.weaver import Weaver
from src.models.baseline_gan import Generator
from src.models.judge import Judge


def get_fashion_mnist_loader(batch_size: int = 64, train: bool = True) -> DataLoader:
    """Get Fashion-MNIST data loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ])

    try:
        dataset = datasets.FashionMNIST(
            root="data",
            train=train,
            download=False,
            transform=transform,
        )
    except RuntimeError:
        dataset = datasets.FashionMNIST(
            root="data",
            train=train,
            download=True,
            transform=transform,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


def train_judge(device: torch.device, epochs: int = 5) -> Judge:
    """Train frozen Judge classifier on Fashion-MNIST."""
    print("Training Judge classifier on Fashion-MNIST...")

    judge = Judge()
    judge.to(device)

    train_loader = get_fashion_mnist_loader(batch_size=128, train=True)
    test_loader = get_fashion_mnist_loader(batch_size=128, train=False)

    optimizer = optim.Adam(judge.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        judge.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = judge(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate
        judge.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = judge(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"  Epoch {epoch+1}/{epochs}: Test Accuracy = {accuracy:.2f}%")

    judge.freeze()
    return judge


class RelationalModel(nn.Module):
    """Simple MLP that learns relational structure from primitive pairs."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()

        # Takes concatenated pair of latent vectors
        self.network = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Predict whether z1 < z2 in the learned ordering."""
        combined = torch.cat([z1, z2], dim=1)
        return self.network(combined).squeeze(-1)


def generate_pairs(
    num_pairs: int,
    train_classes: List[int],
    test_classes: List[int],
    split: str = 'train'
) -> List[Tuple[int, int, int]]:
    """Generate (class1, class2, label) pairs.

    Label: 1 if class1 < class2, 0 otherwise.
    Train: pairs from train_classes only
    Test: pairs involving at least one test_class (holdout compositions)
    """

    pairs = []
    classes = train_classes if split == 'train' else list(range(10))

    while len(pairs) < num_pairs:
        c1, c2 = np.random.choice(classes, size=2, replace=False)

        if split == 'test':
            # Holdout: at least one class must be in test set
            if c1 not in test_classes and c2 not in test_classes:
                continue

        label = 1 if c1 < c2 else 0
        pairs.append((c1, c2, label))

    return pairs


def train_relational_model(
    primitive_checkpoint: Path,
    primitive_type: str,  # 'pedagogical' or 'adversarial'
    train_classes: List[int],
    test_classes: List[int],
    epochs: int = 50,
    batch_size: int = 64,
    latent_dim: int = 64,
    device: str = 'cuda',
) -> Tuple[RelationalModel, dict]:
    """Train relational model on primitive pairs."""

    device = torch.device(device)

    # Load primitive generator
    checkpoint = torch.load(primitive_checkpoint, map_location=device, weights_only=False)

    if primitive_type == 'pedagogical':
        generator = Weaver(latent_dim=latent_dim, num_classes=10)
        generator.load_state_dict(checkpoint['models']['weaver'])
    else:
        generator = Generator(latent_dim=latent_dim, num_classes=10)
        generator.load_state_dict(checkpoint['models']['generator'])

    generator.to(device)
    generator.eval()

    # Train Judge for feature extraction
    judge = train_judge(device, epochs=5)
    judge.eval()

    # Detect Judge feature dimension
    with torch.no_grad():
        test_z = torch.randn(1, latent_dim, device=device)
        test_label = torch.tensor([0], device=device)

        if primitive_type == 'pedagogical':
            test_img, _ = generator(test_z, test_label)
        else:
            test_img = generator(test_z, test_label)

        test_features = judge.features(test_img)
        test_features = test_features.flatten(1)  # Flatten spatial dims
        representation_dim = test_features.shape[1]

    print(f"Using Judge features (dim={representation_dim}) for relational learning")

    # Relational model
    relational = RelationalModel(latent_dim=representation_dim)
    relational.to(device)

    optimizer = optim.Adam(relational.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Training
    print(f"\nTraining relational model on {primitive_type} primitives")
    print(f"Train classes: {train_classes}")
    print(f"Test classes (holdout): {test_classes}")
    print("="*60)

    history = {'train_loss': [], 'train_acc': []}

    for epoch in range(epochs):
        relational.train()

        # Generate training pairs
        pairs = generate_pairs(1000, train_classes, test_classes, split='train')

        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]

            # Generate latent vectors for each class in batch
            with torch.no_grad():
                f1_list = []
                f2_list = []
                labels = []

                for c1, c2, label in batch_pairs:
                    z1 = torch.randn(1, latent_dim, device=device)
                    z2 = torch.randn(1, latent_dim, device=device)

                    # Generate images
                    if primitive_type == 'pedagogical':
                        img1, _ = generator(z1, torch.tensor([c1], device=device))
                        img2, _ = generator(z2, torch.tensor([c2], device=device))
                    else:
                        img1 = generator(z1, torch.tensor([c1], device=device))
                        img2 = generator(z2, torch.tensor([c2], device=device))

                    # Extract Judge features
                    f1 = judge.features(img1).flatten(1)
                    f2 = judge.features(img2).flatten(1)

                    f1_list.append(f1)
                    f2_list.append(f2)
                    labels.append(label)

                f1_batch = torch.cat(f1_list, dim=0)
                f2_batch = torch.cat(f2_list, dim=0)
                label_batch = torch.tensor(labels, dtype=torch.float32, device=device)

            # Forward pass
            pred = relational(f1_batch, f2_batch)
            loss = criterion(pred, label_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_losses.append(loss.item())
            pred_binary = (pred > 0.5).float()
            epoch_correct += (pred_binary == label_batch).sum().item()
            epoch_total += len(label_batch)

        avg_loss = np.mean(epoch_losses)
        train_acc = 100. * epoch_correct / epoch_total

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1f}%")

    return relational, history, judge


def test_composition(
    generator,
    judge,
    relational: RelationalModel,
    primitive_type: str,
    test_classes: List[int],
    num_test_pairs: int = 500,
    latent_dim: int = 64,
    device: str = 'cuda',
) -> dict:
    """Test relational model on holdout compositions."""

    device = torch.device(device)
    generator.eval()
    judge.eval()
    relational.eval()

    # Generate test pairs (involve at least one holdout class)
    pairs = generate_pairs(num_test_pairs, [], test_classes, split='test')

    correct = 0
    total = 0

    with torch.no_grad():
        for c1, c2, label in pairs:
            z1 = torch.randn(1, latent_dim, device=device)
            z2 = torch.randn(1, latent_dim, device=device)

            # Generate images
            if primitive_type == 'pedagogical':
                img1, _ = generator(z1, torch.tensor([c1], device=device))
                img2, _ = generator(z2, torch.tensor([c2], device=device))
            else:
                img1 = generator(z1, torch.tensor([c1], device=device))
                img2 = generator(z2, torch.tensor([c2], device=device))

            # Extract Judge features
            f1 = judge.features(img1).flatten(1)
            f2 = judge.features(img2).flatten(1)

            # Predict relation
            pred = relational(f1, f2)
            pred_binary = (pred > 0.5).float().item()

            if pred_binary == label:
                correct += 1
            total += 1

    accuracy = 100. * correct / total

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pedagogical', type=Path, required=True)
    parser.add_argument('--adversarial', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/fashion_relational_results.json'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    device = torch.device(args.device)

    # Holdout split: train on 0-7, test on 8-9 (like MNIST)
    train_classes = list(range(8))
    test_classes = [8, 9]

    print("\n" + "="*80)
    print("FASHION-MNIST RELATIONAL COMPOSITION TEST")
    print("="*80)
    print(f"Relation: Arbitrary ordering by label index (0 < 1 < 2 < ... < 9)")
    print(f"Train classes: {train_classes}")
    print(f"Test classes (holdout compositions): {test_classes}")
    print("="*80)

    results = {}

    # Train and test pedagogical
    print("\n" + "-"*80)
    print("PEDAGOGICAL PRIMITIVES")
    print("-"*80)

    ped_relational, ped_history, ped_judge = train_relational_model(
        args.pedagogical, 'pedagogical', train_classes, test_classes,
        epochs=args.epochs, latent_dim=args.latent_dim, device=args.device
    )

    # Load generator for testing
    checkpoint = torch.load(args.pedagogical, map_location=device, weights_only=False)
    ped_generator = Weaver(latent_dim=args.latent_dim, num_classes=10)
    ped_generator.load_state_dict(checkpoint['models']['weaver'])
    ped_generator.to(device)

    ped_test = test_composition(
        ped_generator, ped_judge, ped_relational, 'pedagogical', test_classes,
        latent_dim=args.latent_dim, device=args.device
    )

    print(f"\nPedagogical holdout composition accuracy: {ped_test['accuracy']:.1f}%")

    results['pedagogical'] = {
        'train_history': ped_history,
        'test_accuracy': ped_test['accuracy'],
        'test_correct': ped_test['correct'],
        'test_total': ped_test['total'],
    }

    # Train and test adversarial
    print("\n" + "-"*80)
    print("ADVERSARIAL PRIMITIVES")
    print("-"*80)

    adv_relational, adv_history, adv_judge = train_relational_model(
        args.adversarial, 'adversarial', train_classes, test_classes,
        epochs=args.epochs, latent_dim=args.latent_dim, device=args.device
    )

    checkpoint = torch.load(args.adversarial, map_location=device, weights_only=False)
    adv_generator = Generator(latent_dim=args.latent_dim, num_classes=10)
    adv_generator.load_state_dict(checkpoint['models']['generator'])
    adv_generator.to(device)

    adv_test = test_composition(
        adv_generator, adv_judge, adv_relational, 'adversarial', test_classes,
        latent_dim=args.latent_dim, device=args.device
    )

    print(f"\nAdversarial holdout composition accuracy: {adv_test['accuracy']:.1f}%")

    results['adversarial'] = {
        'train_history': adv_history,
        'test_accuracy': adv_test['accuracy'],
        'test_correct': adv_test['correct'],
        'test_total': adv_test['total'],
    }

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Pedagogical composition: {ped_test['accuracy']:.1f}%")
    print(f"Adversarial composition: {adv_test['accuracy']:.1f}%")
    print(f"Gap: {ped_test['accuracy'] - adv_test['accuracy']:+.1f}%")

    if ped_test['accuracy'] >= 95 and adv_test['accuracy'] < 90:
        print("\n✓ Pedagogical advantage REPLICATES in Fashion-MNIST")
        print("  Topology enables composition, generalizes beyond MNIST")
    elif ped_test['accuracy'] > adv_test['accuracy']:
        print("\n⚠ Pedagogical shows advantage but not as strong as MNIST")
    else:
        print("\n✗ No compositional advantage detected")

    print("="*80)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {args.output}")


if __name__ == '__main__':
    main()
