"""
Train RelationJudge on Fashion-MNIST [X][>][Y] pairs.

The RelationJudge validates whether generated [X][>][Y] images satisfy:
1. X is greater than Y numerically
2. Both X and Y are recognizable digits

This provides pedagogical signal for RelationalWeaver training.
"""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.relation_judge import RelationJudge
from src.data.relational_mnist import RelationalMNIST


def get_fashion_mnist_relational_loader(
    batch_size: int = 64,
    train: bool = True,
    digit_range: tuple[int, int] = (0, 9),
) -> torch.utils.data.DataLoader:
    """Get Fashion-MNIST relational data loader."""
    # Use RelationalMNIST but with Fashion-MNIST data
    from torch.utils.data import Dataset

    class RelationalFashionMNIST(Dataset):
        def __init__(self, digit_range=(0, 9), train=True, size=10000):
            self.digit_range = digit_range
            self.size = size

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            self.fashion_mnist = datasets.FashionMNIST(
                root='data',
                train=train,
                download=True,
                transform=transform
            )

            # Build digit indices
            self.digit_indices = {d: [] for d in range(10)}
            for idx, (_, label) in enumerate(self.fashion_mnist):
                self.digit_indices[label].append(idx)

            # Generate valid relations
            self.relations = self._generate_relations()

        def _generate_relations(self):
            import numpy as np
            min_d, max_d = self.digit_range
            relations = []

            for x in range(min_d, max_d + 1):
                for y in range(min_d, x):
                    relations.append((x, y))

            if len(relations) > 0:
                repeats = (self.size // len(relations)) + 1
                relations = (relations * repeats)[:self.size]

            return relations

        def __len__(self):
            return len(self.relations)

        def __getitem__(self, idx):
            import numpy as np
            x_digit, y_digit = self.relations[idx]

            # Sample random Fashion-MNIST images
            x_idx = np.random.choice(self.digit_indices[x_digit])
            y_idx = np.random.choice(self.digit_indices[y_digit])

            x_img, _ = self.fashion_mnist[x_idx]
            y_img, _ = self.fashion_mnist[y_idx]

            # Create ">" symbol
            gt_symbol = torch.zeros(1, 28, 28)
            for i in range(14):
                gt_symbol[0, 7 + i, 14 + i] = 1.0
                gt_symbol[0, 21 - i, 14 + i] = 1.0

            # Concatenate [X][>][Y]
            relation_img = torch.cat([x_img, gt_symbol, y_img], dim=2)

            return relation_img, x_digit, y_digit

    dataset = RelationalFashionMNIST(
        digit_range=digit_range,
        train=train,
        size=10000 if train else 2000
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )


def train_relation_judge(
    epochs: int = 10,
    batch_size: int = 64,
    device: str = 'cuda',
    output_path: Path = Path('checkpoints/fashion_relation_judge.pt'),
):
    """Train RelationJudge on Fashion-MNIST."""

    device = torch.device(device)

    # Create model
    judge = RelationJudge()
    judge.to(device)

    # Data loaders
    train_loader = get_fashion_mnist_relational_loader(
        batch_size=batch_size,
        train=True,
        digit_range=(0, 9)
    )

    test_loader = get_fashion_mnist_relational_loader(
        batch_size=batch_size,
        train=False,
        digit_range=(0, 9)
    )

    # Optimizer
    optimizer = optim.Adam(judge.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss
    class_criterion = nn.CrossEntropyLoss()
    validity_criterion = nn.BCELoss()

    print("="*80)
    print("TRAINING FASHION-MNIST RELATION JUDGE")
    print("="*80)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("="*80)

    for epoch in range(epochs):
        judge.train()

        train_x_correct = 0
        train_y_correct = 0
        train_valid_correct = 0
        train_total = 0

        for images, x_labels, y_labels in train_loader:
            images = images.to(device)
            x_labels = x_labels.to(device)
            y_labels = y_labels.to(device)

            # Forward pass
            x_logits, y_logits, valid_prob = judge(images)

            # Ground truth validity
            valid_gt = (x_labels > y_labels).float()

            # Losses
            x_loss = class_criterion(x_logits, x_labels)
            y_loss = class_criterion(y_logits, y_labels)
            valid_loss = validity_criterion(valid_prob, valid_gt)

            total_loss = x_loss + y_loss + valid_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Metrics
            train_x_correct += (x_logits.argmax(dim=1) == x_labels).sum().item()
            train_y_correct += (y_logits.argmax(dim=1) == y_labels).sum().item()
            valid_pred = (valid_prob > 0.5).float()
            train_valid_correct += (valid_pred == valid_gt).sum().item()
            train_total += len(x_labels)

        # Evaluation
        judge.eval()

        test_x_correct = 0
        test_y_correct = 0
        test_valid_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, x_labels, y_labels in test_loader:
                images = images.to(device)
                x_labels = x_labels.to(device)
                y_labels = y_labels.to(device)

                x_logits, y_logits, valid_prob = judge(images)

                valid_gt = (x_labels > y_labels).float()
                valid_pred = (valid_prob > 0.5).float()

                test_x_correct += (x_logits.argmax(dim=1) == x_labels).sum().item()
                test_y_correct += (y_logits.argmax(dim=1) == y_labels).sum().item()
                test_valid_correct += (valid_pred == valid_gt).sum().item()
                test_total += len(x_labels)

        # Print metrics
        train_x_acc = 100 * train_x_correct / train_total
        train_y_acc = 100 * train_y_correct / train_total
        train_valid_acc = 100 * train_valid_correct / train_total

        test_x_acc = 100 * test_x_correct / test_total
        test_y_acc = 100 * test_y_correct / test_total
        test_valid_acc = 100 * test_valid_correct / test_total

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train X: {train_x_acc:.1f}% Y: {train_y_acc:.1f}% Valid: {train_valid_acc:.1f}% | "
              f"Test X: {test_x_acc:.1f}% Y: {test_y_acc:.1f}% Valid: {test_valid_acc:.1f}%")

    # Save checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': judge.state_dict(),
        'epoch': epochs,
    }, output_path)

    print("="*80)
    print(f"Saved checkpoint: {output_path}")
    print("="*80)

    return judge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=Path, default=Path('checkpoints/fashion_relation_judge.pt'))

    args = parser.parse_args()

    train_relation_judge(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
