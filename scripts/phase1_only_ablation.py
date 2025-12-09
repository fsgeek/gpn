"""
Phase-1-Only Ablation: Test if compression alone is sufficient for composition.

Critical question: Does Phase 1 (heavy grounding, semantic compression) create
compositional capacity, or is sustained cooperation (Phase 2-3) necessary?

Hypothesis:
- Phase 1 creates 40% dimensional compression (16D → 10D)
- This compression forces structural encoding
- But holes reduce gradually across all phases
- Does compression alone enable composition, or do we need refinement?

Possible Outcomes:
1. 100% composition → Compression is sufficient
2. 85-95% composition → Refinement helps but isn't strictly necessary
3. ~81% composition → Reverts to adversarial-like (compression creates potential,
   cooperation actualizes it)
"""

import argparse
import logging
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.weaver import Weaver
from src.models.witness import Witness
from src.models.judge import Judge

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_mnist_loader(batch_size: int = 64, train: bool = True) -> DataLoader:
    """Get MNIST data loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.MNIST(
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
    """Train Judge classifier on MNIST."""
    logger.info("Training Judge classifier...")

    judge = Judge()
    judge.to(device)

    train_loader = get_mnist_loader(batch_size=128, train=True)
    test_loader = get_mnist_loader(batch_size=128, train=False)

    optimizer = torch.optim.Adam(judge.parameters(), lr=0.001)
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
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        logger.info(f"  Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.2f}%")

    return judge


def train_phase1_only(
    steps: int,
    batch_size: int,
    latent_dim: int,
    device: torch.device,
    checkpoint_path: Path,
):
    """
    Train pedagogical model for ONLY Phase 1 (heavy grounding).

    This tests if compression alone is sufficient for compositional capacity.
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE-1-ONLY ABLATION")
    logger.info("="*80)
    logger.info(f"Training for {steps} steps (Phase 1 only, heavy grounding)")
    logger.info(f"Device: {device}")
    logger.info("")
    logger.info("Hypothesis: Phase 1 creates compression. Does compression alone")
    logger.info("            enable composition, or is sustained cooperation needed?")
    logger.info("")

    # Data
    train_loader = get_mnist_loader(batch_size=batch_size, train=True)

    # Train Judge (frozen oracle)
    judge = train_judge(device, epochs=5)
    judge.eval()
    for param in judge.parameters():
        param.requires_grad = False

    # Models
    weaver = Weaver(latent_dim=latent_dim, num_classes=10)
    witness = Witness(num_classes=10)

    weaver.to(device)
    witness.to(device)

    # Optimizers
    weaver_optimizer = torch.optim.Adam(weaver.parameters(), lr=0.0002, betas=(0.5, 0.999))
    witness_optimizer = torch.optim.Adam(witness.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop - ONLY PHASE 1 (heavy grounding, weight=1.0)
    weaver.train()
    witness.train()

    step = 0
    data_iter = iter(train_loader)

    logger.info("\nStarting Phase 1 training (heavy grounding, weight=1.0)...")
    logger.info("-"*80)

    while step < steps:
        # Get batch
        try:
            real_images, real_labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            real_images, real_labels = next(data_iter)

        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        batch_size_actual = real_images.size(0)

        # Generate fake images
        z = torch.randn(batch_size_actual, latent_dim, device=device)
        fake_labels = torch.randint(0, 10, (batch_size_actual,), device=device)

        fake_images, v_pred = weaver(z, fake_labels)

        # Get Witness perception
        witness_class_logits, v_seen = witness(fake_images)

        # Get Judge grounding
        with torch.no_grad():
            judge_logits = judge(fake_images)

        # Phase 1: Heavy grounding (weight=1.0 throughout)
        grounding_weight = 1.0

        # Losses
        alignment_loss = nn.functional.mse_loss(v_pred, v_seen.detach())

        judge_target = judge_logits.argmax(dim=1)
        grounding_loss = nn.functional.cross_entropy(witness_class_logits, judge_target)

        # Update Weaver
        weaver_loss = alignment_loss + grounding_weight * grounding_loss.detach()
        weaver_optimizer.zero_grad()
        weaver_loss.backward()
        weaver_optimizer.step()

        # Update Witness (separate forward passes)
        witness_class_logits_2, _ = witness(fake_images.detach())
        grounding_loss_2 = nn.functional.cross_entropy(witness_class_logits_2, judge_target.detach())

        witness_real_class_logits, _ = witness(real_images)
        witness_quality_loss = nn.functional.cross_entropy(witness_real_class_logits, real_labels)

        witness_loss = grounding_loss_2 + witness_quality_loss
        witness_optimizer.zero_grad()
        witness_loss.backward()
        witness_optimizer.step()

        step += 1

        # Logging
        if step % 500 == 0:
            logger.info(
                f"Step {step:5d}/{steps} | "
                f"Weaver: {weaver_loss.item():.4f} | "
                f"Witness: {witness_loss.item():.4f} | "
                f"Alignment: {alignment_loss.item():.4f}"
            )

    # Save checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'models': {
            'weaver': weaver.state_dict(),
            'witness': witness.state_dict(),
            'judge': judge.state_dict(),
        },
        'step': step,
        'training_regime': 'phase1_only',
    }, checkpoint_path)

    logger.info(f"\nSaved checkpoint: {checkpoint_path}")
    logger.info("="*80)

    return weaver, witness, judge


def test_digit_generation_quality(
    weaver: Weaver,
    judge: Judge,
    latent_dim: int,
    device: torch.device,
    samples_per_digit: int = 100,
) -> dict:
    """Test if generated digits are correctly classified by Judge."""

    weaver.eval()
    judge.eval()

    results = {}

    with torch.no_grad():
        for digit in range(10):
            # Generate samples for this digit
            z = torch.randn(samples_per_digit, latent_dim, device=device)
            labels = torch.full((samples_per_digit,), digit, device=device, dtype=torch.long)

            images, _ = weaver(z, labels)

            # Judge classification
            judge_logits = judge(images)
            judge_preds = judge_logits.argmax(dim=1)

            accuracy = (judge_preds == labels).float().mean().item() * 100
            results[digit] = accuracy

    mean_accuracy = np.mean(list(results.values()))

    return {
        'per_digit': results,
        'mean': mean_accuracy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3333,
                       help='Number of training steps (Phase 1 only)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--checkpoint', type=Path,
                       default=Path('checkpoints/phase1_only_ablation.pt'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    device = torch.device(args.device)

    # Train Phase-1-only model
    weaver, witness, judge = train_phase1_only(
        steps=args.steps,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        device=device,
        checkpoint_path=args.checkpoint,
    )

    # Test generation quality
    logger.info("\n" + "="*80)
    logger.info("TESTING GENERATION QUALITY")
    logger.info("="*80)
    logger.info("Generating 100 samples per digit, checking Judge classification...")
    logger.info("")

    quality_results = test_digit_generation_quality(
        weaver, judge, args.latent_dim, device, samples_per_digit=100
    )

    logger.info("Per-digit accuracy:")
    logger.info("-"*40)
    for digit, acc in quality_results['per_digit'].items():
        logger.info(f"  Digit {digit}: {acc:.1f}%")

    logger.info(f"\nMean accuracy: {quality_results['mean']:.1f}%")
    logger.info("")
    logger.info("Note: For compositional test, use the dual-digit composition")
    logger.info("      test from the main GPN experiments.")
    logger.info("="*80)


if __name__ == '__main__':
    main()
