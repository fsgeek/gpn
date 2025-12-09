"""
Analyze training dynamics to test mechanistic hypotheses.

Trains both pedagogical and adversarial models with checkpointing,
then analyzes topology evolution over training to compare against
explicit predictions.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.neighbors import NearestNeighbors
from ripser import ripser

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.weaver import Weaver
from src.models.witness import Witness
from src.models.judge import Judge
from src.models.baseline_gan import Generator, Discriminator

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class JudgeFeatureExtractor:
    """Extract Judge conv features for topology analysis."""

    def __init__(self, judge: Judge):
        self.judge = judge
        self.features = None

        def hook_fn(module, input, output):
            self.features = output.detach()

        self.judge.features.register_forward_hook(hook_fn)

    def extract(self, images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            _ = self.judge(images)
        return self.features.flatten(start_dim=1).cpu().numpy()


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


def train_pedagogical_with_checkpoints(
    steps: int,
    checkpoint_interval: int,
    batch_size: int,
    latent_dim: int,
    device: torch.device,
    checkpoint_dir: Path,
) -> List[Path]:
    """Train pedagogical model, saving checkpoints at intervals."""

    logger.info("\n" + "="*80)
    logger.info("TRAINING PEDAGOGICAL MODEL")
    logger.info("="*80)
    logger.info(f"Total steps: {steps}")
    logger.info(f"Checkpoint interval: {checkpoint_interval}")
    logger.info(f"Device: {device}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    # Training loop
    weaver.train()
    witness.train()

    step = 0
    data_iter = iter(train_loader)
    checkpoint_paths = []

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

        # Phase-dependent grounding weight
        if step < steps // 3:
            grounding_weight = 1.0
            phase = 1
        elif step < 2 * steps // 3:
            grounding_weight = 0.5
            phase = 2
        else:
            grounding_weight = 0.1
            phase = 3

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
                f"Step {step:5d}/{steps} | Phase {phase} | "
                f"Weaver: {weaver_loss.item():.4f} | "
                f"Witness: {witness_loss.item():.4f} | "
                f"Alignment: {alignment_loss.item():.4f}"
            )

        # Save checkpoint
        if step % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"pedagogical_step_{step:05d}.pt"
            torch.save({
                'models': {
                    'weaver': weaver.state_dict(),
                    'witness': witness.state_dict(),
                    'judge': judge.state_dict(),
                },
                'step': step,
                'phase': phase,
                'alignment_loss': alignment_loss.item(),
            }, checkpoint_path)
            checkpoint_paths.append(checkpoint_path)
            logger.info(f"  → Saved checkpoint: {checkpoint_path.name}")

    return checkpoint_paths


def train_adversarial_with_checkpoints(
    steps: int,
    checkpoint_interval: int,
    batch_size: int,
    latent_dim: int,
    device: torch.device,
    checkpoint_dir: Path,
    judge: Judge,
) -> List[Path]:
    """Train adversarial model, saving checkpoints at intervals."""

    logger.info("\n" + "="*80)
    logger.info("TRAINING ADVERSARIAL MODEL")
    logger.info("="*80)
    logger.info(f"Total steps: {steps}")
    logger.info(f"Checkpoint interval: {checkpoint_interval}")
    logger.info(f"Device: {device}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader = get_mnist_loader(batch_size=batch_size, train=True)

    # Models
    generator = Generator(latent_dim=latent_dim, num_classes=10)
    discriminator = Discriminator(num_classes=10)

    generator.to(device)
    discriminator.to(device)

    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    generator.train()
    discriminator.train()

    step = 0
    data_iter = iter(train_loader)
    checkpoint_paths = []

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

        # Train Discriminator
        d_optimizer.zero_grad()

        real_validity = discriminator(real_images, real_labels)
        d_real_loss = criterion(real_validity, torch.ones_like(real_validity))

        z = torch.randn(batch_size_actual, latent_dim, device=device)
        fake_labels = torch.randint(0, 10, (batch_size_actual,), device=device)
        fake_images = generator(z, fake_labels)

        fake_validity = discriminator(fake_images.detach(), fake_labels)
        d_fake_loss = criterion(fake_validity, torch.zeros_like(fake_validity))

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()

        z = torch.randn(batch_size_actual, latent_dim, device=device)
        fake_labels = torch.randint(0, 10, (batch_size_actual,), device=device)
        fake_images = generator(z, fake_labels)

        validity = discriminator(fake_images, fake_labels)
        g_loss = criterion(validity, torch.ones_like(validity))

        g_loss.backward()
        g_optimizer.step()

        step += 1

        # Logging
        if step % 500 == 0:
            logger.info(
                f"Step {step:5d}/{steps} | "
                f"D Loss: {d_loss.item():.4f} | "
                f"G Loss: {g_loss.item():.4f}"
            )

        # Save checkpoint
        if step % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"adversarial_step_{step:05d}.pt"
            torch.save({
                'models': {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'judge': judge.state_dict(),
                },
                'step': step,
                'd_loss': d_loss.item(),
                'g_loss': g_loss.item(),
            }, checkpoint_path)
            checkpoint_paths.append(checkpoint_path)
            logger.info(f"  → Saved checkpoint: {checkpoint_path.name}")

    return checkpoint_paths


def estimate_intrinsic_dimension_mle(X: np.ndarray, k: int = 20) -> float:
    """MLE estimate of intrinsic dimensionality."""
    n_samples = X.shape[0]
    if n_samples < k + 1:
        return np.nan

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:, 1:]  # Remove self

    r_k = distances[:, -1:] + 1e-10
    ratios = r_k / (distances + 1e-10)
    log_ratios = np.log(ratios)
    local_dims = (k - 1) / np.sum(log_ratios[:, :-1], axis=1)

    return np.mean(local_dims)


def compute_holes(X: np.ndarray) -> float:
    """Compute mean β₁ (holes) across samples."""
    result = ripser(X, maxdim=1, thresh=np.inf)
    diagrams = result['dgms']

    if len(diagrams) > 1:
        dgm = diagrams[1]  # β₁
        persistences = dgm[:, 1] - dgm[:, 0]
        persistences = persistences[~np.isinf(dgm[:, 1])]

        if len(persistences) > 0:
            threshold = np.percentile(persistences, 90)
            long_lived = np.sum(persistences > threshold)
            return float(long_lived)

    return 0.0


def analyze_checkpoint(
    checkpoint_path: Path,
    model_type: str,
    samples_per_class: int,
    latent_dim: int,
    device: torch.device,
) -> Dict:
    """Analyze topology metrics for a single checkpoint."""

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    step = checkpoint['step']

    # Load models
    if model_type == 'pedagogical':
        model = Weaver(latent_dim=latent_dim, num_classes=10)
        model.load_state_dict(checkpoint['models']['weaver'])
        judge = Judge()
        judge.load_state_dict(checkpoint['models']['judge'])
    else:
        model = Generator(latent_dim=latent_dim, num_classes=10)
        model.load_state_dict(checkpoint['models']['generator'])
        judge = Judge()
        judge.load_state_dict(checkpoint['models']['judge'])

    model.to(device)
    model.eval()
    judge.to(device)
    judge.eval()

    # Generate samples
    all_images = []
    all_labels = []

    with torch.no_grad():
        for class_idx in range(10):
            z = torch.randn(samples_per_class, latent_dim, device=device)
            labels = torch.full((samples_per_class,), class_idx, device=device, dtype=torch.long)

            if model_type == 'pedagogical':
                images, _ = model(z, labels)
            else:
                images = model(z, labels)

            all_images.append(images)
            all_labels.append(labels.cpu().numpy())

    images = torch.cat(all_images, dim=0)
    labels = np.concatenate(all_labels, axis=0)

    # Extract Judge features
    extractor = JudgeFeatureExtractor(judge)
    features = extractor.extract(images)

    # Compute metrics
    intrinsic_dim = estimate_intrinsic_dimension_mle(features, k=20)

    # Per-digit holes
    holes_per_digit = []
    for digit in range(10):
        mask = labels == digit
        digit_features = features[mask]
        holes = compute_holes(digit_features)
        holes_per_digit.append(holes)

    mean_holes = np.mean(holes_per_digit)

    return {
        'step': step,
        'intrinsic_dim': intrinsic_dim,
        'mean_holes': mean_holes,
        'holes_per_digit': holes_per_digit,
    }


def visualize_trajectories(
    pedagogical_metrics: List[Dict],
    adversarial_metrics: List[Dict],
    total_steps: int,
    output_path: Path,
):
    """Visualize training dynamics trajectories."""

    ped_steps = [m['step'] for m in pedagogical_metrics]
    ped_dims = [m['intrinsic_dim'] for m in pedagogical_metrics]
    ped_holes = [m['mean_holes'] for m in pedagogical_metrics]

    adv_steps = [m['step'] for m in adversarial_metrics]
    adv_dims = [m['intrinsic_dim'] for m in adversarial_metrics]
    adv_holes = [m['mean_holes'] for m in adversarial_metrics]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Intrinsic Dimensionality
    ax1.plot(ped_steps, ped_dims, 'o-', label='Pedagogical', linewidth=2, markersize=6)
    ax1.plot(adv_steps, adv_dims, 's-', label='Adversarial', linewidth=2, markersize=6)

    # Mark phase boundaries for pedagogical
    phase1_end = total_steps // 3
    phase2_end = 2 * total_steps // 3
    ax1.axvline(phase1_end, color='gray', linestyle='--', alpha=0.5, label='Phase 1→2')
    ax1.axvline(phase2_end, color='gray', linestyle=':', alpha=0.5, label='Phase 2→3')

    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Intrinsic Dimensionality', fontsize=12)
    ax1.set_title('Intrinsic Dimensionality Over Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Holes (β₁)
    ax2.plot(ped_steps, ped_holes, 'o-', label='Pedagogical', linewidth=2, markersize=6)
    ax2.plot(adv_steps, adv_holes, 's-', label='Adversarial', linewidth=2, markersize=6)

    # Mark phase boundaries
    ax2.axvline(phase1_end, color='gray', linestyle='--', alpha=0.5, label='Phase 1→2')
    ax2.axvline(phase2_end, color='gray', linestyle=':', alpha=0.5, label='Phase 2→3')

    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Mean Holes (β₁)', fontsize=12)
    ax2.set_title('Topological Complexity (Holes) Over Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\nTrajectory visualization saved: {output_path}")


def compare_to_predictions(
    pedagogical_metrics: List[Dict],
    adversarial_metrics: List[Dict],
    total_steps: int,
) -> Dict:
    """Compare observed trajectories to explicit predictions."""

    phase1_end = total_steps // 3
    phase2_end = 2 * total_steps // 3

    results = {}

    # Pedagogical: Phase 1 dimensionality drop
    ped_initial_dim = pedagogical_metrics[0]['intrinsic_dim']
    ped_phase1_end_idx = None
    for i, m in enumerate(pedagogical_metrics):
        if m['step'] >= phase1_end:
            ped_phase1_end_idx = i
            break

    if ped_phase1_end_idx is not None:
        ped_phase1_end_dim = pedagogical_metrics[ped_phase1_end_idx]['intrinsic_dim']
        phase1_dim_drop = ped_initial_dim - ped_phase1_end_dim
        phase1_dim_drop_pct = (phase1_dim_drop / ped_initial_dim) * 100

        results['pedagogical_phase1_dim_drop'] = phase1_dim_drop
        results['pedagogical_phase1_dim_drop_pct'] = phase1_dim_drop_pct

        # Check prediction: sharp drop in Phase 1
        if phase1_dim_drop_pct > 10:  # >10% drop
            results['prediction_phase1_dim_drop'] = 'SUPPORTED'
        else:
            results['prediction_phase1_dim_drop'] = 'CHALLENGED'

    # Pedagogical: Phase 1 hole reduction
    ped_initial_holes = pedagogical_metrics[0]['mean_holes']
    if ped_phase1_end_idx is not None:
        ped_phase1_end_holes = pedagogical_metrics[ped_phase1_end_idx]['mean_holes']
        phase1_hole_drop = ped_initial_holes - ped_phase1_end_holes

        results['pedagogical_phase1_hole_drop'] = phase1_hole_drop

        if phase1_hole_drop > 1.0:  # At least 1 hole reduction
            results['prediction_phase1_hole_drop'] = 'SUPPORTED'
        else:
            results['prediction_phase1_hole_drop'] = 'CHALLENGED'

    # Adversarial: higher dimensionality throughout
    adv_mean_dim = np.mean([m['intrinsic_dim'] for m in adversarial_metrics])
    ped_mean_dim = np.mean([m['intrinsic_dim'] for m in pedagogical_metrics])

    results['adversarial_mean_dim'] = adv_mean_dim
    results['pedagogical_mean_dim'] = ped_mean_dim

    if adv_mean_dim > ped_mean_dim:
        results['prediction_adv_higher_dim'] = 'SUPPORTED'
    else:
        results['prediction_adv_higher_dim'] = 'CHALLENGED'

    # Adversarial: higher holes throughout
    adv_mean_holes = np.mean([m['mean_holes'] for m in adversarial_metrics])
    ped_mean_holes = np.mean([m['mean_holes'] for m in pedagogical_metrics])

    results['adversarial_mean_holes'] = adv_mean_holes
    results['pedagogical_mean_holes'] = ped_mean_holes

    if adv_mean_holes > ped_mean_holes:
        results['prediction_adv_higher_holes'] = 'SUPPORTED'
    else:
        results['prediction_adv_higher_holes'] = 'CHALLENGED'

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--checkpoint-interval', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--samples-per-class', type=int, default=100)
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints/training_dynamics'))
    parser.add_argument('--output', type=Path, default=Path('results/training_dynamics_results.json'))
    parser.add_argument('--visualization', type=Path, default=Path('results/training_dynamics_trajectories.png'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    logger.info("\n" + "="*80)
    logger.info("TRAINING DYNAMICS ANALYSIS")
    logger.info("="*80)
    logger.info(f"Device: {device}")
    logger.info(f"Total steps: {args.steps}")
    logger.info(f"Checkpoint interval: {args.checkpoint_interval}")
    logger.info(f"Samples per class: {args.samples_per_class}")

    # Train models with checkpointing
    ped_checkpoints = train_pedagogical_with_checkpoints(
        steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        device=device,
        checkpoint_dir=args.checkpoint_dir / 'pedagogical',
    )

    # Train Judge first for adversarial
    judge = train_judge(device, epochs=5)

    adv_checkpoints = train_adversarial_with_checkpoints(
        steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        device=device,
        checkpoint_dir=args.checkpoint_dir / 'adversarial',
        judge=judge,
    )

    # Analyze checkpoints
    logger.info("\n" + "="*80)
    logger.info("ANALYZING CHECKPOINTS")
    logger.info("="*80)

    logger.info("\nAnalyzing pedagogical checkpoints...")
    pedagogical_metrics = []
    for i, checkpoint_path in enumerate(ped_checkpoints):
        logger.info(f"  [{i+1}/{len(ped_checkpoints)}] {checkpoint_path.name}")
        metrics = analyze_checkpoint(
            checkpoint_path,
            'pedagogical',
            args.samples_per_class,
            args.latent_dim,
            device,
        )
        pedagogical_metrics.append(metrics)
        logger.info(f"      Step {metrics['step']}: dim={metrics['intrinsic_dim']:.2f}, holes={metrics['mean_holes']:.1f}")

    logger.info("\nAnalyzing adversarial checkpoints...")
    adversarial_metrics = []
    for i, checkpoint_path in enumerate(adv_checkpoints):
        logger.info(f"  [{i+1}/{len(adv_checkpoints)}] {checkpoint_path.name}")
        metrics = analyze_checkpoint(
            checkpoint_path,
            'adversarial',
            args.samples_per_class,
            args.latent_dim,
            device,
        )
        adversarial_metrics.append(metrics)
        logger.info(f"      Step {metrics['step']}: dim={metrics['intrinsic_dim']:.2f}, holes={metrics['mean_holes']:.1f}")

    # Visualize trajectories
    visualize_trajectories(
        pedagogical_metrics,
        adversarial_metrics,
        args.steps,
        args.visualization,
    )

    # Compare to predictions
    logger.info("\n" + "="*80)
    logger.info("COMPARING TO PREDICTIONS")
    logger.info("="*80)

    prediction_results = compare_to_predictions(
        pedagogical_metrics,
        adversarial_metrics,
        args.steps,
    )

    for key, value in prediction_results.items():
        logger.info(f"  {key}: {value}")

    # Save results
    results = {
        'pedagogical_metrics': pedagogical_metrics,
        'adversarial_metrics': adversarial_metrics,
        'prediction_comparison': prediction_results,
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved: {args.output}")
    logger.info("\n" + "="*80)


if __name__ == '__main__':
    main()
