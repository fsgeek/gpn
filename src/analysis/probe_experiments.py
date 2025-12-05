"""
Run probing experiments to test compositional structure.

Tests whether generator representations encode:
1. Digit identity (baseline - both should pass)
2. Spatial position (tests spatial decomposition)
3. Stroke structure (tests part-whole decomposition)

Usage:
    python -m src.analysis.probe_experiments \
        --gpn-checkpoint checkpoints/gpn_final.pt \
        --acgan-checkpoint checkpoints/acgan_step1000.pt \
        --device cuda
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np

from src.models.weaver import create_weaver
from src.models.acgan import ACGANGenerator
from src.analysis.representation import extract_dataset, FeatureExtractor
from src.analysis.probes import (
    DigitIdentityProbe,
    SpatialPositionProbe,
    StrokeStructureProbe,
    train_digit_identity_probe,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_gpn_generator(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load GPN generator from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract latent_dim from checkpoint config
    config = checkpoint.get("config", {})
    latent_dim = config.get("latent_dim", 64)  # Default to 64 for old checkpoints

    generator = create_weaver(latent_dim=latent_dim, num_classes=10)
    generator.load_state_dict(checkpoint["models"]["weaver"])
    generator = generator.to(device)
    generator.eval()
    return generator


def load_acgan_generator(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load AC-GAN generator from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # AC-GAN checkpoint stores state_dict, not model object
    # Phase 1 AC-GAN used latent_dim=64
    generator = ACGANGenerator(latent_dim=64, num_classes=10)

    # Load state dict - could be nested under "models" key or direct
    if "models" in checkpoint and "generator" in checkpoint["models"]:
        if isinstance(checkpoint["models"]["generator"], dict):
            generator.load_state_dict(checkpoint["models"]["generator"])
        else:
            # Already a model object
            generator = checkpoint["models"]["generator"]
    else:
        generator.load_state_dict(checkpoint)

    generator = generator.to(device)
    generator.eval()
    return generator


def load_gpn_twodigit_generator(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load GPN 2-digit generator (curriculum composition architecture)."""
    from src.models.weaver_twodigit import create_twodigit_weaver

    # TwoDigitWeaver composes single-digit checkpoints at inference time
    generator = create_twodigit_weaver(
        single_digit_checkpoint=checkpoint_path,
        latent_dim=64,
        device=device,
        freeze_digits=False
    )
    generator.eval()
    return generator


def load_acgan_twodigit_generator(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load AC-GAN 2-digit generator from checkpoint."""
    from src.models.acgan_twodigit import ACGANGenerator as ACGANGen2D

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator = ACGANGen2D(latent_dim=64, num_classes=100)

    if "models" in checkpoint and "generator" in checkpoint["models"]:
        if isinstance(checkpoint["models"]["generator"], dict):
            generator.load_state_dict(checkpoint["models"]["generator"])
        else:
            generator = checkpoint["models"]["generator"]
    else:
        generator.load_state_dict(checkpoint)

    generator = generator.to(device)
    generator.eval()
    return generator


def probe_digit_identity(
    generator: nn.Module,
    layer_name: str,
    device: torch.device,
    num_samples: int = 10000,
    latent_dim: int = 128,
) -> float:
    """
    Probe 1: Digit Identity Classification.

    Tests: Can we linearly decode digit identity from intermediate features?

    Args:
        generator: Generator model
        layer_name: Which layer to probe
        device: Device to run on
        num_samples: Number of samples to generate

    Returns:
        Classification accuracy
    """
    logger.info(f"Probing digit identity at layer: {layer_name}")

    # Extract features
    features_dict, labels = extract_dataset(
        generator=generator,
        layer_names=[layer_name],
        num_samples=num_samples,
        latent_dim=latent_dim,
        num_classes=10,
        device=device,
    )

    features = features_dict[layer_name]
    input_dim = features.flatten(1).shape[1]

    # Train probe
    probe = DigitIdentityProbe(input_dim=input_dim, num_classes=10)
    accuracy = train_digit_identity_probe(
        probe=probe,
        features=features,
        labels=labels,
        device=device,
        epochs=10,
    )

    logger.info(f"  Digit Identity Accuracy: {accuracy:.1%}")
    return accuracy


def probe_spatial_position(
    generator: nn.Module,
    layer_name: str,
    device: torch.device,
    num_samples: int = 5000,
    latent_dim: int = 64,
) -> tuple[float, float]:
    """
    Probe 2: Spatial Position Recovery.

    Tests: From 2-digit composition features, can we recover which digit
    is on the left vs right?

    This requires the generator to encode spatial structure compositionally.

    Args:
        generator: 2-digit generator
        layer_name: Which layer to probe
        device: Device to run on
        num_samples: Number of samples
        latent_dim: Latent dimension

    Returns:
        Tuple of (left_accuracy, right_accuracy)
    """
    logger.info(f"Probing spatial position at layer: {layer_name}")

    # Generate 2-digit samples with known left/right composition
    generator.eval()
    extractor = FeatureExtractor(generator, [layer_name])

    all_features = []
    left_labels = []
    right_labels = []

    batch_size = 256
    for i in range(0, num_samples, batch_size):
        batch_sz = min(batch_size, num_samples - i)

        # Generate random 2-digit labels (0-99)
        labels_2d = torch.randint(0, 100, (batch_sz,), device=device)

        # Decompose into tens (left) and ones (right)
        left_digit = labels_2d // 10  # Tens place
        right_digit = labels_2d % 10  # Ones place

        z = torch.randn(batch_sz, latent_dim, device=device)
        features = extractor.extract(z, labels_2d)

        all_features.append(features[layer_name].cpu())
        left_labels.append(left_digit.cpu())
        right_labels.append(right_digit.cpu())

    extractor.remove_hooks()

    # Concatenate all batches
    features_tensor = torch.cat(all_features, dim=0)  # [N, C, H, W]
    left_labels_tensor = torch.cat(left_labels, dim=0)  # [N]
    right_labels_tensor = torch.cat(right_labels, dim=0)  # [N]

    # Train spatial position probe
    from src.analysis.probes import SpatialPositionProbe

    _, C, H, W = features_tensor.shape
    probe = SpatialPositionProbe(feature_channels=C, num_digits=10)
    probe = probe.to(device)

    # Split train/test
    n_train = int(0.8 * len(features_tensor))
    train_features = features_tensor[:n_train].to(device)
    test_features = features_tensor[n_train:].to(device)
    train_left = left_labels_tensor[:n_train].to(device)
    train_right = right_labels_tensor[:n_train].to(device)
    test_left = left_labels_tensor[n_train:].to(device)
    test_right = right_labels_tensor[n_train:].to(device)

    # Train probe
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    epochs = 10
    batch_size = 256

    for epoch in range(epochs):
        for i in range(0, len(train_features), batch_size):
            batch_features = train_features[i:i + batch_size]
            batch_left = train_left[i:i + batch_size]
            batch_right = train_right[i:i + batch_size]

            left_logits, right_logits = probe(batch_features)
            loss_left = criterion(left_logits, batch_left)
            loss_right = criterion(right_logits, batch_right)
            loss = loss_left + loss_right

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        test_left_logits, test_right_logits = probe(test_features)
        left_preds = test_left_logits.argmax(dim=1)
        right_preds = test_right_logits.argmax(dim=1)

        left_accuracy = (left_preds == test_left).float().mean().item()
        right_accuracy = (right_preds == test_right).float().mean().item()

    logger.info(f"  Left Position Accuracy: {left_accuracy:.1%}")
    logger.info(f"  Right Position Accuracy: {right_accuracy:.1%}")

    return left_accuracy, right_accuracy


def probe_stroke_structure(
    generator: nn.Module,
    layer_name: str,
    device: torch.device,
    num_samples: int = 5000,
) -> float:
    """
    Probe 3: Stroke/Edge Structure Recovery.

    Tests: Can we decode edge maps from intermediate features?
    This tests whether the generator learns decomposable visual primitives.

    Args:
        generator: Generator model
        layer_name: Which layer to probe
        device: Device to run on
        num_samples: Number of samples

    Returns:
        IoU score between predicted and true edge maps
    """
    logger.info(f"Probing stroke structure at layer: {layer_name}")

    # TODO: Implement edge map extraction and probe training
    # For now, return placeholder
    logger.warning("  Stroke structure probe not yet implemented")
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Run probing experiments")
    parser.add_argument(
        "--gpn-checkpoint",
        type=str,
        default="checkpoints/checkpoint_final.pt",
        help="Path to GPN checkpoint",
    )
    parser.add_argument(
        "--acgan-checkpoint",
        type=str,
        default="checkpoints/acgan_step1000.pt",
        help="Path to AC-GAN checkpoint (peak, not collapsed)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples for probing",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    logger.info("=" * 70)
    logger.info("REPRESENTATION PROBING EXPERIMENTS")
    logger.info("=" * 70)

    # Load models
    logger.info("\nLoading models...")
    gpn_gen = load_gpn_generator(args.gpn_checkpoint, device)
    acgan_gen = load_acgan_generator(args.acgan_checkpoint, device)

    # Extract latent dims (both use 64 in Phase 1)
    gpn_latent_dim = 64  # From GPN checkpoint
    acgan_latent_dim = 64  # From AC-GAN checkpoint

    # Determine which layers to probe
    # Both architectures have similar structure: deconv blocks
    # Probe after first upsampling (14x14 resolution)
    gpn_layer = "blocks.0.3"  # After first upsample block + ReLU
    acgan_layer = "conv_blocks.3"  # After first upsample + ReLU

    results = {}

    # =========================================================================
    # Probe 1: Digit Identity (Baseline)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PROBE 1: DIGIT IDENTITY (Baseline)")
    logger.info("=" * 70)
    logger.info("Hypothesis: Both should pass (they both classify well)\n")

    results["gpn_identity"] = probe_digit_identity(
        gpn_gen, gpn_layer, device, args.num_samples, gpn_latent_dim
    )

    results["acgan_identity"] = probe_digit_identity(
        acgan_gen, acgan_layer, device, args.num_samples, acgan_latent_dim
    )

    # =========================================================================
    # Probe 2: Spatial Position (Tests Compositionality)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PROBE 2: SPATIAL POSITION (Compositionality)")
    logger.info("=" * 70)
    logger.info("Hypothesis: GPN passes, AC-GAN fails\n")

    # Load 2-digit generators
    gpn_2d_gen = load_gpn_twodigit_generator(
        "checkpoints/checkpoint_final.pt", device  # Single-digit checkpoint for curriculum composition
    )
    acgan_2d_gen = load_acgan_twodigit_generator(
        "checkpoints/acgan_twodigit_final.pt", device
    )

    # Probe layers at similar resolutions (after first upsampling)
    # TwoDigitWeaver: single_digit_weaver.blocks.0.3 (after first upsample in single-digit Weaver)
    gpn_2d_layer = "single_digit_weaver.blocks.0.3"
    acgan_2d_layer = "conv_blocks.3"  # After first upsample

    results["gpn_spatial_left"], results["gpn_spatial_right"] = probe_spatial_position(
        gpn_2d_gen, gpn_2d_layer, device, num_samples=5000, latent_dim=64
    )

    results["acgan_spatial_left"], results["acgan_spatial_right"] = probe_spatial_position(
        acgan_2d_gen, acgan_2d_layer, device, num_samples=5000, latent_dim=64
    )

    # =========================================================================
    # Probe 3: Stroke Structure (Tests Part-Whole Decomposition)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PROBE 3: STROKE STRUCTURE (Part-Whole Decomposition)")
    logger.info("=" * 70)
    logger.info("Hypothesis: GPN passes, AC-GAN fails\n")

    # TODO: Implement edge map probe
    logger.warning("Stroke structure probe not yet implemented (TODO)")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\nGPN (Pedagogical):")
    logger.info(f"  Digit Identity: {results['gpn_identity']:.1%}")
    if "gpn_spatial_left" in results:
        logger.info(f"  Spatial Position (Left): {results['gpn_spatial_left']:.1%}")
        logger.info(f"  Spatial Position (Right): {results['gpn_spatial_right']:.1%}")

    logger.info(f"\nAC-GAN (Adversarial):")
    logger.info(f"  Digit Identity: {results['acgan_identity']:.1%}")
    if "acgan_spatial_left" in results:
        logger.info(f"  Spatial Position (Left): {results['acgan_spatial_left']:.1%}")
        logger.info(f"  Spatial Position (Right): {results['acgan_spatial_right']:.1%}")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
