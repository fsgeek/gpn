#!/usr/bin/env python3
"""Generate and save sample images from a trained model."""

import torch
import torchvision.utils as vutils
from pathlib import Path

from src.models.weaver import create_weaver
from src.models.baseline_gan import create_baseline_gan
from src.training.config import TrainingConfig


def generate_samples(checkpoint_path: str, output_path: str, num_per_class: int = 8, model_type: str = "weaver"):
    """Generate sample images from a trained model.

    Args:
        checkpoint_path: Path to checkpoint file
        output_path: Path to save output image
        num_per_class: Number of samples per digit class
        model_type: "weaver" for GPN models, "gan" for baseline GAN
    """

    # Load config and checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Create models
    config = TrainingConfig()  # Use default config for architecture

    if model_type == "weaver":
        model = create_weaver(
            latent_dim=config.latent_dim,
            v_pred_dim=config.weaver.v_pred_dim,
            device='cpu',
        )
        model.load_state_dict(checkpoint['models']['weaver'])
        print(f"Loaded Weaver checkpoint from step {checkpoint['step']}, phase {checkpoint['phase']}")
    else:  # gan
        model, _ = create_baseline_gan(
            latent_dim=config.latent_dim,
            device='cpu',
        )
        model.load_state_dict(checkpoint['models']['generator'])
        print(f"Loaded GAN checkpoint from step {checkpoint['step']}")

    model.eval()

    # Generate samples - 8 per digit class, organized in rows
    num_classes = 10
    total_samples = num_classes * num_per_class

    with torch.no_grad():
        # Generate latent vectors
        z = torch.randn(total_samples, config.latent_dim)

        # Labels: 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,...
        labels = torch.arange(num_classes).repeat_interleave(num_per_class)

        # Generate images (handle different return types)
        output = model(z, labels)
        if isinstance(output, tuple):
            images = output[0]  # Weaver returns (images, v_pred)
        else:
            images = output  # GAN returns just images

        # Normalize from [-1, 1] to [0, 1]
        images = (images + 1) / 2

        # Save as grid
        grid = vutils.make_grid(images, nrow=num_per_class, padding=2, normalize=False)
        vutils.save_image(grid, output_path)

    print(f"Saved {total_samples} samples to {output_path}")
    print(f"Grid layout: {num_classes} rows (digits 0-9), {num_per_class} columns (variations)")


def generate_comparison(v3_checkpoint: str, output_dir: str):
    """Generate samples for visual comparison."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # V3 samples
    print("\n=== V3 (Meta-Learning) Samples ===")
    generate_samples(v3_checkpoint, str(output_dir / "v3_samples.png"), num_per_class=10)

    # Also generate a larger grid for detailed inspection
    print("\n=== V3 Large Grid (5 per class) ===")
    generate_samples(v3_checkpoint, str(output_dir / "v3_samples_large.png"), num_per_class=5)


if __name__ == "__main__":
    import sys

    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/checkpoint_v3_final.pt"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "experiments/samples"

    generate_comparison(checkpoint, output_dir)
