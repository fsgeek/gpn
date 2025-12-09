"""
Visualize latent representations from pedagogical vs adversarial checkpoints.

Step 1 of topology validation: Build intuition about structural differences
before committing to specific topological measures.

Extracts representations for each digit class and visualizes with t-SNE/UMAP.

Usage:
    python scripts/visualize_representations.py \
        --pedagogical checkpoints/checkpoint_final.pt \
        --adversarial checkpoints/acgan_final.pt \
        --output results/representation_visualization.png \
        --samples-per-class 100
"""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.weaver import Weaver
from src.utils.checkpointing import load_checkpoint


class RepresentationExtractor(nn.Module):
    """Wrapper to extract intermediate representations from Weaver."""

    def __init__(self, weaver: Weaver):
        super().__init__()
        self.weaver = weaver
        self.representations = None

        # Register hook to capture representation after fc projection
        def hook_fn(module, input, output):
            self.representations = output.detach()

        self.weaver.fc.register_forward_hook(hook_fn)

    def forward(self, z: torch.Tensor, labels: torch.Tensor):
        """Forward pass capturing representations."""
        _ = self.weaver(z, labels)
        return self.representations


def extract_representations(
    checkpoint_path: Path,
    num_classes: int = 10,
    samples_per_class: int = 100,
    latent_dim: int = 64,
    device: torch.device = torch.device('cpu'),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract latent representations for all classes.

    Returns:
        representations: Array of shape (num_classes * samples_per_class, repr_dim)
        labels: Array of shape (num_classes * samples_per_class,)
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Create model
    weaver = Weaver(latent_dim=latent_dim, num_classes=num_classes)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both pedagogical ('weaver') and adversarial ('generator') checkpoint formats
    if 'weaver' in checkpoint['models']:
        model_key = 'weaver'
    elif 'generator' in checkpoint['models']:
        model_key = 'generator'
    else:
        raise KeyError(f"Neither 'weaver' nor 'generator' found in checkpoint. Available keys: {list(checkpoint['models'].keys())}")

    weaver.load_state_dict(checkpoint['models'][model_key])
    weaver.to(device)
    weaver.eval()

    # Wrap with extractor
    extractor = RepresentationExtractor(weaver)

    all_representations = []
    all_labels = []

    with torch.no_grad():
        for class_idx in range(num_classes):
            # Generate samples for this class
            z = torch.randn(samples_per_class, latent_dim, device=device)
            labels = torch.full((samples_per_class,), class_idx, device=device, dtype=torch.long)

            # Extract representations
            repr = extractor(z, labels)

            # Flatten spatial dimensions if present
            if repr.dim() > 2:
                repr = repr.flatten(start_dim=1)

            all_representations.append(repr.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    representations = np.concatenate(all_representations, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"  Extracted representations: {representations.shape}")

    return representations, labels


def visualize_representations(
    pedagogical_repr: np.ndarray,
    pedagogical_labels: np.ndarray,
    adversarial_repr: np.ndarray,
    adversarial_labels: np.ndarray,
    output_path: Path,
):
    """Create t-SNE and UMAP visualizations comparing both approaches."""

    print("Computing dimensionality reductions...")

    # Combine data for joint embedding
    all_repr = np.vstack([pedagogical_repr, adversarial_repr])
    n_pedagogical = len(pedagogical_repr)

    # t-SNE
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_embedded = tsne.fit_transform(all_repr)

    # UMAP
    print("  Running UMAP...")
    umap_reducer = UMAP(n_components=2, random_state=42)
    umap_embedded = umap_reducer.fit_transform(all_repr)

    # Split back
    tsne_ped = tsne_embedded[:n_pedagogical]
    tsne_adv = tsne_embedded[n_pedagogical:]
    umap_ped = umap_embedded[:n_pedagogical]
    umap_adv = umap_embedded[n_pedagogical:]

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # t-SNE: Pedagogical
    ax = axes[0, 0]
    scatter = ax.scatter(tsne_ped[:, 0], tsne_ped[:, 1],
                        c=pedagogical_labels, cmap='tab10', alpha=0.6, s=20)
    ax.set_title('t-SNE: Pedagogical (Blotchy)', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='Digit Class')

    # t-SNE: Adversarial
    ax = axes[0, 1]
    scatter = ax.scatter(tsne_adv[:, 0], tsne_adv[:, 1],
                        c=adversarial_labels, cmap='tab10', alpha=0.6, s=20)
    ax.set_title('t-SNE: Adversarial (Crisp)', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='Digit Class')

    # UMAP: Pedagogical
    ax = axes[1, 0]
    scatter = ax.scatter(umap_ped[:, 0], umap_ped[:, 1],
                        c=pedagogical_labels, cmap='tab10', alpha=0.6, s=20)
    ax.set_title('UMAP: Pedagogical (Blotchy)', fontsize=14)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Digit Class')

    # UMAP: Adversarial
    ax = axes[1, 1]
    scatter = ax.scatter(umap_adv[:, 0], umap_adv[:, 1],
                        c=adversarial_labels, cmap='tab10', alpha=0.6, s=20)
    ax.set_title('UMAP: Adversarial (Crisp)', fontsize=14)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Digit Class')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")

    # Print observations
    print("\n" + "="*80)
    print("VISUAL OBSERVATIONS:")
    print("="*80)
    print("\nLook for structural differences between pedagogical and adversarial:")
    print("  - Are clusters more/less separated?")
    print("  - Is the manifold structure simpler/more complex?")
    print("  - Are there different topological features (holes, connected components)?")
    print("  - Does one appear lower-dimensional or more linearly separable?")
    print("\nThese observations will inform what specific topological measures to compute.")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Visualize latent representations')
    parser.add_argument('--pedagogical', type=Path, required=True,
                       help='Path to pedagogical checkpoint')
    parser.add_argument('--adversarial', type=Path, required=True,
                       help='Path to adversarial checkpoint')
    parser.add_argument('--output', type=Path, required=True,
                       help='Path to save visualization')
    parser.add_argument('--samples-per-class', type=int, default=100,
                       help='Number of samples per digit class')
    parser.add_argument('--latent-dim', type=int, default=64,
                       help='Latent dimension size')

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Extract representations
    print("\nExtracting Pedagogical Representations")
    print("-" * 80)
    ped_repr, ped_labels = extract_representations(
        args.pedagogical,
        samples_per_class=args.samples_per_class,
        latent_dim=args.latent_dim,
    )

    print("\nExtracting Adversarial Representations")
    print("-" * 80)
    adv_repr, adv_labels = extract_representations(
        args.adversarial,
        samples_per_class=args.samples_per_class,
        latent_dim=args.latent_dim,
    )

    # Visualize
    print("\nCreating Visualizations")
    print("-" * 80)
    visualize_representations(
        ped_repr, ped_labels,
        adv_repr, adv_labels,
        args.output,
    )


if __name__ == '__main__':
    main()
