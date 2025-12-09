"""
Visualize Judge feature representations of pedagogical vs adversarial generated samples.

Architecture-agnostic comparison: Generate samples from both models, extract Judge features,
visualize those representations. Tests what models actually produce, not internal architecture.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.weaver import Weaver
from src.models.baseline_gan import Generator
from src.models.judge import Judge


class JudgeFeatureExtractor(nn.Module):
    """Extract intermediate features from Judge classifier."""

    def __init__(self, judge: Judge):
        super().__init__()
        self.judge = judge
        self.features = None

        # Hook to capture features before final classification
        def hook_fn(module, input, output):
            self.features = output.detach()

        # Register hook on features module (conv layers)
        # Judge architecture: self.features (conv) → self.classifier (FC)
        self.judge.features.register_forward_hook(hook_fn)

    def forward(self, images: torch.Tensor):
        """Forward pass capturing features."""
        _ = self.judge(images)
        # Flatten spatial dimensions
        features = self.features.flatten(start_dim=1)
        return features


def generate_samples_pedagogical(
    checkpoint_path: Path,
    num_classes: int,
    samples_per_class: int,
    latent_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """Generate samples from pedagogical Weaver."""
    print(f"Loading pedagogical checkpoint: {checkpoint_path}")

    weaver = Weaver(latent_dim=latent_dim, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    weaver.load_state_dict(checkpoint['models']['weaver'])
    weaver.to(device)
    weaver.eval()

    all_images = []
    all_labels = []

    with torch.no_grad():
        for class_idx in range(num_classes):
            z = torch.randn(samples_per_class, latent_dim, device=device)
            labels = torch.full((samples_per_class,), class_idx, device=device, dtype=torch.long)

            images, _ = weaver(z, labels)
            all_images.append(images)
            all_labels.append(labels.cpu().numpy())

    images = torch.cat(all_images, dim=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"  Generated {len(images)} pedagogical samples")
    return images, labels


def generate_samples_adversarial(
    checkpoint_path: Path,
    num_classes: int,
    samples_per_class: int,
    latent_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """Generate samples from adversarial Generator."""
    print(f"Loading adversarial checkpoint: {checkpoint_path}")

    generator = Generator(latent_dim=latent_dim, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['models']['generator'])
    generator.to(device)
    generator.eval()

    all_images = []
    all_labels = []

    with torch.no_grad():
        for class_idx in range(num_classes):
            z = torch.randn(samples_per_class, latent_dim, device=device)
            labels = torch.full((samples_per_class,), class_idx, device=device, dtype=torch.long)

            images = generator(z, labels)
            all_images.append(images)
            all_labels.append(labels.cpu().numpy())

    images = torch.cat(all_images, dim=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"  Generated {len(images)} adversarial samples")
    return images, labels


def extract_judge_features(
    images: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Extract Judge features from images."""
    judge = Judge()
    judge.to(device)
    judge.eval()

    extractor = JudgeFeatureExtractor(judge)

    with torch.no_grad():
        features = extractor(images)

    return features.cpu().numpy()


def visualize_features(
    ped_features: np.ndarray,
    ped_labels: np.ndarray,
    adv_features: np.ndarray,
    adv_labels: np.ndarray,
    output_path: Path,
):
    """Visualize Judge features via t-SNE and UMAP."""
    print("\nComputing dimensionality reductions...")

    # Combine for joint embedding
    all_features = np.vstack([ped_features, adv_features])
    n_ped = len(ped_features)

    # t-SNE
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_embedded = tsne.fit_transform(all_features)

    # UMAP
    print("  Running UMAP...")
    umap_reducer = UMAP(n_components=2, random_state=42)
    umap_embedded = umap_reducer.fit_transform(all_features)

    # Split back
    tsne_ped = tsne_embedded[:n_ped]
    tsne_adv = tsne_embedded[n_ped:]
    umap_ped = umap_embedded[:n_ped]
    umap_adv = umap_embedded[n_ped:]

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # t-SNE: Pedagogical
    ax = axes[0, 0]
    scatter = ax.scatter(tsne_ped[:, 0], tsne_ped[:, 1],
                        c=ped_labels, cmap='tab10', alpha=0.6, s=20)
    ax.set_title('t-SNE: Pedagogical Samples (Judge Features)', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='Digit Class')

    # t-SNE: Adversarial
    ax = axes[0, 1]
    scatter = ax.scatter(tsne_adv[:, 0], tsne_adv[:, 1],
                        c=adv_labels, cmap='tab10', alpha=0.6, s=20)
    ax.set_title('t-SNE: Adversarial Samples (Judge Features)', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='Digit Class')

    # UMAP: Pedagogical
    ax = axes[1, 0]
    scatter = ax.scatter(umap_ped[:, 0], umap_ped[:, 1],
                        c=ped_labels, cmap='tab10', alpha=0.6, s=20)
    ax.set_title('UMAP: Pedagogical Samples (Judge Features)', fontsize=14)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Digit Class')

    # UMAP: Adversarial
    ax = axes[1, 1]
    scatter = ax.scatter(umap_adv[:, 0], umap_adv[:, 1],
                        c=adv_labels, cmap='tab10', alpha=0.6, s=20)
    ax.set_title('UMAP: Adversarial Samples (Judge Features)', fontsize=14)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Digit Class')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")

    print("\n" + "="*80)
    print("OBSERVATIONS TO MAKE:")
    print("="*80)
    print("Look for structural differences in Judge feature space:")
    print("  - Are pedagogical samples more/less separated by class?")
    print("  - Is manifold structure simpler/more complex?")
    print("  - Different topology (connected components, holes)?")
    print("  - Linear separability differences?")
    print("\nJudge features capture compositionally-relevant information.")
    print("Differences here suggest structural properties that affect composition.")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Visualize Judge features of generated samples')
    parser.add_argument('--pedagogical', type=Path, required=True)
    parser.add_argument('--adversarial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples-per-class', type=int, default=100)
    parser.add_argument('--latent-dim', type=int, default=64)

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')

    # Generate samples
    print("\nGenerating Pedagogical Samples")
    print("-" * 80)
    ped_images, ped_labels = generate_samples_pedagogical(
        args.pedagogical, 10, args.samples_per_class, args.latent_dim, device
    )

    print("\nGenerating Adversarial Samples")
    print("-" * 80)
    adv_images, adv_labels = generate_samples_adversarial(
        args.adversarial, 10, args.samples_per_class, args.latent_dim, device
    )

    # Extract Judge features
    print("\nExtracting Judge Features")
    print("-" * 80)
    print("  Pedagogical samples...")
    ped_features = extract_judge_features(ped_images, device)
    print(f"    Features shape: {ped_features.shape}")

    print("  Adversarial samples...")
    adv_features = extract_judge_features(adv_images, device)
    print(f"    Features shape: {adv_features.shape}")

    # Visualize
    visualize_features(ped_features, ped_labels, adv_features, adv_labels, args.output)


if __name__ == '__main__':
    main()
