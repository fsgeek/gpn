"""
Feature extraction from generator intermediate representations.

Uses forward hooks to capture activations at specified layers.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn


class FeatureExtractor:
    """
    Extract intermediate features from generator networks using forward hooks.

    Usage:
        extractor = FeatureExtractor(generator, layer_names=['conv_blocks.3'])
        features = extractor.extract(z, labels)
        # features['conv_blocks.3'] contains the activations
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
    ) -> None:
        """
        Initialize feature extractor.

        Args:
            model: Generator model to extract features from
            layer_names: List of layer names to hook (e.g., 'conv_blocks.3')
        """
        self.model = model
        self.layer_names = layer_names
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks = []

        # Register forward hooks
        for name in layer_names:
            layer = self._get_layer(name)
            hook = layer.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

    def _get_layer(self, name: str) -> nn.Module:
        """Get layer by dotted name (e.g., 'conv_blocks.3')."""
        parts = name.split('.')
        layer = self.model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer

    def _make_hook(self, name: str):
        """Create hook function that stores activations."""
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def extract(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward pass and extract features.

        Args:
            z: Latent vectors [B, latent_dim]
            labels: Class labels [B]

        Returns:
            Dictionary mapping layer names to extracted features
        """
        self.features = {}
        self.model.eval()

        with torch.no_grad():
            # Handle different generator interfaces
            output = self.model(z, labels)

            # GPN returns (images, v_pred), others return just images
            if isinstance(output, tuple):
                images, _ = output
            else:
                images = output

        return self.features.copy()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        """Cleanup hooks when extractor is destroyed."""
        self.remove_hooks()


def extract_dataset(
    generator: nn.Module,
    layer_names: List[str],
    num_samples: int,
    latent_dim: int,
    num_classes: int,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Extract features for a dataset of generated samples.

    Args:
        generator: Generator model
        layer_names: Layers to extract features from
        num_samples: Number of samples to generate
        latent_dim: Latent dimension
        num_classes: Number of classes
        device: Device to run on
        batch_size: Batch size for generation

    Returns:
        Tuple of (features_dict, labels):
            - features_dict: Dict mapping layer names to concatenated features [N, ...]
            - labels: Generated labels [N]
    """
    extractor = FeatureExtractor(generator, layer_names)

    all_features = {name: [] for name in layer_names}
    all_labels = []

    generator.eval()

    for i in range(0, num_samples, batch_size):
        batch_sz = min(batch_size, num_samples - i)

        z = torch.randn(batch_sz, latent_dim, device=device)
        labels = torch.randint(0, num_classes, (batch_sz,), device=device)

        features = extractor.extract(z, labels)

        for name in layer_names:
            all_features[name].append(features[name].cpu())

        all_labels.append(labels.cpu())

    # Concatenate all batches
    features_dict = {
        name: torch.cat(all_features[name], dim=0)
        for name in layer_names
    }
    labels_tensor = torch.cat(all_labels, dim=0)

    extractor.remove_hooks()

    return features_dict, labels_tensor
