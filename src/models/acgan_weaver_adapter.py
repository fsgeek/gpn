"""
AC-GAN to Weaver adapter.

Wraps ACGANGenerator to match Weaver interface for RelationalWeaver compatibility.
"""

import torch
import torch.nn as nn
from src.models.acgan import ACGANGenerator


class ACGANWeaverAdapter(nn.Module):
    """
    Adapter to make ACGANGenerator compatible with Weaver interface.

    Weaver interface: forward(z, labels) -> (image, logits)
    ACGANGenerator interface: forward(z, labels) -> image

    This adapter adds dummy logits to match Weaver expected output.
    """

    def __init__(self, acgan_generator: ACGANGenerator):
        super().__init__()
        self.generator = acgan_generator
        self.num_classes = acgan_generator.num_classes

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate images using AC-GAN generator.

        Args:
            z: Latent noise [B, latent_dim]
            labels: Class labels [B]

        Returns:
            Tuple of (images, dummy_logits) to match Weaver interface
        """
        images = self.generator(z, labels)

        # Create dummy logits (not used, but required by interface)
        batch_size = z.size(0)
        dummy_logits = torch.zeros(batch_size, self.num_classes, device=z.device)

        return images, dummy_logits
