"""
Two-Digit Weaver: Sequential generator for 2-digit numbers.

Generates 2-digit images by:
1. Using a pre-trained single-digit Weaver to generate each digit
2. Concatenating the digits to form the full number

This enables curriculum learning:
- Phase 0: Train single-digit Weaver (GPN-1)
- Phase 1: Freeze single-digit Weaver, train composition
- Phase 2: Unfreeze, fine-tune end-to-end
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from pathlib import Path

from src.models.weaver import Weaver, create_weaver


class TwoDigitWeaver(nn.Module):
    """
    Sequential generator for 2-digit MNIST numbers.

    Composes two single-digit images into a 2-digit number.

    Args:
        single_digit_weaver: Pre-trained Weaver for single digits
        latent_dim: Latent dimension for generation
    """

    def __init__(
        self,
        single_digit_weaver: Weaver,
        latent_dim: int = 64,
    ):
        super().__init__()

        self.single_digit_weaver = single_digit_weaver
        self.latent_dim = latent_dim

        # Learnable latent split for tens/ones positions
        # This allows the model to learn position-specific generation
        self.latent_split = nn.Linear(latent_dim, latent_dim * 2)

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate 2-digit images.

        Args:
            z: Latent vectors [B, latent_dim]
            labels: Target numbers 0-99 [B]

        Returns:
            images: Generated 2-digit images [B, 1, 28, 56]
            v_pred: Predicted V vectors (concatenated from both digits) [B, v_dim*2]
        """
        batch_size = z.size(0)

        # Decompose labels into tens and ones digits
        tens = labels // 10
        ones = labels % 10

        # Split latent for each position
        z_split = self.latent_split(z)  # [B, latent_dim * 2]
        z_tens = z_split[:, :self.latent_dim]
        z_ones = z_split[:, self.latent_dim:]

        # Generate each digit
        img_tens, v_tens = self.single_digit_weaver(z_tens, tens)  # [B, 1, 28, 28]
        img_ones, v_ones = self.single_digit_weaver(z_ones, ones)  # [B, 1, 28, 28]

        # Concatenate horizontally
        images = torch.cat([img_tens, img_ones], dim=3)  # [B, 1, 28, 56]

        # Concatenate v predictions
        v_pred = torch.cat([v_tens, v_ones], dim=1)  # [B, v_dim * 2]

        return images, v_pred

    def freeze_digit_weaver(self):
        """Freeze the single-digit Weaver (for Phase 1 training)."""
        for param in self.single_digit_weaver.parameters():
            param.requires_grad = False
        print("Single-digit Weaver frozen")

    def unfreeze_digit_weaver(self):
        """Unfreeze the single-digit Weaver (for Phase 2 training)."""
        for param in self.single_digit_weaver.parameters():
            param.requires_grad = True
        print("Single-digit Weaver unfrozen")

    def is_digit_weaver_frozen(self) -> bool:
        """Check if single-digit Weaver is frozen."""
        return not next(self.single_digit_weaver.parameters()).requires_grad


class TwoDigitWeaverDirect(nn.Module):
    """
    Direct generator for 2-digit MNIST (no composition).

    Used as ablation: generates 28x56 images directly without
    decomposing into digits first.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        v_pred_dim: int = 16,
        hidden_dims: list[int] = [256, 512, 1024],
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.v_pred_dim = v_pred_dim

        # Label embedding (100 classes for 2-digit)
        self.label_embed = nn.Embedding(100, latent_dim)

        # Generator network (outputs 28x56)
        self.fc1 = nn.Linear(latent_dim * 2, hidden_dims[0] * 7 * 14)

        self.deconv = nn.Sequential(
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dims[0], hidden_dims[1], 4, 2, 1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dims[1], hidden_dims[2], 4, 2, 1),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dims[2], 1, 3, 1, 1),
            nn.Tanh(),
        )

        # V prediction head
        self.v_pred = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, v_pred_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate 2-digit images directly.

        Args:
            z: Latent vectors [B, latent_dim]
            labels: Target numbers 0-99 [B]

        Returns:
            images: Generated 2-digit images [B, 1, 28, 56]
            v_pred: Predicted V vectors [B, v_pred_dim]
        """
        label_embed = self.label_embed(labels)
        combined = torch.cat([z, label_embed], dim=1)

        # Generate image
        x = self.fc1(combined)
        x = x.view(-1, 256, 7, 14)  # [B, 256, 7, 14]
        images = self.deconv(x)  # [B, 1, 28, 56]

        # V prediction
        v_pred = self.v_pred(combined)

        return images, v_pred


def create_twodigit_weaver(
    single_digit_checkpoint: str,
    latent_dim: int = 64,
    v_pred_dim: int = 16,
    device: Optional[torch.device] = None,
    freeze_digits: bool = True,
) -> TwoDigitWeaver:
    """
    Create a TwoDigitWeaver from a pre-trained single-digit Weaver.

    Args:
        single_digit_checkpoint: Path to single-digit Weaver checkpoint
        latent_dim: Latent dimension
        v_pred_dim: V prediction dimension
        device: Device to load to
        freeze_digits: Whether to freeze the single-digit Weaver

    Returns:
        Initialized TwoDigitWeaver
    """
    device = device or torch.device("cpu")

    # Load single-digit Weaver
    single_weaver = create_weaver(
        latent_dim=latent_dim,
        v_pred_dim=v_pred_dim,
        device=device,
    )

    checkpoint = torch.load(single_digit_checkpoint, map_location=device, weights_only=False)
    if "models" in checkpoint and "weaver" in checkpoint["models"]:
        single_weaver.load_state_dict(checkpoint["models"]["weaver"])
    else:
        single_weaver.load_state_dict(checkpoint)

    # Create TwoDigitWeaver
    model = TwoDigitWeaver(single_weaver, latent_dim=latent_dim).to(device)

    if freeze_digits:
        model.freeze_digit_weaver()

    return model


def create_twodigit_weaver_direct(
    latent_dim: int = 64,
    v_pred_dim: int = 16,
    device: Optional[torch.device] = None,
) -> TwoDigitWeaverDirect:
    """
    Create a direct TwoDigitWeaver (no composition, for ablation).

    Args:
        latent_dim: Latent dimension
        v_pred_dim: V prediction dimension
        device: Device to load to

    Returns:
        Initialized TwoDigitWeaverDirect
    """
    device = device or torch.device("cpu")
    return TwoDigitWeaverDirect(latent_dim, v_pred_dim).to(device)


if __name__ == "__main__":
    # Quick test
    print("Testing TwoDigitWeaver...")

    device = torch.device("cpu")

    # Check if we have a trained single-digit Weaver
    checkpoint_path = "checkpoints/checkpoint_v3_final.pt"
    if not Path(checkpoint_path).exists():
        checkpoint_path = "checkpoints/checkpoint_v3nometa_step3000.pt"

    if Path(checkpoint_path).exists():
        print(f"Loading single-digit Weaver from {checkpoint_path}")
        model = create_twodigit_weaver(
            single_digit_checkpoint=checkpoint_path,
            device=device,
            freeze_digits=True,
        )

        # Test generation
        z = torch.randn(4, 64)
        labels = torch.tensor([42, 17, 99, 0])

        with torch.no_grad():
            images, v_pred = model(z, labels)

        print(f"Generated images shape: {images.shape}")  # Should be [4, 1, 28, 56]
        print(f"V prediction shape: {v_pred.shape}")
        print(f"Labels: {labels.tolist()}")
        print(f"Digit Weaver frozen: {model.is_digit_weaver_frozen()}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        print("Testing direct Weaver instead...")

        model = create_twodigit_weaver_direct(device=device)
        z = torch.randn(4, 64)
        labels = torch.tensor([42, 17, 99, 0])

        images, v_pred = model(z, labels)
        print(f"Generated images shape: {images.shape}")
        print(f"V prediction shape: {v_pred.shape}")
