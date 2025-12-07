"""
Relational Weaver: Generates X > Y relational displays

Architecture:
- Composes single-digit Weaver twice (for X and Y positions)
- Adds ">" symbol between digits
- Output: [X][>][Y] as 28x84 image

Key difference from TwoDigitWeaver:
- Must learn relational constraint: X > Y
- Tests whether position-invariant primitives can encode context-dependent semantics
- Same digit "5" means different things as left vs right

Training approach:
- Uses RelationJudge as pedagogical signal
- Maximize P(X > Y) from judge's prediction
- Curriculum: train on [0,4], transfer to [5,9]
"""

import torch
import torch.nn as nn
from src.models.weaver import Weaver, create_weaver


class RelationalWeaver(nn.Module):
    """
    Generator for relational displays: X > Y

    Composes single-digit weaver to create [X][>][Y] images
    where X must be numerically greater than Y.
    """

    def __init__(
        self,
        single_digit_weaver: Weaver,
        latent_dim: int = 64,
        freeze_digits: bool = False,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.single_digit_weaver = single_digit_weaver

        if freeze_digits:
            for param in self.single_digit_weaver.parameters():
                param.requires_grad = False

        # Learnable latent split (similar to TwoDigitWeaver)
        # Maps input latent to separate latents for X and Y positions
        self.latent_splitter = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, latent_dim * 2),  # Output for both X and Y
        )

    def forward(
        self,
        z: torch.Tensor,
        relation_label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate relational display [X][>][Y].

        Args:
            z: Latent noise [B, latent_dim]
            relation_label: Relation index 0-44 for curriculum [0,4] pairs,
                           or 0-99 for full [0,9] range (maps to X,Y digits)

        Returns:
            Generated [X][>][Y] images [B, 1, 28, 84]
        """
        B = z.size(0)
        device = z.device

        # Decompose relation_label into X and Y digits
        # For curriculum [0,4]: 0-44 maps to valid (X,Y) pairs where X > Y
        # For full range [0,9]: use same logic as TwoDigitWeaver (tens/ones)

        # Check if curriculum mode (labels < 45) or full mode (labels < 100)
        max_label = relation_label.max().item()

        if max_label < 45:
            # Curriculum mode: decode relation index to (X, Y) pairs
            # Valid pairs for [0,4]: (1,0), (2,0), (2,1), (3,0), (3,1), (3,2), ...
            # Total: 1+2+3+4 = 10 pairs
            x_digit, y_digit = self._decode_curriculum_labels(relation_label)
        else:
            # Full mode: use 2-digit decomposition (0-99)
            x_digit = relation_label // 10  # Tens place
            y_digit = relation_label % 10   # Ones place

        # Split latent into X and Y components
        latent_pair = self.latent_splitter(z)  # [B, latent_dim * 2]
        z_x = latent_pair[:, :self.latent_dim]
        z_y = latent_pair[:, self.latent_dim:]

        # Generate X and Y digits (unpack tuple: Weaver returns (image, logits))
        x_img, _ = self.single_digit_weaver(z_x, x_digit)  # [B, 1, 28, 28]
        y_img, _ = self.single_digit_weaver(z_y, y_digit)  # [B, 1, 28, 28]

        # Create ">" symbol (same as in dataset)
        gt_symbol = torch.zeros(B, 1, 28, 28, device=device)
        for i in range(14):
            gt_symbol[:, 0, 7 + i, 14 + i] = 1.0  # Upper diagonal
            gt_symbol[:, 0, 21 - i, 14 + i] = 1.0  # Lower diagonal

        # Concatenate: [X][>][Y]
        relation_img = torch.cat([x_img, gt_symbol, y_img], dim=3)  # [B, 1, 28, 84]

        return relation_img

    def _decode_curriculum_labels(
        self,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode curriculum relation labels to (X, Y) digit pairs.

        For digit_range [0, 4], valid relations are:
        0: (1,0), 1: (2,0), 2: (2,1), 3: (3,0), 4: (3,1), 5: (3,2),
        6: (4,0), 7: (4,1), 8: (4,2), 9: (4,3)

        General formula for X > Y:
        - X ranges from 1 to max_digit
        - For each X, Y ranges from 0 to X-1
        - Index = sum(1..X-1) + Y = X(X-1)/2 + Y

        Inverse: given index, find (X, Y)
        - X = ceil(sqrt(2 * index + 0.25) + 0.5)
        - Y = index - X(X-1)/2
        """
        device = labels.device
        B = labels.size(0)

        x_digit = torch.zeros(B, dtype=torch.long, device=device)
        y_digit = torch.zeros(B, dtype=torch.long, device=device)

        # Decode each label
        for i in range(B):
            idx = labels[i].item()

            # Find X such that X(X-1)/2 <= idx < X(X+1)/2
            x = 1
            cumsum = 0
            while cumsum + x <= idx:
                cumsum += x
                x += 1

            y = idx - cumsum

            x_digit[i] = x
            y_digit[i] = y

        return x_digit, y_digit

    def generate_from_digits(
        self,
        z: torch.Tensor,
        x_digit: torch.Tensor,
        y_digit: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate relational display directly from X and Y digit labels.

        Used for Phase 1.6 holdout training where we specify exact (X,Y) pairs.

        Args:
            z: Latent noise [B, latent_dim]
            x_digit: Left digit labels [B]
            y_digit: Right digit labels [B]

        Returns:
            Generated [X][>][Y] images [B, 1, 28, 84]
        """
        B = z.size(0)
        device = z.device

        # Split latent into X and Y components
        latent_pair = self.latent_splitter(z)  # [B, latent_dim * 2]
        z_x = latent_pair[:, :self.latent_dim]
        z_y = latent_pair[:, self.latent_dim:]

        # Generate X and Y digits
        x_img, _ = self.single_digit_weaver(z_x, x_digit)  # [B, 1, 28, 28]
        y_img, _ = self.single_digit_weaver(z_y, y_digit)  # [B, 1, 28, 28]

        # Create ">" symbol
        gt_symbol = torch.zeros(B, 1, 28, 28, device=device)
        for i in range(14):
            gt_symbol[:, 0, 7 + i, 14 + i] = 1.0  # Upper diagonal
            gt_symbol[:, 0, 21 - i, 14 + i] = 1.0  # Lower diagonal

        # Concatenate: [X][>][Y]
        relation_img = torch.cat([x_img, gt_symbol, y_img], dim=3)  # [B, 1, 28, 84]

        return relation_img


def create_relational_weaver(
    single_digit_checkpoint: str,
    latent_dim: int = 64,
    device: torch.device | None = None,
    freeze_digits: bool = False,
) -> RelationalWeaver:
    """
    Create RelationalWeaver from single-digit checkpoint.

    Args:
        single_digit_checkpoint: Path to trained single-digit Weaver
        latent_dim: Latent dimension
        device: Device to load to
        freeze_digits: Whether to freeze single-digit weights

    Returns:
        RelationalWeaver model
    """
    device = device or torch.device("cpu")

    # Load single-digit weaver
    checkpoint = torch.load(
        single_digit_checkpoint,
        map_location=device,
        weights_only=False,
    )

    # Extract config if available
    config = checkpoint.get("config", {})
    checkpoint_latent_dim = config.get("latent_dim", latent_dim)

    single_digit_weaver = create_weaver(
        latent_dim=checkpoint_latent_dim,
        num_classes=10,
    )

    single_digit_weaver.load_state_dict(checkpoint["models"]["weaver"])
    single_digit_weaver = single_digit_weaver.to(device)

    # Create relational weaver
    relational_weaver = RelationalWeaver(
        single_digit_weaver=single_digit_weaver,
        latent_dim=checkpoint_latent_dim,
        freeze_digits=freeze_digits,
    )

    return relational_weaver.to(device)
