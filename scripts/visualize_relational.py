#!/usr/bin/env python3
"""
Visualize RelationalWeaver generated samples.

Shows what the weaver learned to generate for both curriculum [0,4]
and transfer [5,9] digits.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.relational_weaver import create_relational_weaver
from models.relation_judge import create_relation_judge
from training.config import TrainingConfig


def visualize_samples(
    weaver,
    judge,
    relations: list[tuple[int, int]],
    title: str,
    n_samples: int = 5,
    device: torch.device = torch.device("cpu"),
):
    """Generate and visualize samples for given relations."""
    weaver.eval()
    judge.eval()

    fig, axes = plt.subplots(n_samples, 1, figsize=(8, 2 * n_samples))
    if n_samples == 1:
        axes = [axes]

    with torch.no_grad():
        for idx, relation_tuple in enumerate(relations[:n_samples]):
            x_digit, y_digit, label = relation_tuple

            # Generate sample
            z = torch.randn(1, 64, device=device)
            relation_label = torch.tensor([label], device=device)

            # Generate image
            img = weaver(z, relation_label)  # [1, 1, 28, 84]

            # Get judge's assessment
            x_logits, y_logits, valid_prob = judge(img)

            x_pred = x_logits.argmax(dim=1).item()
            y_pred = y_logits.argmax(dim=1).item()
            validity = valid_prob.item()

            # Visualize
            img_np = img[0, 0].cpu().numpy()
            axes[idx].imshow(img_np, cmap="gray", vmin=0, vmax=1)
            axes[idx].set_title(
                f"Target: {x_digit} > {y_digit} | "
                f"Judge sees: {x_pred} > {y_pred} | "
                f"Valid: {validity:.1%}"
            )
            axes[idx].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    config = TrainingConfig.from_yaml("configs/gpn1_default.yaml")

    # Load trained models
    print("Loading trained models...")
    checkpoint_path = Path("checkpoints/relational_step5000.pt")
    if not checkpoint_path.exists():
        checkpoint_path = Path("checkpoints/relational_step4000.pt")
    if not checkpoint_path.exists():
        print("Error: No checkpoint found")
        return

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Create weaver from single-digit checkpoint
    single_digit_ckpt = "checkpoints/checkpoint_final.pt"
    weaver = create_relational_weaver(
        single_digit_checkpoint=single_digit_ckpt,
        device=device,
    )
    # Load the trained relational weaver state
    weaver.load_state_dict(checkpoint["models"]["weaver"])
    weaver.eval()

    # Load judge
    judge = create_relation_judge(
        checkpoint_path="checkpoints/relation_judge.pt",
        device=device,
        freeze=True,
    )
    print("Models loaded successfully")

    # Define test relations with proper label encoding
    # Curriculum [0,4]: uses indices 0-9
    curriculum_relations = [
        (4, 0, 6),  # (x, y, label_idx)
        (3, 1, 4),
        (4, 2, 8),
        (3, 0, 3),
        (2, 1, 2),
    ]

    # Transfer [5,9]: uses 2-digit encoding (xy as 10*x + y)
    transfer_relations = [
        (9, 5, 95),  # (x, y, label)
        (8, 6, 86),
        (7, 5, 75),
        (9, 7, 97),
        (6, 5, 65),
    ]

    # Generate curriculum samples
    print("\nGenerating curriculum [0,4] samples...")
    fig1 = visualize_samples(
        weaver,
        judge,
        curriculum_relations,
        "Curriculum [0,4] Samples",
        n_samples=5,
        device=device,
    )
    output_path1 = Path("experiments/relational_curriculum_samples.png")
    fig1.savefig(output_path1, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path1}")

    # Generate transfer samples
    print("\nGenerating transfer [5,9] samples...")
    fig2 = visualize_samples(
        weaver,
        judge,
        transfer_relations,
        "Transfer [5,9] Samples - The Mystery",
        n_samples=5,
        device=device,
    )
    output_path2 = Path("experiments/relational_transfer_samples.png")
    fig2.savefig(output_path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path2}")

    # Also check what the judge actually predicts for transfer
    print("\n" + "=" * 60)
    print("TRANSFER DIGIT ANALYSIS")
    print("=" * 60)

    with torch.no_grad():
        for x_digit, y_digit, label in transfer_relations:
            z = torch.randn(1, 64, device=device)
            relation_label = torch.tensor([label], device=device)
            img = weaver(z, relation_label)

            x_logits, y_logits, valid_prob = judge(img)
            x_pred = x_logits.argmax(dim=1).item()
            y_pred = y_logits.argmax(dim=1).item()

            print(
                f"Target: {x_digit} > {y_digit} | "
                f"Judge: {x_pred} > {y_pred} | "
                f"Valid: {valid_prob.item():.1%} | "
                f"Correct relation: {x_pred > y_pred}"
            )

    plt.show()


if __name__ == "__main__":
    main()
