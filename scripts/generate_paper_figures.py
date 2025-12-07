#!/usr/bin/env python3
"""
Generate publication-quality figures for compositional generalization paper.

Figures:
1. Phase 1.5 vs 1.6 comparison (coverage boundary)
2. Sample quality comparison (success vs failure)
3. Experimental design diagram
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.relational_weaver import create_relational_weaver


def figure1_coverage_boundary():
    """
    Figure 1: The Coverage Boundary

    Shows Phase 1.5 (0% transfer) vs Phase 1.6 (100% transfer)
    with experimental conditions side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Phase 1.5
    ax1 = axes[0]
    ax1.text(0.5, 0.9, "Phase 1.5: Novel Primitives",
             ha='center', va='top', fontsize=16, fontweight='bold',
             transform=ax1.transAxes)

    ax1.text(0.5, 0.75, "Training: Digits [0-4]\n10 valid X>Y pairs",
             ha='center', va='center', fontsize=12,
             transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax1.text(0.5, 0.5, "Test: Digits [5-9]\nCompletely unseen in\nrelational context",
             ha='center', va='center', fontsize=12,
             transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    ax1.text(0.5, 0.25, "Result: 0% digit accuracy\n68.8% relation accuracy\n(random noise)",
             ha='center', va='center', fontsize=12, fontweight='bold',
             transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.9))

    ax1.text(0.5, 0.05, "FAILURE\nLabel-specific memorization",
             ha='center', va='center', fontsize=14, fontweight='bold',
             color='darkred', transform=ax1.transAxes)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Phase 1.6
    ax2 = axes[1]
    ax2.text(0.5, 0.9, "Phase 1.6: Novel Combinations",
             ha='center', va='top', fontsize=16, fontweight='bold',
             transform=ax2.transAxes)

    ax2.text(0.5, 0.75, "Training: Digits [0-9]\n41 valid X>Y pairs\n(exclude {7>3, 8>2, 9>1, 6>4})",
             ha='center', va='center', fontsize=12,
             transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax2.text(0.5, 0.5, "Test: Hold-out pairs\n{7>3, 8>2, 9>1, 6>4}\nNovel combinations of\nseen digits",
             ha='center', va='center', fontsize=12,
             transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax2.text(0.5, 0.25, "Result: 100% X accuracy\n76.6% Y accuracy\n100% relation accuracy",
             ha='center', va='center', fontsize=12, fontweight='bold',
             transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.9))

    ax2.text(0.5, 0.05, "SUCCESS\nCompositional generalization",
             ha='center', va='center', fontsize=14, fontweight='bold',
             color='darkgreen', transform=ax2.transAxes)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.suptitle("The Coverage Boundary: Same Architecture, Different Coverage → Opposite Outcomes",
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = Path("experiments/figures/fig1_coverage_boundary.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")

    return output_path


def figure2_sample_comparison():
    """
    Figure 2: Sample Quality Comparison

    Shows actual generated images:
    - Phase 1.5 curriculum (success)
    - Phase 1.5 transfer (failure)
    - Phase 1.6 hold-out (success)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    # Row 1: Phase 1.5 Curriculum (Success)
    print("Generating Phase 1.5 curriculum samples...")
    phase15_checkpoint = torch.load(
        "checkpoints/relational_final.pt",
        map_location=device,
        weights_only=False
    )
    weaver15 = create_relational_weaver(
        single_digit_checkpoint="checkpoints/checkpoint_final.pt",
        device=device,
    )
    weaver15.load_state_dict(phase15_checkpoint["models"]["weaver"])
    weaver15.eval()

    curriculum_pairs = [(4, 0), (3, 1), (4, 2), (3, 0)]

    with torch.no_grad():
        for i, (x, y) in enumerate(curriculum_pairs):
            z = torch.randn(1, 64, device=device)
            x_labels = torch.tensor([x], device=device)
            y_labels = torch.tensor([y], device=device)
            img = weaver15.generate_from_digits(z, x_labels, y_labels)

            img_np = img[0, 0].cpu().numpy()
            axes[0, i].imshow(img_np, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f"{x} > {y}", fontsize=12)
            axes[0, i].axis('off')

    axes[0, 0].text(-0.15, 0.5, "Phase 1.5\nCurriculum\n(Seen)",
                    transform=axes[0, 0].transAxes,
                    fontsize=12, fontweight='bold', rotation=90,
                    va='center', ha='right')

    # Row 2: Phase 1.5 Transfer (Failure)
    print("Generating Phase 1.5 transfer samples...")
    transfer_pairs = [(9, 5), (8, 6), (7, 5), (9, 8)]

    with torch.no_grad():
        for i, (x, y) in enumerate(transfer_pairs):
            z = torch.randn(1, 64, device=device)
            x_labels = torch.tensor([x], device=device)
            y_labels = torch.tensor([y], device=device)
            img = weaver15.generate_from_digits(z, x_labels, y_labels)

            img_np = img[0, 0].cpu().numpy()
            axes[1, i].imshow(img_np, cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f"{x} > {y}", fontsize=12)
            axes[1, i].axis('off')

    axes[1, 0].text(-0.15, 0.5, "Phase 1.5\nTransfer\n(Novel digits)",
                    transform=axes[1, 0].transAxes,
                    fontsize=12, fontweight='bold', rotation=90,
                    va='center', ha='right')

    # Row 3: Phase 1.6 Hold-out (Success)
    print("Generating Phase 1.6 hold-out samples...")
    phase16_checkpoint = torch.load(
        "checkpoints/relational_holdout_final.pt",
        map_location=device,
        weights_only=False
    )
    weaver16 = create_relational_weaver(
        single_digit_checkpoint="checkpoints/checkpoint_final.pt",
        device=device,
    )
    weaver16.load_state_dict(phase16_checkpoint["models"]["weaver"])
    weaver16.eval()

    holdout_pairs = [(7, 3), (8, 2), (9, 1), (6, 4)]

    with torch.no_grad():
        for i, (x, y) in enumerate(holdout_pairs):
            z = torch.randn(1, 64, device=device)
            x_labels = torch.tensor([x], device=device)
            y_labels = torch.tensor([y], device=device)
            img = weaver16.generate_from_digits(z, x_labels, y_labels)

            img_np = img[0, 0].cpu().numpy()
            axes[2, i].imshow(img_np, cmap='gray', vmin=0, vmax=1)
            axes[2, i].set_title(f"{x} > {y}", fontsize=12)
            axes[2, i].axis('off')

    axes[2, 0].text(-0.15, 0.5, "Phase 1.6\nHold-out\n(Novel combos)",
                    transform=axes[2, 0].transAxes,
                    fontsize=12, fontweight='bold', rotation=90,
                    va='center', ha='right')

    plt.suptitle("Sample Quality: Coverage Determines Generalization",
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_path = Path("experiments/figures/fig2_sample_comparison.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")

    return output_path


def figure3_experimental_design():
    """
    Figure 3: Experimental Design Diagram

    Visual schematic showing:
    - Pre-trained primitives
    - Frozen vs trainable components
    - Coverage sets
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Architecture diagram
    ax.text(0.5, 0.95, "Experimental Architecture",
            ha='center', fontsize=18, fontweight='bold',
            transform=ax.transAxes)

    # Single-digit weaver (frozen)
    rect1 = mpatches.FancyBboxPatch((0.1, 0.7), 0.35, 0.15,
                                     boxstyle="round,pad=0.01",
                                     linewidth=3, edgecolor='blue',
                                     facecolor='lightblue', alpha=0.5)
    ax.add_patch(rect1)
    ax.text(0.275, 0.775, "Single-Digit Weaver\n(Pre-trained & FROZEN)\nDigits [0-9]",
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Latent splitter (trainable)
    rect2 = mpatches.FancyBboxPatch((0.55, 0.7), 0.35, 0.15,
                                     boxstyle="round,pad=0.01",
                                     linewidth=3, edgecolor='red',
                                     facecolor='lightcoral', alpha=0.5)
    ax.add_patch(rect2)
    ax.text(0.725, 0.775, "Latent Splitter\n(TRAINABLE)\nLearn relational routing",
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Coverage sets
    ax.text(0.5, 0.55, "Coverage Sets",
            ha='center', fontsize=16, fontweight='bold',
            transform=ax.transAxes)

    # Phase 1.5
    rect3 = mpatches.FancyBboxPatch((0.05, 0.3), 0.4, 0.2,
                                     boxstyle="round,pad=0.01",
                                     linewidth=2, edgecolor='black',
                                     facecolor='#ffe6e6', alpha=0.8)
    ax.add_patch(rect3)
    ax.text(0.25, 0.45, "Phase 1.5", fontsize=14, fontweight='bold', ha='center')
    ax.text(0.25, 0.40, "Training: [0,4] (10 pairs)", fontsize=10, ha='center')
    ax.text(0.25, 0.36, "Test: [5,9] (novel digits)", fontsize=10, ha='center')
    ax.text(0.25, 0.32, "Result: 0% transfer ✗", fontsize=11,
            fontweight='bold', color='red', ha='center')

    # Phase 1.6
    rect4 = mpatches.FancyBboxPatch((0.55, 0.3), 0.4, 0.2,
                                     boxstyle="round,pad=0.01",
                                     linewidth=2, edgecolor='black',
                                     facecolor='#e6ffe6', alpha=0.8)
    ax.add_patch(rect4)
    ax.text(0.75, 0.45, "Phase 1.6", fontsize=14, fontweight='bold', ha='center')
    ax.text(0.75, 0.40, "Training: [0,9] (41 pairs)", fontsize=10, ha='center')
    ax.text(0.75, 0.36, "Test: Hold-out combos", fontsize=10, ha='center')
    ax.text(0.75, 0.32, "Result: 100% transfer ✓", fontsize=11,
            fontweight='bold', color='green', ha='center')

    # Key insight
    ax.text(0.5, 0.15, "Key Insight:", fontsize=14, fontweight='bold', ha='center')
    ax.text(0.5, 0.09, "\"Primitive competence\" (good digit generator) ≠ \"compositional license\" (seen in relational context)",
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    output_path = Path("experiments/figures/fig3_experimental_design.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")

    return output_path


if __name__ == "__main__":
    print("Generating publication-quality figures...")
    print("=" * 60)

    figure1_coverage_boundary()
    print()

    figure2_sample_comparison()
    print()

    figure3_experimental_design()
    print()

    print("=" * 60)
    print("All figures generated successfully!")
    print("Output directory: experiments/figures/")
