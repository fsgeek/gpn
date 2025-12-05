#!/usr/bin/env python3
"""
Test whether the single-digit Weaver inside RelationalWeaver
can still generate transfer digits [5-9] after relational training.

This answers el jefe's question: did we lose access to the primitives?
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.relational_weaver import create_relational_weaver
from models.weaver import Weaver


def test_single_digit_generation():
    """Test if single-digit weaver can still generate all digits."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained RelationalWeaver
    checkpoint_path = Path("checkpoints/relational_final.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create weaver with original checkpoint
    weaver = create_relational_weaver(
        single_digit_checkpoint="checkpoints/checkpoint_final.pt",
        device=device,
    )
    weaver.load_state_dict(checkpoint["models"]["weaver"])
    weaver.eval()

    # Extract the single-digit weaver
    single_digit_weaver = weaver.single_digit_weaver

    print("Testing single-digit generation capabilities...")
    print("=" * 60)

    # Test all digits [0-9]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    with torch.no_grad():
        for digit in range(10):
            z = torch.randn(1, 64, device=device)
            labels = torch.tensor([digit], device=device)

            # Generate using the single-digit weaver
            img, logits = single_digit_weaver(z, labels)

            # Visualize
            img_np = img[0, 0].cpu().numpy()
            axes[digit].imshow(img_np, cmap='gray', vmin=0, vmax=1)
            axes[digit].set_title(f"Digit {digit}")
            axes[digit].axis('off')

            # Check quality
            pred_digit = logits.argmax(dim=1).item()
            is_curriculum = digit < 5
            status = "✓" if pred_digit == digit else "✗"

            print(f"Digit {digit} ({'curriculum' if is_curriculum else 'transfer '}): "
                  f"Predicted {pred_digit} {status}")

    plt.suptitle("Single-Digit Weaver After Relational Training",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = Path("experiments/single_digit_primitives_test.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    return output_path


def compare_frozen_vs_trained():
    """Compare frozen vs trained single-digit weaver."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("COMPARING: Frozen vs Trained Single-Digit Weaver")
    print("=" * 60)

    # Check if single-digit weaver was frozen during training
    checkpoint_path = Path("checkpoints/relational_final.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    weaver = create_relational_weaver(
        single_digit_checkpoint="checkpoints/checkpoint_final.pt",
        device=device,
        freeze_digits=False,  # Don't freeze, we'll check params
    )
    weaver.load_state_dict(checkpoint["models"]["weaver"])

    # Check if parameters have requires_grad
    single_weaver_params = list(weaver.single_digit_weaver.parameters())
    learnable = sum(p.requires_grad for p in single_weaver_params)

    print(f"Single-digit weaver parameters: {len(single_weaver_params)}")
    print(f"Learnable parameters: {learnable}")
    print(f"Was frozen during training: {learnable == 0}")

    # Compare with original checkpoint
    original_ckpt = torch.load("checkpoints/checkpoint_final.pt",
                               map_location=device, weights_only=False)
    original_weaver_state = original_ckpt["models"]["weaver"]

    # Check if states match
    current_state = weaver.single_digit_weaver.state_dict()

    differences = []
    for key in original_weaver_state.keys():
        if key in current_state:
            original_param = original_weaver_state[key]
            current_param = current_state[key]

            if not torch.allclose(original_param, current_param, atol=1e-6):
                diff = (original_param - current_param).abs().max().item()
                differences.append((key, diff))

    if differences:
        print(f"\n⚠ Single-digit weaver WAS modified during training!")
        print(f"Modified parameters: {len(differences)}")
        print("Top 5 largest changes:")
        for key, diff in sorted(differences, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {key}: max diff = {diff:.6f}")
    else:
        print(f"\n✓ Single-digit weaver was perfectly preserved (frozen)")


if __name__ == "__main__":
    test_single_digit_generation()
    compare_frozen_vs_trained()
