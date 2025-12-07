"""
Calibration Analysis: Epistemic Honesty Probe

Tests whether pedagogical training produces models that "know when they're wrong" -
specifically on compositional tasks where uncertainty matters most.

Core Question: Does the teacher's epistemic honesty (blotchy, can't fake texture)
transfer to the student's uncertainty calibration?

Expected: GPN better calibrated than GAN, especially on compositional OOD cases.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def compute_expected_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between confidence and accuracy across bins.
    A well-calibrated model's confidence should match its accuracy.

    Args:
        confidences: Max softmax probability for each prediction [N]
        predictions: Predicted class labels [N]
        labels: Ground truth labels [N]
        n_bins: Number of bins for calibration curve

    Returns:
        ece: Expected Calibration Error (lower is better)
        calibration_data: Dict with bins, accuracies, confidences, counts
    """
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Check correctness
    accuracies = (predictions == labels).astype(float)

    # Initialize storage
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = in_bin.sum()

        if bin_count > 0:
            # Compute accuracy and avg confidence in this bin
            bin_accuracy = accuracies[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()

            # Contribution to ECE
            ece += (bin_count / len(confidences)) * abs(bin_accuracy - bin_confidence)

            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

        bin_counts.append(bin_count)

    calibration_data = {
        'bin_boundaries': bin_boundaries,
        'bin_accuracies': np.array(bin_accuracies),
        'bin_confidences': np.array(bin_confidences),
        'bin_counts': np.array(bin_counts),
    }

    return ece, calibration_data


def plot_reliability_diagram(
    calibration_data: Dict[str, np.ndarray],
    title: str = "Reliability Diagram",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot reliability diagram showing confidence vs accuracy.

    A perfectly calibrated model falls on the diagonal.
    Overconfident models are above diagonal, underconfident below.

    Args:
        calibration_data: Output from compute_expected_calibration_error
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Extract data
    bin_boundaries = calibration_data['bin_boundaries']
    bin_accuracies = calibration_data['bin_accuracies']
    bin_confidences = calibration_data['bin_confidences']
    bin_counts = calibration_data['bin_counts']

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')

    # Plot actual calibration
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    # Width proportional to count (show which bins have more samples)
    max_count = bin_counts.max() if bin_counts.max() > 0 else 1
    bar_widths = 0.08 * (bin_counts / max_count)

    for i, (center, acc, conf, width) in enumerate(
        zip(bin_centers, bin_accuracies, bin_confidences, bar_widths)
    ):
        if bin_counts[i] > 0:
            # Draw bar from confidence to accuracy
            ax.bar(
                conf, acc, width=width,
                alpha=0.6, edgecolor='blue', linewidth=2,
                color='lightblue' if acc >= conf else 'lightcoral'
            )
            # Mark the point
            ax.plot(conf, acc, 'bo', markersize=8)

    # Shading: overconfident (above diagonal) vs underconfident (below)
    ax.fill_between([0, 1], [0, 1], 1, alpha=0.1, color='red',
                     label='Overconfident Region')
    ax.fill_between([0, 1], 0, [0, 1], alpha=0.1, color='blue',
                     label='Underconfident Region')

    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def compute_calibration_on_dataset(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    task_type: str = "classification",
) -> Tuple[float, Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration metrics for a model on a dataset.

    Args:
        model: Neural network model
        dataloader: DataLoader providing (inputs, labels)
        device: Device to run on
        task_type: "classification" or "relational"

    Returns:
        ece: Expected Calibration Error
        calibration_data: Bin statistics
        confidences: All confidence scores
        predictions: All predictions
        labels: All true labels
    """
    model.eval()

    all_confidences = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(inputs)

            # Get predictions and confidences
            probs = F.softmax(logits, dim=1)
            confidences, predictions = probs.max(dim=1)

            all_confidences.append(confidences.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batches
    confidences = np.concatenate(all_confidences)
    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)

    # Compute ECE
    ece, calibration_data = compute_expected_calibration_error(
        confidences, predictions, labels
    )

    return ece, calibration_data, confidences, predictions, labels


def compare_calibration(
    gpn_ece: float,
    gpn_calibration: Dict[str, np.ndarray],
    gan_ece: float,
    gan_calibration: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of GPN vs GAN calibration.

    Args:
        gpn_ece: GPN Expected Calibration Error
        gpn_calibration: GPN calibration data
        gan_ece: GAN Expected Calibration Error
        gan_calibration: GAN calibration data
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure with dual reliability diagrams
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot GPN
    plot_reliability_on_axis(
        ax1, gpn_calibration,
        f"Pedagogical (GPN)\nECE = {gpn_ece:.4f}",
        color='green'
    )

    # Plot GAN
    plot_reliability_on_axis(
        ax2, gan_calibration,
        f"Adversarial (GAN)\nECE = {gan_ece:.4f}",
        color='red'
    )

    # Overall title
    delta_ece = gan_ece - gpn_ece
    fig.suptitle(
        f"Epistemic Honesty: Does Pedagogical Training Transfer?\n"
        f"ΔECE = {delta_ece:.4f} ({'GPN more calibrated' if delta_ece > 0 else 'GAN more calibrated'})",
        fontsize=16, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_reliability_on_axis(
    ax: plt.Axes,
    calibration_data: Dict[str, np.ndarray],
    title: str,
    color: str = 'blue',
):
    """Helper to plot reliability diagram on existing axis."""
    bin_boundaries = calibration_data['bin_boundaries']
    bin_accuracies = calibration_data['bin_accuracies']
    bin_confidences = calibration_data['bin_confidences']
    bin_counts = calibration_data['bin_counts']

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect')

    # Actual calibration
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    max_count = bin_counts.max() if bin_counts.max() > 0 else 1
    bar_widths = 0.08 * (bin_counts / max_count)

    for i, (center, acc, conf, width) in enumerate(
        zip(bin_centers, bin_accuracies, bin_confidences, bar_widths)
    ):
        if bin_counts[i] > 0:
            ax.bar(conf, acc, width=width, alpha=0.6,
                   edgecolor=color, linewidth=2,
                   color=f'light{color}' if acc >= conf else 'lightcoral')
            ax.plot(conf, acc, f'{color[0]}o', markersize=8)

    ax.fill_between([0, 1], [0, 1], 1, alpha=0.1, color='red')
    ax.fill_between([0, 1], 0, [0, 1], alpha=0.1, color='blue')

    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=10)


def compute_compositional_calibration(
    model: torch.nn.Module,
    in_distribution_loader: torch.utils.data.DataLoader,
    compositional_ood_loader: torch.utils.data.DataLoader,
    novel_primitive_loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
) -> Dict[str, Tuple[float, Dict]]:
    """
    Compute calibration separately for:
    1. In-distribution (training pairs)
    2. Compositional OOD (novel combinations, seen primitives)
    3. Novel primitive OOD (unseen digits in relations)

    This is the KEY probe: does model know when compositional reasoning is uncertain?

    Args:
        model: Trained model
        in_distribution_loader: Training pairs
        compositional_ood_loader: Hold-out pairs (novel combinations)
        novel_primitive_loader: Novel primitives in relations (Phase 1.5 style)
        device: Device

    Returns:
        Dictionary mapping split name to (ece, calibration_data)
    """
    results = {}

    # In-distribution
    ece_in, calib_in, conf_in, pred_in, lab_in = compute_calibration_on_dataset(
        model, in_distribution_loader, device
    )
    results['in_distribution'] = (ece_in, calib_in)

    # Compositional OOD
    ece_comp, calib_comp, conf_comp, pred_comp, lab_comp = compute_calibration_on_dataset(
        model, compositional_ood_loader, device
    )
    results['compositional_ood'] = (ece_comp, calib_comp)

    # Novel primitive OOD (if provided)
    if novel_primitive_loader is not None:
        ece_novel, calib_novel, conf_novel, pred_novel, lab_novel = compute_calibration_on_dataset(
            model, novel_primitive_loader, device
        )
        results['novel_primitive_ood'] = (ece_novel, calib_novel)

    return results


def plot_compositional_calibration_comparison(
    gpn_results: Dict[str, Tuple[float, Dict]],
    gan_results: Dict[str, Tuple[float, Dict]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Three-way comparison: In-dist vs Compositional OOD vs Novel Primitive OOD.

    This answers: "Does GPN know when it's uncertain about composition?"

    Args:
        gpn_results: GPN calibration on 3 splits
        gan_results: GAN calibration on 3 splits
        save_path: Optional save path

    Returns:
        Figure with 2x3 grid of reliability diagrams
    """
    n_splits = len(gpn_results)
    fig, axes = plt.subplots(2, n_splits, figsize=(8*n_splits, 14))

    split_names = list(gpn_results.keys())

    for col, split_name in enumerate(split_names):
        # GPN (top row)
        gpn_ece, gpn_calib = gpn_results[split_name]
        plot_reliability_on_axis(
            axes[0, col], gpn_calib,
            f"GPN - {split_name.replace('_', ' ').title()}\nECE = {gpn_ece:.4f}",
            color='green'
        )

        # GAN (bottom row)
        gan_ece, gan_calib = gan_results[split_name]
        plot_reliability_on_axis(
            axes[1, col], gan_calib,
            f"GAN - {split_name.replace('_', ' ').title()}\nECE = {gan_ece:.4f}",
            color='red'
        )

    # Overall title
    fig.suptitle(
        "Compositional Uncertainty Probe: Does Model Know When It's Wrong?\n"
        "Honest Teacher → Honest Student on Compositional Cases?",
        fontsize=18, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def summarize_calibration_results(
    gpn_results: Dict[str, Tuple[float, Dict]],
    gan_results: Dict[str, Tuple[float, Dict]],
) -> str:
    """
    Generate text summary of calibration comparison.

    Returns:
        Formatted string summarizing key findings
    """
    lines = []
    lines.append("=" * 80)
    lines.append("EPISTEMIC HONESTY PROBE: CALIBRATION ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Question: Does the teacher's epistemic honesty transfer to student?")
    lines.append("Method: Expected Calibration Error (ECE) - lower is better")
    lines.append("")

    for split_name in gpn_results.keys():
        gpn_ece, _ = gpn_results[split_name]
        gan_ece, _ = gan_results[split_name]
        delta = gan_ece - gpn_ece

        lines.append(f"{split_name.replace('_', ' ').title()}:")
        lines.append(f"  GPN ECE: {gpn_ece:.4f}")
        lines.append(f"  GAN ECE: {gan_ece:.4f}")
        lines.append(f"  Δ (GAN - GPN): {delta:+.4f} ({'GPN better' if delta > 0 else 'GAN better'})")
        lines.append("")

    # Overall conclusion
    lines.append("-" * 80)

    # Check if GPN wins on compositional OOD specifically
    if 'compositional_ood' in gpn_results:
        gpn_comp_ece = gpn_results['compositional_ood'][0]
        gan_comp_ece = gan_results['compositional_ood'][0]

        if gpn_comp_ece < gan_comp_ece:
            lines.append("✓ FINDING: GPN shows better calibration on compositional OOD cases")
            lines.append("  → Honest teacher produced honest student on novel compositions")
        else:
            lines.append("✗ FINDING: GAN shows better calibration on compositional OOD cases")
            lines.append("  → Epistemic honesty did NOT transfer as hypothesized")

    lines.append("=" * 80)

    return "\n".join(lines)
