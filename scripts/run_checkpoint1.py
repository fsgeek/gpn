#!/usr/bin/env python3
"""
Checkpoint 1: The Representation Story

Orchestrates all Week 1 analyses to answer the core question:
"Does the teacher's epistemic honesty transfer to the student?"

Runs in priority order (per plan):
1. Calibration Analysis [FIRST-CLASS] - Epistemic honesty probe
2. CKA Analysis - Representation similarity
3. PCA Dimensionality - Effective dimensionality
4. Stroke Structure Probe - Edge/topology decoding
5. Linear Separability - Compositional features
6. Spatial Decomposition - Position-invariance probes

Outputs:
- Statistical test results with effect sizes
- Visualizations for all metrics
- Summary report for workshop paper
"""

import sys
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Tuple
import argparse
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import analysis modules
from analysis.calibration import (
    compute_compositional_calibration,
    plot_compositional_calibration_comparison,
    summarize_calibration_results,
)
from analysis.similarity import (
    analyze_representation_similarity,
    plot_cka_comparison,
)
from analysis.geometry import (
    analyze_representation_geometry,
    compare_geometry_metrics,
)
from analysis.probes import (
    DigitIdentityProbe,
    SpatialPositionProbe,
    StrokeStructureProbe,
    train_digit_identity_probe,
    train_spatial_position_probe,
    train_stroke_structure_probe,
    compare_probe_results,
)
from analysis.statistics import (
    compare_metrics_across_layers,
    summarize_statistical_tests,
    StatisticalTest,
)
from analysis.representation import FeatureExtractor

# Import models
from models.single_digit_weaver import SingleDigitWeaver
from models.latent_splitter import LatentSplitter

# Import data
from data.relational_mnist import create_relational_mnist_dataloaders


def load_model_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple:
    """
    Load a trained GPN or GAN model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on

    Returns:
        Tuple of (weaver, splitter, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config
    config = checkpoint.get('config', {})

    # Reconstruct models
    weaver = SingleDigitWeaver(
        latent_dim=config.get('latent_dim', 64),
        hidden_dim=config.get('hidden_dim', 256),
    )

    splitter = LatentSplitter(
        latent_dim=config.get('latent_dim', 64),
        num_digits=10,
    )

    # Load state dicts
    weaver.load_state_dict(checkpoint['weaver_state_dict'])
    splitter.load_state_dict(checkpoint['splitter_state_dict'])

    weaver = weaver.to(device).eval()
    splitter = splitter.to(device).eval()

    return weaver, splitter, config


def extract_features_from_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    layers: List[str],
) -> Dict[str, np.ndarray]:
    """
    Extract intermediate features from specified layers.

    Args:
        model: Model to extract from
        dataloader: Data to run through model
        device: Device
        layers: List of layer names to extract

    Returns:
        Dictionary mapping layer names to feature arrays
    """
    extractor = FeatureExtractor(model, layers)

    all_features = {layer: [] for layer in layers}
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            # Forward pass with feature extraction
            features = extractor.extract(inputs)

            for layer in layers:
                all_features[layer].append(features[layer].cpu().numpy())

            all_labels.append(labels.cpu().numpy())

    # Concatenate
    features_dict = {
        layer: np.concatenate(all_features[layer], axis=0)
        for layer in layers
    }

    labels = np.concatenate(all_labels, axis=0)

    return features_dict, labels


def run_calibration_analysis(
    gpn_model: torch.nn.Module,
    gan_model: torch.nn.Module,
    in_dist_loader: torch.utils.data.DataLoader,
    comp_ood_loader: torch.utils.data.DataLoader,
    novel_prim_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
) -> Tuple[Dict, Dict]:
    """
    Run calibration analysis (Priority 1).

    Tests: Does GPN show better calibration than GAN,
    especially on compositional OOD cases?

    Args:
        gpn_model: Pedagogical model
        gan_model: Adversarial model
        in_dist_loader: In-distribution data
        comp_ood_loader: Compositional OOD (novel combinations)
        novel_prim_loader: Novel primitive OOD
        device: Device
        output_dir: Where to save results

    Returns:
        Tuple of (gpn_results, gan_results)
    """
    print("\n" + "=" * 80)
    print("CALIBRATION ANALYSIS: Epistemic Honesty Probe")
    print("=" * 80)

    # Compute calibration on all splits
    print("\nComputing GPN calibration...")
    gpn_results = compute_compositional_calibration(
        gpn_model, in_dist_loader, comp_ood_loader, novel_prim_loader, device
    )

    print("Computing GAN calibration...")
    gan_results = compute_compositional_calibration(
        gan_model, in_dist_loader, comp_ood_loader, novel_prim_loader, device
    )

    # Visualize
    print("Generating visualizations...")
    fig = plot_compositional_calibration_comparison(
        gpn_results, gan_results, save_path=output_dir / "calibration_comparison.png"
    )

    # Summary
    summary = summarize_calibration_results(gpn_results, gan_results)
    print("\n" + summary)

    # Save summary
    with open(output_dir / "calibration_summary.txt", "w") as f:
        f.write(summary)

    return gpn_results, gan_results


def run_similarity_analysis(
    gpn_features: Dict[str, np.ndarray],
    gan_features: Dict[str, np.ndarray],
    output_dir: Path,
) -> Dict:
    """
    Run representation similarity analysis (Priority 2).

    Tests: Do GPN and GAN have fundamentally different
    representational geometry?

    Args:
        gpn_features: GPN features by layer
        gan_features: GAN features by layer
        output_dir: Where to save results

    Returns:
        Similarity analysis results
    """
    print("\n" + "=" * 80)
    print("SIMILARITY ANALYSIS: Representational Geometry")
    print("=" * 80)

    # Analyze
    print("\nComputing CKA and SVCCA...")
    results = analyze_representation_similarity(
        gpn_features, gan_features, save_dir=output_dir
    )

    # Visualize
    print("Generating CKA matrices...")
    plot_cka_comparison(
        results['cka_gpn_gpn'],
        results['cka_gan_gan'],
        results['cka_gpn_gan'],
        save_path=output_dir / "cka_comparison.png",
    )

    # Summary
    print("\nSimilarity Summary:")
    print(f"  Mean CKA (GPN-GPN): {results['cka_gpn_gpn'].mean():.4f}")
    print(f"  Mean CKA (GAN-GAN): {results['cka_gan_gan'].mean():.4f}")
    print(f"  Mean CKA (GPN-GAN): {results['cka_gpn_gan'].mean():.4f}")

    return results


def run_geometry_analysis(
    gpn_features: Dict[str, np.ndarray],
    gan_features: Dict[str, np.ndarray],
    gpn_labels: np.ndarray,
    gan_labels: np.ndarray,
    output_dir: Path,
) -> Tuple[Dict, Dict]:
    """
    Run geometric structure analysis (Priority 3).

    Tests: Does GPN have lower effective dimensionality
    and more linear separability?

    Args:
        gpn_features: GPN features by layer
        gan_features: GAN features by layer
        gpn_labels: GPN labels
        gan_labels: GAN labels
        output_dir: Where to save results

    Returns:
        Tuple of (gpn_geometry, gan_geometry)
    """
    print("\n" + "=" * 80)
    print("GEOMETRY ANALYSIS: Dimensionality and Separability")
    print("=" * 80)

    # Analyze both models
    print("\nAnalyzing GPN geometry...")
    gpn_geometry = analyze_representation_geometry(
        gpn_features, gpn_labels, save_dir=output_dir / "gpn"
    )

    print("Analyzing GAN geometry...")
    gan_geometry = analyze_representation_geometry(
        gan_features, gan_labels, save_dir=output_dir / "gan"
    )

    # Compare
    print("\nGenerating comparison plots...")
    compare_geometry_metrics(
        gpn_geometry, gan_geometry, save_path=output_dir / "geometry_comparison.png"
    )

    # Summary
    print("\nGeometry Summary:")
    for layer in gpn_geometry.keys():
        print(f"\nLayer: {layer}")
        print(f"  GPN Dimensionality: {gpn_geometry[layer]['effective_dim']}")
        print(f"  GAN Dimensionality: {gan_geometry[layer]['effective_dim']}")
        print(f"  GPN Separability: {gpn_geometry[layer]['separability']:.4f}")
        print(f"  GAN Separability: {gan_geometry[layer]['separability']:.4f}")

    return gpn_geometry, gan_geometry


def run_probe_analysis(
    gpn_features: Dict[str, np.ndarray],
    gan_features: Dict[str, np.ndarray],
    labels: np.ndarray,
    images: np.ndarray,
    device: torch.device,
    output_dir: Path,
) -> Tuple[Dict, Dict]:
    """
    Run linear probe analysis (Priority 4-6).

    Tests all three probes:
    1. Digit Identity (baseline)
    2. Spatial Position (compositional structure)
    3. Stroke Structure (topological encoding)

    Args:
        gpn_features: GPN features by layer
        gan_features: GAN features by layer
        labels: Ground truth labels
        images: Original images for edge extraction
        device: Device
        output_dir: Where to save results

    Returns:
        Tuple of (gpn_probe_results, gan_probe_results)
    """
    print("\n" + "=" * 80)
    print("PROBE ANALYSIS: Compositional Structure Tests")
    print("=" * 80)

    gpn_results = {}
    gan_results = {}

    # Select a representative layer (e.g., middle layer)
    layer = list(gpn_features.keys())[len(gpn_features) // 2]
    print(f"\nUsing layer: {layer}")

    gpn_feat = torch.from_numpy(gpn_features[layer]).float()
    gan_feat = torch.from_numpy(gan_features[layer]).float()
    labels_t = torch.from_numpy(labels).long()
    images_t = torch.from_numpy(images).float()

    # 1. Digit Identity Probe
    print("\n1. Testing Digit Identity Probe...")
    feat_dim = gpn_feat.flatten(1).shape[1]

    gpn_id_probe = DigitIdentityProbe(feat_dim)
    gan_id_probe = DigitIdentityProbe(feat_dim)

    gpn_id_acc = train_digit_identity_probe(gpn_id_probe, gpn_feat, labels_t, device)
    gan_id_acc = train_digit_identity_probe(gan_id_probe, gan_feat, labels_t, device)

    gpn_results['identity_accuracy'] = gpn_id_acc
    gan_results['identity_accuracy'] = gan_id_acc

    print(f"  GPN Identity Accuracy: {gpn_id_acc:.4f}")
    print(f"  GAN Identity Accuracy: {gan_id_acc:.4f}")

    # 2. Spatial Position Probe (if applicable - needs 2-digit images)
    if gpn_feat.ndim == 4 and gpn_feat.shape[-1] > gpn_feat.shape[-2]:
        print("\n2. Testing Spatial Position Probe...")
        # For this demo, we'll skip since we need proper 2-digit data
        # In real implementation, would use relational pairs
        print("  [Skipping - requires 2-digit composition data]")
    else:
        print("\n2. Spatial Position Probe: N/A (requires 2-digit data)")

    # 3. Stroke Structure Probe
    print("\n3. Testing Stroke Structure Probe...")
    if gpn_feat.ndim == 4:  # Spatial features
        channels = gpn_feat.shape[1]

        gpn_stroke_probe = StrokeStructureProbe(channels)
        gan_stroke_probe = StrokeStructureProbe(channels)

        gpn_loss, gpn_iou = train_stroke_structure_probe(
            gpn_stroke_probe, gpn_feat, images_t, device
        )
        gan_loss, gan_iou = train_stroke_structure_probe(
            gan_stroke_probe, gan_feat, images_t, device
        )

        gpn_results['stroke_iou'] = gpn_iou
        gan_results['stroke_iou'] = gan_iou

        print(f"  GPN Stroke IoU: {gpn_iou:.4f}")
        print(f"  GAN Stroke IoU: {gan_iou:.4f}")
    else:
        print("  [Skipping - requires spatial features]")

    # Comparison
    comparison = compare_probe_results(gpn_results, gan_results, "All Probes")
    print("\n" + comparison)

    with open(output_dir / "probe_results.txt", "w") as f:
        f.write(comparison)

    return gpn_results, gan_results


def run_statistical_tests(
    gpn_metrics: Dict[str, List[float]],
    gan_metrics: Dict[str, List[float]],
    metric_names: List[str],
    output_dir: Path,
) -> List[StatisticalTest]:
    """
    Run rigorous statistical tests on all metrics.

    Tests:
    - Paired t-tests (layers matched)
    - Bootstrap confidence intervals
    - Effect sizes (Cohen's d)
    - Multiple testing correction

    Args:
        gpn_metrics: GPN metric values by metric name
        gan_metrics: GAN metric values by metric name
        metric_names: Names of metrics
        output_dir: Where to save results

    Returns:
        List of statistical test results
    """
    print("\n" + "=" * 80)
    print("STATISTICAL TESTING: Rigorous Validation")
    print("=" * 80)

    tests = []

    for metric_name in metric_names:
        if metric_name in gpn_metrics and metric_name in gan_metrics:
            test = compare_metrics_across_layers(
                gpn_metrics[metric_name],
                gan_metrics[metric_name],
                metric_name,
            )
            tests.append(test)

    # Summary
    summary = summarize_statistical_tests(tests, metric_names)
    print("\n" + summary)

    with open(output_dir / "statistical_tests.txt", "w") as f:
        f.write(summary)

    return tests


def main():
    parser = argparse.ArgumentParser(
        description="Run Checkpoint 1 analysis: The Representation Story"
    )
    parser.add_argument(
        "--gpn-checkpoint",
        type=Path,
        required=True,
        help="Path to GPN model checkpoint",
    )
    parser.add_argument(
        "--gan-checkpoint",
        type=Path,
        required=True,
        help="Path to GAN model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/checkpoint1"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation",
    )

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("CHECKPOINT 1: THE REPRESENTATION STORY")
    print("=" * 80)
    print(f"\nGPN Checkpoint: {args.gpn_checkpoint}")
    print(f"GAN Checkpoint: {args.gan_checkpoint}")
    print(f"Output Directory: {output_dir}")
    print(f"Device: {device}")

    # Load models
    print("\nLoading models...")
    gpn_weaver, gpn_splitter, gpn_config = load_model_checkpoint(
        args.gpn_checkpoint, device
    )
    gan_weaver, gan_splitter, gan_config = load_model_checkpoint(
        args.gan_checkpoint, device
    )

    # Create dataloaders
    print("Creating dataloaders...")
    in_dist_loader, comp_ood_loader, novel_prim_loader = (
        create_relational_mnist_dataloaders(
            batch_size=args.batch_size,
            novel_combinations=True,
            novel_primitives=True,
        )
    )

    # RUN ALL ANALYSES (in priority order)

    # 1. Calibration Analysis [FIRST-CLASS]
    gpn_calib, gan_calib = run_calibration_analysis(
        gpn_splitter,
        gan_splitter,
        in_dist_loader,
        comp_ood_loader,
        novel_prim_loader,
        device,
        output_dir / "calibration",
    )

    # 2. Extract features for remaining analyses
    print("\nExtracting features from models...")
    layers_to_extract = ['layer1', 'layer2', 'layer3', 'layer4']

    gpn_features, gpn_labels = extract_features_from_model(
        gpn_weaver, in_dist_loader, device, layers_to_extract
    )
    gan_features, gan_labels = extract_features_from_model(
        gan_weaver, in_dist_loader, device, layers_to_extract
    )

    # Get images for probe analysis
    images = next(iter(in_dist_loader))[0].numpy()

    # 3. Similarity Analysis
    similarity_results = run_similarity_analysis(
        gpn_features, gan_features, output_dir / "similarity"
    )

    # 4. Geometry Analysis
    gpn_geometry, gan_geometry = run_geometry_analysis(
        gpn_features,
        gan_features,
        gpn_labels,
        gan_labels,
        output_dir / "geometry",
    )

    # 5. Probe Analysis
    gpn_probes, gan_probes = run_probe_analysis(
        gpn_features, gan_features, gpn_labels, images, device, output_dir / "probes"
    )

    # 6. Statistical Testing
    # Collect metrics for testing
    metric_names = ['calibration_ece', 'effective_dim', 'separability', 'stroke_iou']
    gpn_metrics = {
        'calibration_ece': [gpn_calib['in_distribution'][0]],
        'effective_dim': [gpn_geometry[layer]['effective_dim'] for layer in layers_to_extract],
        'separability': [gpn_geometry[layer]['separability'] for layer in layers_to_extract],
        'stroke_iou': [gpn_probes.get('stroke_iou', 0.0)],
    }
    gan_metrics = {
        'calibration_ece': [gan_calib['in_distribution'][0]],
        'effective_dim': [gan_geometry[layer]['effective_dim'] for layer in layers_to_extract],
        'separability': [gan_geometry[layer]['separability'] for layer in layers_to_extract],
        'stroke_iou': [gan_probes.get('stroke_iou', 0.0)],
    }

    statistical_tests = run_statistical_tests(
        gpn_metrics, gan_metrics, metric_names, output_dir / "statistics"
    )

    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print("CHECKPOINT 1 COMPLETE: KEY FINDINGS")
    print("=" * 80)

    print("\n1. CALIBRATION (Epistemic Honesty):")
    gpn_comp_ece = gpn_calib['compositional_ood'][0]
    gan_comp_ece = gan_calib['compositional_ood'][0]
    print(f"   GPN Compositional ECE: {gpn_comp_ece:.4f}")
    print(f"   GAN Compositional ECE: {gan_comp_ece:.4f}")
    if gpn_comp_ece < gan_comp_ece:
        print("   ✓ GPN shows better compositional calibration (epistemic honesty)")
    else:
        print("   ✗ GAN shows better compositional calibration")

    print("\n2. REPRESENTATION GEOMETRY:")
    print(f"   Cross-model CKA: {similarity_results['cka_gpn_gan'].mean():.4f}")
    print("   (Lower = more different geometry)")

    print("\n3. EFFECTIVE DIMENSIONALITY:")
    avg_gpn_dim = np.mean(gpn_metrics['effective_dim'])
    avg_gan_dim = np.mean(gan_metrics['effective_dim'])
    print(f"   GPN: {avg_gpn_dim:.1f} dims")
    print(f"   GAN: {avg_gan_dim:.1f} dims")

    print("\n4. COMPOSITIONAL STRUCTURE:")
    if 'stroke_iou' in gpn_probes and 'stroke_iou' in gan_probes:
        print(f"   GPN Stroke IoU: {gpn_probes['stroke_iou']:.4f}")
        print(f"   GAN Stroke IoU: {gan_probes['stroke_iou']:.4f}")

    print("\n" + "=" * 80)
    print(f"All results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
