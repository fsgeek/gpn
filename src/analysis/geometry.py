"""
Representation Geometry Analysis

Analyzes the geometric structure of learned representations to understand
how pedagogical vs adversarial training shapes feature spaces.

Core analyses:
- PCA: Effective dimensionality and variance structure
- Linear separability: How easily can classes be separated at each layer?
- Feature correlations: Degree of entanglement vs independence

Hypothesis: Pedagogical training produces more compact, linearly separable,
and disentangled representations that support compositional transfer.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def compute_pca_spectrum(
    features: np.ndarray,
    n_components: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Compute PCA spectrum (eigenvalues) of features.

    The spectrum reveals how many dimensions are actually being used:
    - Steep decay: low effective dimensionality (compact representation)
    - Slow decay: high effective dimensionality (spread across many dims)

    Args:
        features: Feature matrix [N, D]
        n_components: Number of PCA components (None = all)

    Returns:
        eigenvalues: Variance explained by each component
        cumulative_variance: Cumulative variance explained
        pca: Fitted PCA object
    """
    if n_components is None:
        n_components = min(features.shape)

    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(features)

    # Get variance explained
    eigenvalues = pca.explained_variance_
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    return eigenvalues, cumulative_variance, pca


def compute_effective_dimensionality(
    features: np.ndarray,
    threshold: float = 0.95,
) -> int:
    """
    Compute effective dimensionality: number of components needed
    to explain `threshold` fraction of variance.

    Lower effective dimensionality = more compact representation.

    Args:
        features: Feature matrix [N, D]
        threshold: Variance threshold (default 0.95 = 95%)

    Returns:
        Number of dimensions needed to explain threshold variance
    """
    _, cumulative_variance, _ = compute_pca_spectrum(features)

    # Find first component where cumulative variance exceeds threshold
    effective_dim = np.searchsorted(cumulative_variance, threshold) + 1

    return int(effective_dim)


def compute_participation_ratio(eigenvalues: np.ndarray) -> float:
    """
    Compute participation ratio: measure of how many dimensions actively used.

    PR = (sum eigenvalues)^2 / sum(eigenvalues^2)

    PR = 1: Only one dimension used
    PR = D: All D dimensions used equally

    Lower PR = more structured/compact representation.

    Args:
        eigenvalues: PCA eigenvalues

    Returns:
        Participation ratio
    """
    sum_eigs = np.sum(eigenvalues)
    sum_eigs_sq = np.sum(eigenvalues ** 2)

    if sum_eigs_sq == 0:
        return 0.0

    pr = (sum_eigs ** 2) / sum_eigs_sq

    return pr


def measure_linear_separability(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
) -> float:
    """
    Measure how linearly separable classes are in feature space.

    Trains a linear classifier (logistic regression) and returns accuracy.
    Higher accuracy = more linearly separable.

    Args:
        features: Feature matrix [N, D]
        labels: Class labels [N]
        test_size: Fraction of data for testing

    Returns:
        Test accuracy of linear classifier
    """
    from sklearn.model_selection import train_test_split

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels, random_state=42
    )

    # Train linear classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def compute_feature_correlations(features: np.ndarray) -> np.ndarray:
    """
    Compute pairwise correlation matrix of features.

    High correlations = entangled, redundant features.
    Low correlations = independent, disentangled features.

    Args:
        features: Feature matrix [N, D]

    Returns:
        Correlation matrix [D, D]
    """
    # Center features
    features_centered = features - features.mean(axis=0, keepdims=True)

    # Compute correlation
    correlation = np.corrcoef(features_centered, rowvar=False)

    return correlation


def measure_feature_independence(features: np.ndarray) -> float:
    """
    Measure degree of feature independence (disentanglement).

    Returns mean absolute correlation (off-diagonal).
    Lower = more independent features.

    Args:
        features: Feature matrix [N, D]

    Returns:
        Mean absolute off-diagonal correlation
    """
    corr_matrix = compute_feature_correlations(features)

    # Get off-diagonal elements
    n_features = corr_matrix.shape[0]
    mask = ~np.eye(n_features, dtype=bool)
    off_diag = corr_matrix[mask]

    # Mean absolute correlation
    mean_abs_corr = np.abs(off_diag).mean()

    return mean_abs_corr


def analyze_layer_geometry(
    features: np.ndarray,
    labels: np.ndarray,
    layer_name: str,
) -> Dict:
    """
    Complete geometric analysis of a single layer.

    Args:
        features: Feature matrix [N, D]
        labels: Class labels [N]
        layer_name: Name of layer

    Returns:
        Dictionary with all geometric metrics
    """
    results = {
        'layer_name': layer_name,
        'n_samples': features.shape[0],
        'n_features': features.shape[1],
    }

    # PCA analysis
    eigenvalues, cumulative_variance, pca = compute_pca_spectrum(features)
    results['eigenvalues'] = eigenvalues
    results['cumulative_variance'] = cumulative_variance
    results['effective_dim_95'] = compute_effective_dimensionality(features, 0.95)
    results['effective_dim_99'] = compute_effective_dimensionality(features, 0.99)
    results['participation_ratio'] = compute_participation_ratio(eigenvalues)

    # Separability
    results['linear_separability'] = measure_linear_separability(features, labels)

    # Feature independence
    results['mean_feature_correlation'] = measure_feature_independence(features)

    return results


def plot_pca_spectrum_comparison(
    gpn_eigenvalues: np.ndarray,
    gan_eigenvalues: np.ndarray,
    layer_name: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Compare PCA spectra between GPN and GAN.

    Args:
        gpn_eigenvalues: GPN eigenvalues
        gan_eigenvalues: GAN eigenvalues
        layer_name: Name of layer
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Eigenvalue spectrum
    n_gpn = len(gpn_eigenvalues)
    n_gan = len(gan_eigenvalues)
    max_components = min(50, n_gpn, n_gan)  # Plot first 50 for clarity

    ax1.plot(range(1, max_components+1), gpn_eigenvalues[:max_components],
             'g-o', linewidth=2, markersize=4, label='GPN (Pedagogical)')
    ax1.plot(range(1, max_components+1), gan_eigenvalues[:max_components],
             'r-s', linewidth=2, markersize=4, label='GAN (Adversarial)')
    ax1.set_xlabel('Component Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue (Variance)', fontsize=12)
    ax1.set_title('PCA Spectrum', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Cumulative variance
    gpn_cumvar = np.cumsum(gpn_eigenvalues) / np.sum(gpn_eigenvalues)
    gan_cumvar = np.cumsum(gan_eigenvalues) / np.sum(gan_eigenvalues)

    ax2.plot(range(1, max_components+1), gpn_cumvar[:max_components],
             'g-o', linewidth=2, markersize=4, label='GPN (Pedagogical)')
    ax2.plot(range(1, max_components+1), gan_cumvar[:max_components],
             'r-s', linewidth=2, markersize=4, label='GAN (Adversarial)')
    ax2.axhline(y=0.95, color='k', linestyle='--', linewidth=1, label='95% threshold')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Variance Explained', fontsize=12)
    ax2.set_title('Effective Dimensionality', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    fig.suptitle(
        f'Representation Geometry: {layer_name}\n'
        f'Steeper decay = lower effective dimensionality = more compact',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_geometric_metrics_comparison(
    gpn_metrics: List[Dict],
    gan_metrics: List[Dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Multi-panel comparison of geometric metrics across layers.

    Args:
        gpn_metrics: List of metric dicts for GPN layers
        gan_metrics: List of metric dicts for GAN layers
        save_path: Optional save path

    Returns:
        Figure with 4 subplots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Extract layer names (assume same for both)
    layer_names = [m['layer_name'] for m in gpn_metrics]
    x_pos = np.arange(len(layer_names))

    # Panel 1: Effective Dimensionality (95%)
    gpn_effdim = [m['effective_dim_95'] for m in gpn_metrics]
    gan_effdim = [m['effective_dim_95'] for m in gan_metrics]

    width = 0.35
    axes[0, 0].bar(x_pos - width/2, gpn_effdim, width, label='GPN', color='green', alpha=0.7)
    axes[0, 0].bar(x_pos + width/2, gan_effdim, width, label='GAN', color='red', alpha=0.7)
    axes[0, 0].set_ylabel('Effective Dimensionality', fontsize=11)
    axes[0, 0].set_title('Effective Dimensionality (95% variance)', fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Panel 2: Linear Separability
    gpn_sep = [m['linear_separability'] for m in gpn_metrics]
    gan_sep = [m['linear_separability'] for m in gan_metrics]

    axes[0, 1].bar(x_pos - width/2, gpn_sep, width, label='GPN', color='green', alpha=0.7)
    axes[0, 1].bar(x_pos + width/2, gan_sep, width, label='GAN', color='red', alpha=0.7)
    axes[0, 1].set_ylabel('Linear Accuracy', fontsize=11)
    axes[0, 1].set_title('Linear Separability', fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Panel 3: Feature Independence (lower = better)
    gpn_indep = [m['mean_feature_correlation'] for m in gpn_metrics]
    gan_indep = [m['mean_feature_correlation'] for m in gan_metrics]

    axes[1, 0].bar(x_pos - width/2, gpn_indep, width, label='GPN', color='green', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, gan_indep, width, label='GAN', color='red', alpha=0.7)
    axes[1, 0].set_ylabel('Mean Abs Correlation', fontsize=11)
    axes[1, 0].set_title('Feature Entanglement (lower = more independent)', fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Panel 4: Participation Ratio
    gpn_pr = [m['participation_ratio'] for m in gpn_metrics]
    gan_pr = [m['participation_ratio'] for m in gan_metrics]

    axes[1, 1].bar(x_pos - width/2, gpn_pr, width, label='GPN', color='green', alpha=0.7)
    axes[1, 1].bar(x_pos + width/2, gan_pr, width, label='GAN', color='red', alpha=0.7)
    axes[1, 1].set_ylabel('Participation Ratio', fontsize=11)
    axes[1, 1].set_title('Participation Ratio (lower = more structured)', fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        'Geometric Analysis: GPN vs GAN Across Layers\n'
        'Lower dimensionality + higher separability + lower correlation = better composition?',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def summarize_geometry_analysis(
    gpn_metrics: List[Dict],
    gan_metrics: List[Dict],
) -> str:
    """
    Generate text summary of geometry analysis.

    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("REPRESENTATION GEOMETRY ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # Average across layers
    gpn_avg_effdim = np.mean([m['effective_dim_95'] for m in gpn_metrics])
    gan_avg_effdim = np.mean([m['effective_dim_95'] for m in gan_metrics])

    gpn_avg_sep = np.mean([m['linear_separability'] for m in gpn_metrics])
    gan_avg_sep = np.mean([m['linear_separability'] for m in gan_metrics])

    gpn_avg_corr = np.mean([m['mean_feature_correlation'] for m in gpn_metrics])
    gan_avg_corr = np.mean([m['mean_feature_correlation'] for m in gan_metrics])

    lines.append("Average Across Layers:")
    lines.append(f"  Effective Dimensionality (95%):")
    lines.append(f"    GPN: {gpn_avg_effdim:.1f}")
    lines.append(f"    GAN: {gan_avg_effdim:.1f}")
    lines.append(f"    Δ (GAN - GPN): {gan_avg_effdim - gpn_avg_effdim:+.1f}")
    lines.append("")
    lines.append(f"  Linear Separability:")
    lines.append(f"    GPN: {gpn_avg_sep:.3f}")
    lines.append(f"    GAN: {gan_avg_sep:.3f}")
    lines.append(f"    Δ (GPN - GAN): {gpn_avg_sep - gan_avg_sep:+.3f}")
    lines.append("")
    lines.append(f"  Feature Independence (mean correlation):")
    lines.append(f"    GPN: {gpn_avg_corr:.3f}")
    lines.append(f"    GAN: {gan_avg_corr:.3f}")
    lines.append(f"    Δ (GAN - GPN): {gan_avg_corr - gpn_avg_corr:+.3f}")
    lines.append("")

    # Interpretation
    lines.append("-" * 80)

    findings = []
    if gpn_avg_effdim < gan_avg_effdim * 0.9:
        findings.append("✓ GPN has LOWER effective dimensionality (more compact)")
    if gpn_avg_sep > gan_avg_sep + 0.05:
        findings.append("✓ GPN has HIGHER linear separability")
    if gpn_avg_corr < gan_avg_corr * 0.9:
        findings.append("✓ GPN has MORE INDEPENDENT features (less entangled)")

    if findings:
        lines.append("KEY FINDINGS:")
        for finding in findings:
            lines.append(f"  {finding}")
        lines.append("")
        lines.append("INTERPRETATION: Pedagogical training produces more compact,")
        lines.append("separable, and disentangled representations - consistent with")
        lines.append("better compositional transfer.")
    else:
        lines.append("NO CLEAR GEOMETRIC DIFFERENCES FOUND")
        lines.append("GPN and GAN representations have similar structure.")

    lines.append("=" * 80)

    return "\n".join(lines)


def analyze_representation_geometry(
    gpn_features: Dict[str, np.ndarray],
    gan_features: Dict[str, np.ndarray],
    labels: np.ndarray,
    save_dir: Optional[Path] = None,
) -> Dict:
    """
    Complete geometry analysis pipeline for all layers.

    Args:
        gpn_features: Dict of layer_name -> features [N, D]
        gan_features: Dict of layer_name -> features [N, D]
        labels: Class labels [N]
        save_dir: Optional directory to save outputs

    Returns:
        Dictionary with all results
    """
    results = {}

    # Ensure same layers
    shared_layers = sorted(set(gpn_features.keys()) & set(gan_features.keys()))

    # Analyze each layer
    gpn_metrics = []
    gan_metrics = []

    for layer in shared_layers:
        gpn_metrics.append(
            analyze_layer_geometry(gpn_features[layer], labels, layer)
        )
        gan_metrics.append(
            analyze_layer_geometry(gan_features[layer], labels, layer)
        )

    results['gpn_metrics'] = gpn_metrics
    results['gan_metrics'] = gan_metrics
    results['layers'] = shared_layers

    # Visualizations
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Multi-panel comparison
        plot_geometric_metrics_comparison(
            gpn_metrics, gan_metrics,
            save_path=save_dir / "geometry_comparison.png"
        )

        # PCA spectrum for each layer
        for layer in shared_layers:
            plot_pca_spectrum_comparison(
                gpn_features[layer],
                gan_features[layer],
                layer,
                save_path=save_dir / f"pca_spectrum_{layer}.png"
            )

        # Summary text
        summary = summarize_geometry_analysis(gpn_metrics, gan_metrics)
        (save_dir / "geometry_summary.txt").write_text(summary)
        results['summary'] = summary

    return results
