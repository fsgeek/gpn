"""
Representation Similarity Analysis

Compares learned representations between GPN and GAN to understand
structural differences in how they encode information.

Core metrics:
- CKA (Centered Kernel Alignment): Measures similarity of representation geometry
- SVCCA (Singular Vector CCA): Canonical correlation after dimensionality reduction
- Procrustes Distance: Optimal alignment distance between feature spaces

Question: Do GPN and GAN learn fundamentally different representational structures
despite achieving similar task performance?
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes


def center_kernel(K: np.ndarray) -> np.ndarray:
    """
    Center a kernel matrix.

    Centering removes the mean from each row/column, making the kernel
    invariant to constant shifts in the data.

    Args:
        K: Kernel matrix [N, N]

    Returns:
        Centered kernel matrix [N, N]
    """
    n = K.shape[0]
    unit = np.ones((n, n)) / n
    return K - unit @ K - K @ unit + unit @ K @ unit


def compute_cka(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = 'linear',
) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two representations.

    CKA measures similarity of representational geometry, invariant to
    orthogonal transformations and isotropic scaling.

    A CKA value of 1.0 means identical geometry (up to rotation/scaling).
    A CKA value of 0.0 means completely uncorrelated representations.

    Reference: Kornblith et al. (2019) "Similarity of Neural Network Representations
    Revisited" https://arxiv.org/abs/1905.00414

    Args:
        X: First representation [N, D1] (N samples, D1 features)
        Y: Second representation [N, D2] (N samples, D2 features)
        kernel: Kernel type ('linear' or 'rbf')

    Returns:
        CKA similarity score in [0, 1]
    """
    assert X.shape[0] == Y.shape[0], "Must have same number of samples"

    if kernel == 'linear':
        # Linear kernel: K = X @ X.T
        K = X @ X.T
        L = Y @ Y.T
    elif kernel == 'rbf':
        # RBF kernel (not commonly used for CKA, but available)
        from sklearn.metrics.pairwise import rbf_kernel
        K = rbf_kernel(X)
        L = rbf_kernel(Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Center kernels
    K_c = center_kernel(K)
    L_c = center_kernel(L)

    # Compute CKA
    # CKA(K, L) = <K_c, L_c>_F / sqrt(<K_c, K_c>_F * <L_c, L_c>_F)
    # where <A, B>_F is the Frobenius inner product
    numerator = np.sum(K_c * L_c)  # Frobenius inner product
    denominator = np.sqrt(np.sum(K_c * K_c) * np.sum(L_c * L_c))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def compute_cka_layerwise(
    features_a: Dict[str, np.ndarray],
    features_b: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute pairwise CKA between all layers of two models.

    This produces a matrix showing which layers of model A are similar
    to which layers of model B.

    Args:
        features_a: Dict mapping layer names to features [N, D_i]
        features_b: Dict mapping layer names to features [N, D_j]

    Returns:
        CKA matrix [len(features_a), len(features_b)]
    """
    layers_a = sorted(features_a.keys())
    layers_b = sorted(features_b.keys())

    cka_matrix = np.zeros((len(layers_a), len(layers_b)))

    for i, layer_a in enumerate(layers_a):
        for j, layer_b in enumerate(layers_b):
            X = features_a[layer_a]
            Y = features_b[layer_b]
            cka_matrix[i, j] = compute_cka(X, Y)

    return cka_matrix


def compute_svcca(
    X: np.ndarray,
    Y: np.ndarray,
    threshold: float = 0.99,
) -> Tuple[float, np.ndarray]:
    """
    Compute Singular Vector Canonical Correlation Analysis (SVCCA).

    SVCCA first reduces dimensionality via SVD (keeping components that
    explain `threshold` variance), then computes canonical correlations.

    Reference: Raghu et al. (2017) "SVCCA: Singular Vector Canonical
    Correlation Analysis for Deep Learning Dynamics and Interpretability"
    https://arxiv.org/abs/1706.05806

    Args:
        X: First representation [N, D1]
        Y: Second representation [N, D2]
        threshold: Variance threshold for SVD (default 0.99)

    Returns:
        mean_correlation: Mean of canonical correlations
        correlations: All canonical correlation coefficients
    """
    assert X.shape[0] == Y.shape[0], "Must have same number of samples"

    # Step 1: SVD dimensionality reduction
    def svd_reduce(A, threshold):
        """Reduce dimensionality by keeping top singular vectors."""
        # Center
        A_centered = A - A.mean(axis=0, keepdims=True)

        # SVD
        U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)

        # Compute variance explained
        variance_explained = np.cumsum(S**2) / np.sum(S**2)

        # Keep components up to threshold
        n_components = np.searchsorted(variance_explained, threshold) + 1
        n_components = min(n_components, len(S))

        # Project onto top components
        A_reduced = U[:, :n_components] @ np.diag(S[:n_components])

        return A_reduced

    X_reduced = svd_reduce(X, threshold)
    Y_reduced = svd_reduce(Y, threshold)

    # Step 2: Canonical Correlation Analysis
    correlations = compute_cca(X_reduced, Y_reduced)

    mean_correlation = np.mean(correlations)

    return mean_correlation, correlations


def compute_cca(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute Canonical Correlation Analysis between X and Y.

    Returns the canonical correlation coefficients (sorted descending).

    Args:
        X: First representation [N, D1]
        Y: Second representation [N, D2]

    Returns:
        Canonical correlations [min(D1, D2)]
    """
    n, d1 = X.shape
    _, d2 = Y.shape

    # Center data
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Add small regularization for numerical stability
    reg = 1e-6

    # Covariance matrices
    Cxx = (X.T @ X) / n + reg * np.eye(d1)
    Cyy = (Y.T @ Y) / n + reg * np.eye(d2)
    Cxy = (X.T @ Y) / n

    # Solve generalized eigenvalue problem
    # (Cxy @ Cyy^-1 @ Cyx) @ a = λ^2 @ Cxx @ a
    Cxx_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cxx))
    Cyy_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cyy))

    # Transform to standard eigenvalue problem
    T = Cxx_inv_sqrt.T @ Cxy @ Cyy_inv_sqrt
    U, S, Vt = np.linalg.svd(T, full_matrices=False)

    # Singular values are the canonical correlations
    correlations = S

    # Clip to [0, 1] (numerical errors can push slightly outside)
    correlations = np.clip(correlations, 0, 1)

    return correlations


def compute_procrustes_distance(
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    """
    Compute Procrustes distance between two representations.

    Finds the optimal rotation/reflection to align X with Y,
    then computes the residual distance.

    Lower distance = more similar representational structure.

    Args:
        X: First representation [N, D]
        Y: Second representation [N, D]

    Returns:
        Procrustes distance (normalized Frobenius norm of residual)
    """
    assert X.shape == Y.shape, "Representations must have same shape for Procrustes"

    # Center data
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    # Normalize scale
    X_norm = X_centered / np.linalg.norm(X_centered, 'fro')
    Y_norm = Y_centered / np.linalg.norm(Y_centered, 'fro')

    # Find optimal rotation matrix
    R, _ = orthogonal_procrustes(X_norm, Y_norm)

    # Compute distance after optimal alignment
    distance = np.linalg.norm(X_norm @ R - Y_norm, 'fro')

    return distance


def plot_cka_matrix(
    cka_matrix: np.ndarray,
    layers_a: List[str],
    layers_b: List[str],
    title: str = "CKA Similarity Matrix",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Visualize CKA similarity matrix as heatmap.

    Diagonal elements (if same model) show self-similarity progression.
    Off-diagonal shows cross-model layer correspondences.

    Args:
        cka_matrix: CKA values [len(layers_a), len(layers_b)]
        layers_a: Layer names for rows
        layers_b: Layer names for columns
        title: Plot title
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(cka_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    # Labels
    ax.set_xticks(range(len(layers_b)))
    ax.set_yticks(range(len(layers_a)))
    ax.set_xticklabels(layers_b, rotation=45, ha='right')
    ax.set_yticklabels(layers_a)

    ax.set_xlabel('Model B Layers', fontsize=12)
    ax.set_ylabel('Model A Layers', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CKA Similarity', fontsize=12)

    # Add text annotations
    for i in range(len(layers_a)):
        for j in range(len(layers_b)):
            text = ax.text(j, i, f'{cka_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_cka_comparison(
    cka_gpn_self: np.ndarray,
    cka_gan_self: np.ndarray,
    cka_gpn_gan: np.ndarray,
    layer_names: List[str],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Three-panel comparison:
    1. GPN internal structure (self-similarity)
    2. GAN internal structure (self-similarity)
    3. Cross-model similarity (GPN vs GAN)

    Args:
        cka_gpn_self: GPN layer-layer CKA
        cka_gan_self: GAN layer-layer CKA
        cka_gpn_gan: GPN-GAN cross-model CKA
        layer_names: Names of layers
        save_path: Optional save path

    Returns:
        Figure with 3 heatmaps
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # GPN self-similarity
    im1 = ax1.imshow(cka_gpn_self, cmap='RdYlGn', vmin=0, vmax=1)
    ax1.set_title('GPN Internal Structure\n(Layer-Layer Similarity)', fontweight='bold')
    ax1.set_xlabel('GPN Layers')
    ax1.set_ylabel('GPN Layers')
    setup_heatmap_axes(ax1, layer_names, layer_names)
    plt.colorbar(im1, ax=ax1, label='CKA')

    # GAN self-similarity
    im2 = ax2.imshow(cka_gan_self, cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_title('GAN Internal Structure\n(Layer-Layer Similarity)', fontweight='bold')
    ax2.set_xlabel('GAN Layers')
    ax2.set_ylabel('GAN Layers')
    setup_heatmap_axes(ax2, layer_names, layer_names)
    plt.colorbar(im2, ax=ax2, label='CKA')

    # Cross-model similarity
    im3 = ax3.imshow(cka_gpn_gan, cmap='RdYlGn', vmin=0, vmax=1)
    ax3.set_title('Cross-Model Similarity\n(GPN vs GAN)', fontweight='bold')
    ax3.set_xlabel('GAN Layers')
    ax3.set_ylabel('GPN Layers')
    setup_heatmap_axes(ax3, layer_names, layer_names)
    plt.colorbar(im3, ax=ax3, label='CKA')

    fig.suptitle(
        'Representational Similarity Analysis\n'
        'Do Pedagogical and Adversarial Training Produce Different Geometries?',
        fontsize=16, fontweight='bold', y=1.02
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def setup_heatmap_axes(ax, row_labels, col_labels):
    """Helper to set up heatmap axis labels."""
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)


def summarize_similarity_analysis(
    cka_gpn_self: np.ndarray,
    cka_gan_self: np.ndarray,
    cka_gpn_gan: np.ndarray,
    layer_names: List[str],
) -> str:
    """
    Generate text summary of similarity analysis.

    Returns:
        Formatted string with key findings
    """
    lines = []
    lines.append("=" * 80)
    lines.append("REPRESENTATION SIMILARITY ANALYSIS (CKA)")
    lines.append("=" * 80)
    lines.append("")

    # Mean similarities
    mean_gpn_self = np.mean(cka_gpn_self[np.triu_indices_from(cka_gpn_self, k=1)])
    mean_gan_self = np.mean(cka_gan_self[np.triu_indices_from(cka_gan_self, k=1)])
    mean_cross = np.mean(cka_gpn_gan)

    lines.append("Mean Layer-Layer Similarities:")
    lines.append(f"  GPN (internal): {mean_gpn_self:.4f}")
    lines.append(f"  GAN (internal): {mean_gan_self:.4f}")
    lines.append(f"  GPN-GAN (cross): {mean_cross:.4f}")
    lines.append("")

    # Diagonal analysis (if square matrix)
    if cka_gpn_gan.shape[0] == cka_gpn_gan.shape[1]:
        diag_similarity = np.diagonal(cka_gpn_gan)
        mean_diag = np.mean(diag_similarity)
        lines.append(f"Corresponding Layer Similarity (diagonal): {mean_diag:.4f}")
        lines.append("")

    # Interpretation
    lines.append("-" * 80)
    if mean_cross < 0.5:
        lines.append("✓ FINDING: GPN and GAN have DIFFERENT representational geometries")
        lines.append("  → Training objective shapes internal structure")
    else:
        lines.append("✗ FINDING: GPN and GAN have SIMILAR representational geometries")
        lines.append("  → Differences may lie elsewhere (not in layer representations)")

    lines.append("=" * 80)

    return "\n".join(lines)


def analyze_representation_similarity(
    gpn_features: Dict[str, np.ndarray],
    gan_features: Dict[str, np.ndarray],
    save_dir: Optional[Path] = None,
) -> Dict:
    """
    Complete similarity analysis pipeline.

    Computes:
    1. CKA matrices (self and cross)
    2. SVCCA scores
    3. Visualizations
    4. Summary statistics

    Args:
        gpn_features: Dict of layer_name -> features [N, D]
        gan_features: Dict of layer_name -> features [N, D]
        save_dir: Optional directory to save outputs

    Returns:
        Dictionary with all results
    """
    results = {}

    # Ensure same layers
    shared_layers = sorted(set(gpn_features.keys()) & set(gan_features.keys()))

    # CKA analysis
    cka_gpn_self = compute_cka_layerwise(
        {k: gpn_features[k] for k in shared_layers},
        {k: gpn_features[k] for k in shared_layers}
    )
    cka_gan_self = compute_cka_layerwise(
        {k: gan_features[k] for k in shared_layers},
        {k: gan_features[k] for k in shared_layers}
    )
    cka_gpn_gan = compute_cka_layerwise(
        {k: gpn_features[k] for k in shared_layers},
        {k: gan_features[k] for k in shared_layers}
    )

    results['cka_gpn_self'] = cka_gpn_self
    results['cka_gan_self'] = cka_gan_self
    results['cka_gpn_gan'] = cka_gpn_gan
    results['layers'] = shared_layers

    # SVCCA for each layer pair
    svcca_scores = []
    for layer in shared_layers:
        mean_corr, corrs = compute_svcca(gpn_features[layer], gan_features[layer])
        svcca_scores.append(mean_corr)
    results['svcca_scores'] = np.array(svcca_scores)

    # Visualizations
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Three-panel CKA comparison
        plot_cka_comparison(
            cka_gpn_self, cka_gan_self, cka_gpn_gan,
            shared_layers,
            save_path=save_dir / "cka_comparison.png"
        )

        # Summary text
        summary = summarize_similarity_analysis(
            cka_gpn_self, cka_gan_self, cka_gpn_gan, shared_layers
        )
        (save_dir / "similarity_summary.txt").write_text(summary)
        results['summary'] = summary

    return results
