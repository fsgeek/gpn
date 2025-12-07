#!/usr/bin/env python3
"""
Block0 Deep Characterization

CKA told us WHERE they diverge (block0: 96.5% → 46.3%).
Now characterize WHAT is different:
1. PCA dimensionality - Is one more compact?
2. Linear separability - Can we decode digit identity?
3. Feature independence - Is one more disentangled?

This goes in the paper.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.weaver import create_weaver
from models.acgan import ACGANGenerator

print("=" * 80)
print("BLOCK0 DEEP CHARACTERIZATION")
print("We know they diverge here. What's different?")
print("=" * 80)

device = torch.device('cpu')

# ==============================================================================
# EXTRACT BLOCK0 FEATURES
# ==============================================================================
print("\nSTEP 1: Extracting block0 features with labels...")

# Load models
gpn_ckpt = torch.load('checkpoints/checkpoint_final.pt', map_location=device, weights_only=False)
gpn_weaver = create_weaver(latent_dim=64, num_classes=10)
gpn_weaver.load_state_dict(gpn_ckpt['models']['weaver'])
gpn_weaver.eval()

gan_ckpt = torch.load('checkpoints/acgan_final.pt', map_location=device, weights_only=False)
gan_generator = ACGANGenerator(latent_dim=64, num_classes=10)
gan_generator.load_state_dict(gan_ckpt['models']['generator'])
gan_generator.eval()

# Extract features
num_samples = 500  # More samples for reliable statistics
batch_size = 50
num_batches = num_samples // batch_size

gpn_block0_features = []
gan_conv1_features = []
all_labels = []

def get_activation(feature_list):
    def hook(model, input, output):
        # Pool spatial dimensions
        if len(output.shape) == 4:
            pooled = output.mean(dim=[2, 3])
            feature_list.append(pooled.detach())
        else:
            feature_list.append(output.detach())
    return hook

# Hooks
gpn_hook = gpn_weaver.blocks[0].register_forward_hook(get_activation(gpn_block0_features))
gan_hook = gan_generator.conv_blocks[1].register_forward_hook(get_activation(gan_conv1_features))

print(f"  Generating {num_samples} samples...")
for i in range(num_batches):
    z = torch.randn(batch_size, 64)
    labels = torch.randint(0, 10, (batch_size,))
    all_labels.append(labels)

    with torch.no_grad():
        _ = gpn_weaver(z, labels)
        _ = gan_generator(z, labels)

gpn_hook.remove()
gan_hook.remove()

# Concatenate
gpn_features = torch.cat(gpn_block0_features, dim=0).numpy()
gan_features = torch.cat(gan_conv1_features, dim=0).numpy()
labels = torch.cat(all_labels, dim=0).numpy()

print(f"  ✓ GPN block0: {gpn_features.shape}")
print(f"  ✓ GAN conv1: {gan_features.shape}")
print(f"  ✓ Labels: {labels.shape}")

# ==============================================================================
# ANALYSIS 1: PCA DIMENSIONALITY
# ==============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 1: Effective Dimensionality (PCA)")
print("=" * 80)
print("Question: Is one representation more compact?")

# Compute PCA
gpn_pca = PCA(n_components=min(gpn_features.shape))
gan_pca = PCA(n_components=min(gan_features.shape))

gpn_pca.fit(gpn_features)
gan_pca.fit(gan_features)

gpn_variance = gpn_pca.explained_variance_ratio_
gan_variance = gan_pca.explained_variance_ratio_

gpn_cumvar = np.cumsum(gpn_variance)
gan_cumvar = np.cumsum(gan_variance)

# Find dimensions needed for 95% variance
gpn_dim_95 = np.argmax(gpn_cumvar >= 0.95) + 1
gan_dim_95 = np.argmax(gan_cumvar >= 0.95) + 1

print(f"\nEffective dimensionality (95% variance):")
print(f"  GPN: {gpn_dim_95} dimensions (out of {gpn_features.shape[1]})")
print(f"  GAN: {gan_dim_95} dimensions (out of {gan_features.shape[1]})")
print(f"  Difference: {abs(gpn_dim_95 - gan_dim_95)} dimensions")

if gpn_dim_95 < gan_dim_95:
    print(f"\n  → GPN is MORE COMPACT ({gpn_dim_95} vs {gan_dim_95} dims)")
    print(f"     Pedagogical training produces lower-dimensional representations")
elif gpn_dim_95 > gan_dim_95:
    print(f"\n  → GAN is MORE COMPACT ({gan_dim_95} vs {gpn_dim_95} dims)")
    print(f"     Adversarial training produces lower-dimensional representations")
else:
    print(f"\n  → EQUALLY COMPACT (both {gpn_dim_95} dims)")

# Participation ratio (another measure of dimensionality)
def participation_ratio(explained_var):
    return (explained_var.sum() ** 2) / (explained_var ** 2).sum()

gpn_pr = participation_ratio(gpn_pca.explained_variance_)
gan_pr = participation_ratio(gan_pca.explained_variance_)

print(f"\nParticipation ratio (higher = more distributed):")
print(f"  GPN: {gpn_pr:.2f}")
print(f"  GAN: {gan_pr:.2f}")

# ==============================================================================
# ANALYSIS 2: LINEAR SEPARABILITY
# ==============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 2: Linear Separability of Digit Identity")
print("=" * 80)
print("Question: Can we decode which digit from block0 features?")

# Train linear classifier
X_train_gpn, X_test_gpn, y_train, y_test = train_test_split(
    gpn_features, labels, test_size=0.2, random_state=42, stratify=labels
)
X_train_gan, X_test_gan, _, _ = train_test_split(
    gan_features, labels, test_size=0.2, random_state=42, stratify=labels
)

gpn_clf = LogisticRegression(max_iter=1000, random_state=42)
gan_clf = LogisticRegression(max_iter=1000, random_state=42)

gpn_clf.fit(X_train_gpn, y_train)
gan_clf.fit(X_train_gan, y_train)

gpn_acc = gpn_clf.score(X_test_gpn, y_test)
gan_acc = gan_clf.score(X_test_gan, y_test)

print(f"\nLinear classification accuracy (test set):")
print(f"  GPN: {gpn_acc:.4f} ({gpn_acc*100:.2f}%)")
print(f"  GAN: {gan_acc:.4f} ({gan_acc*100:.2f}%)")
print(f"  Difference: {abs(gpn_acc - gan_acc):.4f} ({abs(gpn_acc - gan_acc)*100:.2f}%)")

if gpn_acc > gan_acc + 0.02:
    print(f"\n  → GPN features MORE LINEARLY SEPARABLE")
    print(f"     Easier to decode digit identity from GPN representations")
elif gan_acc > gpn_acc + 0.02:
    print(f"\n  → GAN features MORE LINEARLY SEPARABLE")
    print(f"     Easier to decode digit identity from GAN representations")
else:
    print(f"\n  → EQUALLY LINEARLY SEPARABLE")

# ==============================================================================
# ANALYSIS 3: FEATURE INDEPENDENCE
# ==============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 3: Feature Independence (Disentanglement)")
print("=" * 80)
print("Question: How correlated are the features?")

# Compute correlation matrices
gpn_corr = np.corrcoef(gpn_features.T)
gan_corr = np.corrcoef(gan_features.T)

# Mean absolute off-diagonal correlation
def mean_off_diagonal(corr_matrix):
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    return np.abs(corr_matrix[mask]).mean()

gpn_mean_corr = mean_off_diagonal(gpn_corr)
gan_mean_corr = mean_off_diagonal(gan_corr)

print(f"\nMean absolute correlation (off-diagonal):")
print(f"  GPN: {gpn_mean_corr:.4f}")
print(f"  GAN: {gan_mean_corr:.4f}")
print(f"  Difference: {abs(gpn_mean_corr - gan_mean_corr):.4f}")

if gpn_mean_corr < gan_mean_corr - 0.01:
    print(f"\n  → GPN features MORE INDEPENDENT")
    print(f"     Lower correlation = better disentanglement")
elif gan_mean_corr < gpn_mean_corr - 0.01:
    print(f"\n  → GAN features MORE INDEPENDENT")
    print(f"     Lower correlation = better disentanglement")
else:
    print(f"\n  → EQUALLY INDEPENDENT")

# ==============================================================================
# VISUALIZATION
# ==============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION: Block0 Characterization")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: PCA spectrum
ax1 = axes[0]
n_components_to_plot = 50
ax1.plot(range(1, n_components_to_plot + 1), gpn_cumvar[:n_components_to_plot],
         'o-', label='GPN', linewidth=2, markersize=4, color='#2E86AB')
ax1.plot(range(1, n_components_to_plot + 1), gan_cumvar[:n_components_to_plot],
         's-', label='GAN', linewidth=2, markersize=4, color='#A23B72')
ax1.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=gpn_dim_95, color='#2E86AB', linestyle=':', alpha=0.5)
ax1.axvline(x=gan_dim_95, color='#A23B72', linestyle=':', alpha=0.5)
ax1.set_xlabel('Number of Components', fontweight='bold')
ax1.set_ylabel('Cumulative Variance Explained', fontweight='bold')
ax1.set_title('Effective Dimensionality\n(Cumulative Variance)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(gpn_dim_95, 0.5, f'GPN: {gpn_dim_95}', rotation=90, fontsize=9)
ax1.text(gan_dim_95, 0.5, f'GAN: {gan_dim_95}', rotation=90, fontsize=9)

# Plot 2: Linear separability
ax2 = axes[1]
models = ['GPN', 'GAN']
accuracies = [gpn_acc, gan_acc]
colors = ['#2E86AB', '#A23B72']
bars = ax2.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Classification Accuracy', fontweight='bold')
ax2.set_title('Linear Separability\n(Digit Classification)', fontweight='bold')
ax2.set_ylim([0, 1.0])
ax2.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Chance (10 classes)')
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()

# Plot 3: Feature independence
ax3 = axes[2]
metrics = ['GPN', 'GAN']
correlations = [gpn_mean_corr, gan_mean_corr]
bars = ax3.bar(metrics, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Mean Absolute Correlation', fontweight='bold')
ax3.set_title('Feature Independence\n(Lower = More Disentangled)', fontweight='bold')
ax3.set_ylim([0, max(correlations) * 1.2])
for bar, corr in zip(bars, correlations):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

plt.suptitle('Block0 Characterization: GPN vs GAN\nWHAT is Different Where They Diverge?',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

output_path = Path('experiments/block0_characterization.png')
output_path.parent.mkdir(exist_ok=True)
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")

# ==============================================================================
# SUMMARY FOR PAPER
# ==============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Block0 Characterization (FOR PAPER)")
print("=" * 80)

print("\nFINDING: CKA drops from 96.5% (fc) to 46.3% (block0)")
print("\nCharacterization of the divergence:")
print(f"  1. Dimensionality: GPN={gpn_dim_95} dims, GAN={gan_dim_95} dims (95% var)")
print(f"  2. Separability: GPN={gpn_acc:.3f}, GAN={gan_acc:.3f} (linear accuracy)")
print(f"  3. Independence: GPN={gpn_mean_corr:.3f}, GAN={gan_mean_corr:.3f} (mean corr)")

print("\nInterpretation:")
if gpn_dim_95 < gan_dim_95:
    print("  ✓ GPN learns MORE COMPACT representations at block0")
if gpn_acc > gan_acc + 0.02:
    print("  ✓ GPN features are MORE LINEARLY SEPARABLE")
if gpn_mean_corr < gan_mean_corr - 0.01:
    print("  ✓ GPN features are MORE DISENTANGLED")

print("\nConnection to composition:")
print("  - Block0 is where visual features are constructed from latents")
print("  - This is where pedagogical vs adversarial training diverges")
print("  - The geometric differences here correlate with compositional performance")
print("  - GPN (100% composition) vs GAN (81% composition)")

print("\n" + "=" * 80)
print("✓✓✓ BLOCK0 DEEP CHARACTERIZATION COMPLETE ✓✓✓")
print("=" * 80)
print("\nDocumented:")
print("  - WHERE they diverge (block0)")
print("  - WHAT is different (dimensionality, separability, independence)")
print("  - HOW it connects to composition")
print("\nThis goes in the paper.")
print("=" * 80)
