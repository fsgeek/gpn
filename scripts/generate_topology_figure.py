#!/usr/bin/env python3
"""
Generate topology comparison figure for paper.

Shows intrinsic dimensionality and persistent homology (β₁) differences
between pedagogical and adversarial training.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from topology_analysis_summary.md
metrics = {
    'Intrinsic\nDimensionality': {'Pedagogical': 9.94, 'Adversarial': 13.55},
    'Topological\nHoles (β₁)': {'Pedagogical': 5.6, 'Adversarial': 8.0},
}

# Compositional accuracy for context
composition = {'Pedagogical': 100.0, 'Adversarial': 81.1}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Colors
ped_color = '#2ecc71'  # Green
adv_color = '#e74c3c'  # Red

# Plot 1: Intrinsic Dimensionality
ax1 = axes[0]
x = np.arange(2)
width = 0.6
vals = [metrics['Intrinsic\nDimensionality']['Pedagogical'],
        metrics['Intrinsic\nDimensionality']['Adversarial']]
colors = [ped_color, adv_color]
bars1 = ax1.bar(x, vals, width, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Dimensions', fontsize=12)
ax1.set_title('Intrinsic Dimensionality\n(lower = simpler)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(['Pedagogical', 'Adversarial'], fontsize=11)
ax1.set_ylim(0, 16)
# Add value labels
for bar, val in zip(bars1, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
# Add difference annotation
ax1.annotate('', xy=(0, 13.55), xytext=(1, 13.55),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax1.text(0.5, 14.3, '-36%', ha='center', fontsize=11, fontweight='bold')

# Plot 2: Topological Holes
ax2 = axes[1]
vals = [metrics['Topological\nHoles (β₁)']['Pedagogical'],
        metrics['Topological\nHoles (β₁)']['Adversarial']]
bars2 = ax2.bar(x, vals, width, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Mean Holes (β₁)', fontsize=12)
ax2.set_title('Topological Complexity\n(lower = smoother)', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['Pedagogical', 'Adversarial'], fontsize=11)
ax2.set_ylim(0, 10)
# Add value labels
for bar, val in zip(bars2, vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
# Add difference annotation
ax2.annotate('', xy=(0, 8.0), xytext=(1, 8.0),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax2.text(0.5, 8.6, '-43%', ha='center', fontsize=11, fontweight='bold')

# Plot 3: Compositional Transfer (the outcome)
ax3 = axes[2]
vals = [composition['Pedagogical'], composition['Adversarial']]
bars3 = ax3.bar(x, vals, width, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Accuracy (%)', fontsize=12)
ax3.set_title('Compositional Transfer\n(higher = better)', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(['Pedagogical', 'Adversarial'], fontsize=11)
ax3.set_ylim(0, 110)
ax3.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
# Add value labels
for bar, val in zip(bars3, vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
# Add difference annotation
ax3.annotate('', xy=(0, 81.1), xytext=(1, 81.1),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax3.text(0.5, 73, '+18.9%', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/topology_comparison.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('results/figures/topology_comparison.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: results/figures/topology_comparison.png")
print("Saved: results/figures/topology_comparison.pdf")

# plt.show()  # Skip interactive display
