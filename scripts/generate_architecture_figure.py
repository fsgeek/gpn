#!/usr/bin/env python3
"""
Generate GPN architecture diagram for paper.

Shows Weaver/Witness/Judge triad with information flow and the key
separation between training signal (Witness) and verification (Judge).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')

# Colors
weaver_color = '#3498db'    # Blue
witness_color = '#9b59b6'   # Purple
judge_color = '#e74c3c'     # Red
latent_color = '#95a5a6'    # Gray
output_color = '#2ecc71'    # Green

# Box style
box_style = dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', linewidth=2)

# === WEAVER ===
weaver_box = FancyBboxPatch((0.5, 3), 2.5, 2, boxstyle="round,pad=0.1",
                             facecolor=weaver_color, edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(weaver_box)
ax.text(1.75, 4, 'WEAVER\n(Generator)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# === WITNESS ===
witness_box = FancyBboxPatch((3.75, 5.5), 2.5, 2, boxstyle="round,pad=0.1",
                              facecolor=witness_color, edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(witness_box)
ax.text(5, 6.5, 'WITNESS\n(Evaluator)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# === JUDGE ===
judge_box = FancyBboxPatch((3.75, 0.5), 2.5, 2, boxstyle="round,pad=0.1",
                            facecolor=judge_color, edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(judge_box)
ax.text(5, 1.5, 'JUDGE\n(Ground Truth)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
# Frozen indicator
ax.text(5, 0.3, '(frozen)', ha='center', va='center', fontsize=9, fontstyle='italic', color='gray')

# === OUTPUTS ===
output_box = FancyBboxPatch((7, 3), 2.5, 2, boxstyle="round,pad=0.1",
                             facecolor=output_color, edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(output_box)
ax.text(8.25, 4, 'Generated\nOutputs', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# === LATENT INPUT ===
ax.annotate('Latent z\n+ Labels', xy=(0.5, 4), xytext=(-0.8, 4),
            fontsize=10, ha='center', va='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# === ARROWS ===
arrow_style = dict(arrowstyle='->', color='black', lw=2, connectionstyle='arc3,rad=0')

# Weaver -> Output
ax.annotate('', xy=(7, 4), xytext=(3, 4),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Output -> Witness (images go up to Witness)
ax.annotate('', xy=(5, 5.5), xytext=(7.5, 5),
            arrowprops=dict(arrowstyle='->', color=witness_color, lw=2, connectionstyle='arc3,rad=-0.2'))
ax.text(6.8, 5.6, 'images', fontsize=9, color=witness_color, fontstyle='italic')

# Output -> Judge (images go down to Judge)
ax.annotate('', xy=(5, 2.5), xytext=(7.5, 3),
            arrowprops=dict(arrowstyle='->', color=judge_color, lw=2, connectionstyle='arc3,rad=0.2'))
ax.text(6.8, 2.4, 'images', fontsize=9, color=judge_color, fontstyle='italic')

# Witness -> Weaver (v_seen training signal) - DASHED for training
ax.annotate('', xy=(2.5, 5), xytext=(4.2, 5.5),
            arrowprops=dict(arrowstyle='->', color=witness_color, lw=2,
                          connectionstyle='arc3,rad=0.3', linestyle='dashed'))
ax.text(2.8, 5.7, 'v_seen\n(training)', fontsize=9, color=witness_color, ha='center', fontstyle='italic')

# Judge -> Witness (grounding signal) - DASHED
ax.annotate('', xy=(4.5, 5.5), xytext=(4.5, 2.5),
            arrowprops=dict(arrowstyle='->', color=judge_color, lw=2, linestyle='dashed'))
ax.text(4.1, 4, 'grounding', fontsize=9, color=judge_color, rotation=90, va='center', fontstyle='italic')

# Weaver v_pred (internal prediction)
ax.text(1.75, 2.7, 'v_pred', fontsize=9, color=weaver_color, ha='center', fontstyle='italic')
ax.annotate('', xy=(1.75, 3), xytext=(1.75, 2.5),
            arrowprops=dict(arrowstyle='->', color=weaver_color, lw=1.5))

# === KEY INSIGHT BOX ===
insight_box = FancyBboxPatch((0.2, 0.2), 3, 1.3, boxstyle="round,pad=0.1",
                              facecolor='#f9f9f9', edgecolor='gray', linewidth=1, alpha=0.9)
ax.add_patch(insight_box)
ax.text(1.7, 0.85, 'Key: Weaver trains against\nWitness but is verified by Judge',
        ha='center', va='center', fontsize=9, fontstyle='italic')

# === LEGEND ===
ax.text(8.5, 7.5, 'Solid = data flow', fontsize=9, ha='left')
ax.text(8.5, 7.1, 'Dashed = training signal', fontsize=9, ha='left')

# === TITLE ===
ax.text(5, 7.8, 'GPN Architecture: Constitutional Separation',
        ha='center', va='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/architecture_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('results/figures/architecture_diagram.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: results/figures/architecture_diagram.png")
print("Saved: results/figures/architecture_diagram.pdf")
