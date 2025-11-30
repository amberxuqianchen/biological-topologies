#!/usr/bin/env python3
"""
Create publication-quality visualizations comparing TDA methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs('figures', exist_ok=True)

# ============================================================================
# FIGURE 1: AUROC Comparison Bar Chart
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

methods = [
    'Graph\nOnly',
    'Graph +\nPerturbation',
    'Graph +\nBifiltration',
    'Graph + Perturb\n+ Bifiltration',
]
aurocs = [0.514, 0.602, 0.686, 0.683]
errors = [0.064, 0.062, 0.061, 0.060]

colors = ['#95a5a6', '#3498db', '#e74c3c', '#9b59b6']

bars = ax.bar(methods, aurocs, yerr=errors, capsize=5, color=colors, 
              edgecolor='black', linewidth=1.5)

ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='Random (0.5)')
ax.set_ylabel('AUROC', fontsize=14)
ax.set_title('AD Gene Classification: Comparing TDA Methods', fontsize=16, fontweight='bold')
ax.set_ylim(0.4, 0.8)

# Add value labels
for bar, val in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

# Add improvement annotations
ax.annotate('', xy=(2, 0.686), xytext=(1, 0.602),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(1.5, 0.66, '+8.4%\np<0.001', ha='center', fontsize=11, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/tda_method_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/tda_method_comparison.png")

# ============================================================================
# FIGURE 2: Feature Importance (TDA only)
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

features = [
    ('delta_H1_ptm_slope', 0.0697, 'bifilt'),
    ('delta_H2_ptm_slope', 0.0651, 'bifilt'),
    ('delta_H3', 0.0641, 'perturb'),
    ('delta_H1_ptm_range', 0.0594, 'bifilt'),
    ('delta_H2_ptm100', 0.0554, 'bifilt'),
    ('delta_H1', 0.0549, 'perturb'),
    ('delta_H2', 0.0534, 'perturb'),
    ('delta_H1_ptm100', 0.0525, 'bifilt'),
    ('delta_H2_ptm_range', 0.0522, 'bifilt'),
    ('delta_H1_ptm25', 0.0522, 'bifilt'),
]

names = [f[0] for f in features]
importances = [f[1] for f in features]
colors = ['#e74c3c' if f[2] == 'bifilt' else '#3498db' for f in features]

bars = ax.barh(range(len(features)), importances, color=colors)
ax.set_yticks(range(len(features)))
ax.set_yticklabels(names, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (Random Forest)', fontsize=12)
ax.set_title('Top 10 TDA Features for AD Classification', fontsize=14, fontweight='bold')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#e74c3c', label='Bifiltration TDA'),
    mpatches.Patch(facecolor='#3498db', label='Perturbation TDA'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig('figures/tda_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/tda_feature_importance.png")

# ============================================================================
# FIGURE 3: Bifiltration Concept Diagram
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Load actual data for the plot
bifilt_df = pd.read_csv('computed_data/tda_bifiltration_features.csv')
ad = bifilt_df[bifilt_df['is_ad'] == True]
bg = bifilt_df[bifilt_df['is_ad'] == False]

thresholds = [10, 25, 50, 75, 90, 100]

# Panel A: H1 across PTM thresholds
ax = axes[0]
ad_h1 = [ad[f'delta_H1_ptm{t}'].mean() for t in thresholds]
bg_h1 = [bg[f'delta_H1_ptm{t}'].mean() for t in thresholds]
ad_h1_err = [ad[f'delta_H1_ptm{t}'].std() / np.sqrt(len(ad)) for t in thresholds]
bg_h1_err = [bg[f'delta_H1_ptm{t}'].std() / np.sqrt(len(bg)) for t in thresholds]

ax.errorbar(thresholds, ad_h1, yerr=ad_h1_err, marker='o', capsize=3, 
            color='red', label='AD genes', linewidth=2, markersize=8)
ax.errorbar(thresholds, bg_h1, yerr=bg_h1_err, marker='s', capsize=3, 
            color='blue', label='Background', linewidth=2, markersize=8)
ax.set_xlabel('PTM Percentile', fontsize=11)
ax.set_ylabel('ΔH₁ (perturbation impact)', fontsize=11)
ax.set_title('A) H₁ Perturbation vs PTM Level', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel B: H2 across PTM thresholds
ax = axes[1]
ad_h2 = [ad[f'delta_H2_ptm{t}'].mean() for t in thresholds]
bg_h2 = [bg[f'delta_H2_ptm{t}'].mean() for t in thresholds]
ad_h2_err = [ad[f'delta_H2_ptm{t}'].std() / np.sqrt(len(ad)) for t in thresholds]
bg_h2_err = [bg[f'delta_H2_ptm{t}'].std() / np.sqrt(len(bg)) for t in thresholds]

ax.errorbar(thresholds, ad_h2, yerr=ad_h2_err, marker='o', capsize=3, 
            color='red', label='AD genes', linewidth=2, markersize=8)
ax.errorbar(thresholds, bg_h2, yerr=bg_h2_err, marker='s', capsize=3, 
            color='blue', label='Background', linewidth=2, markersize=8)
ax.set_xlabel('PTM Percentile', fontsize=11)
ax.set_ylabel('ΔH₂ (perturbation impact)', fontsize=11)
ax.set_title('B) H₂ Perturbation vs PTM Level', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel C: Slope comparison
ax = axes[2]
categories = ['AD genes', 'Background']
h1_slopes = [ad['delta_H1_ptm_slope'].mean(), bg['delta_H1_ptm_slope'].mean()]
h2_slopes = [ad['delta_H2_ptm_slope'].mean(), bg['delta_H2_ptm_slope'].mean()]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, h1_slopes, width, label='ΔH₁ slope', color='#e74c3c')
bars2 = ax.bar(x + width/2, h2_slopes, width, label='ΔH₂ slope', color='#3498db')

ax.set_ylabel('Slope (change per PTM percentile)', fontsize=11)
ax.set_title('C) PTM-Response Slopes', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=10)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/bifiltration_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/bifiltration_analysis.png")

print("\n✅ All visualizations created!")

