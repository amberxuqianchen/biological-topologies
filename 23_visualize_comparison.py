#!/usr/bin/env python3
"""
Create publication-quality visualizations comparing TDA methods.

This script reads from the computed data and generates figures dynamically.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)

print("="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading data...")

# Load bifiltration data
bifilt_df = pd.read_csv('computed_data/tda_bifiltration_features.csv')

# Load perturbation data
ad_perturb = pd.read_csv('computed_data/tda_perturbation_alzheimers.csv')
cand_perturb = pd.read_csv('computed_data/tda_perturbation_top_candidates.csv')

# Get background gene IDs from bifiltration
bg_gene_ids = set(bifilt_df[bifilt_df['is_ad'] == False]['node_id'].values)
cand_filtered = cand_perturb[cand_perturb['node_id'].isin(bg_gene_ids)]
perturb_df = pd.concat([ad_perturb, cand_filtered], ignore_index=True)

# Merge
merged = bifilt_df.merge(
    perturb_df[['node_id', 'delta_H0', 'delta_H1', 'delta_H2', 'delta_H3']], 
    on='node_id', how='inner', suffixes=('', '_perturb')
)

print(f"  Merged data: {len(merged)} genes")

y = merged['is_ad'].astype(int).values
ad = bifilt_df[bifilt_df['is_ad'] == True]
bg = bifilt_df[bifilt_df['is_ad'] == False]

# ============================================================================
# DEFINE FEATURE SETS
# ============================================================================

GRAPH = ['degree', 'ego_original_size', 'ego_actual_size']
PERTURB_TDA = ['delta_H0', 'delta_H1', 'delta_H2', 'delta_H3']

# Dynamically build bifiltration feature list (including H3)
BIFILT_DELTA = []
BIFILT_DERIVED = []
for dim in [1, 2, 3]:
    for pct in [10, 25, 50, 75, 90, 100]:
        BIFILT_DELTA.append(f'delta_H{dim}_ptm{pct}')
    BIFILT_DERIVED.extend([f'delta_H{dim}_ptm_slope', f'delta_H{dim}_ptm_range'])

BIFILT_ALL = BIFILT_DELTA + BIFILT_DERIVED

# ============================================================================
# COMPUTE AUROC VALUES
# ============================================================================

print("\nComputing AUROC values...")

def get_auroc(features):
    available = [f for f in features if f in merged.columns]
    if len(available) == 0:
        return 0.5, 0.0
    X = merged[available].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')
    return np.mean(scores), np.std(scores)

graph_auroc, graph_std = get_auroc(GRAPH)
perturb_auroc, perturb_std = get_auroc(GRAPH + PERTURB_TDA)
bifilt_auroc, bifilt_std = get_auroc(GRAPH + [f for f in BIFILT_ALL if f in merged.columns])
both_auroc, both_std = get_auroc(GRAPH + PERTURB_TDA + [f for f in BIFILT_ALL if f in merged.columns])

print(f"  Graph only:        {graph_auroc:.3f} ± {graph_std:.3f}")
print(f"  Graph + Perturb:   {perturb_auroc:.3f} ± {perturb_std:.3f}")
print(f"  Graph + Bifilt:    {bifilt_auroc:.3f} ± {bifilt_std:.3f}")
print(f"  Graph + Both:      {both_auroc:.3f} ± {both_std:.3f}")

# ============================================================================
# FIGURE 1: AUROC Comparison Bar Chart
# ============================================================================

print("\nCreating Figure 1: AUROC comparison...")

fig, ax = plt.subplots(figsize=(12, 7))

methods = ['Graph\nOnly', 'Graph +\nPerturbation', 'Graph +\nBifiltration', 'Graph + Perturb\n+ Bifiltration']
aurocs = [graph_auroc, perturb_auroc, bifilt_auroc, both_auroc]
errors = [graph_std, perturb_std, bifilt_std, both_std]
colors = ['#95a5a6', '#3498db', '#e74c3c', '#9b59b6']

bars = ax.bar(methods, aurocs, yerr=errors, capsize=5, color=colors, 
              edgecolor='black', linewidth=1.5)

ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
ax.set_ylabel('AUROC', fontsize=14)
ax.set_title('AD Gene Classification: Comparing TDA Methods', fontsize=16, fontweight='bold')
ax.set_ylim(0.4, 0.85)

# Add value labels
for bar, val in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

# Add improvement annotation
improvement = (bifilt_auroc - perturb_auroc) * 100
ax.annotate('', xy=(2, bifilt_auroc), xytext=(1, perturb_auroc),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(1.5, (bifilt_auroc + perturb_auroc)/2 + 0.03, 
        f'+{improvement:.1f}%', ha='center', fontsize=11, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/tda_method_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/tda_method_comparison.png")

# ============================================================================
# FIGURE 2: Feature Importance (TDA only)
# ============================================================================

print("\nCreating Figure 2: Feature importance...")

# Compute feature importance dynamically
tda_features = [f for f in PERTURB_TDA + BIFILT_ALL if f in merged.columns]
X = merged[tda_features].fillna(0).values

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X, y)

importance_df = pd.DataFrame({
    'feature': tda_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#e74c3c' if 'ptm' in f else '#3498db' for f in importance_df['feature']]
bars = ax.barh(range(len(importance_df)), importance_df['importance'].values, color=colors)
ax.set_yticks(range(len(importance_df)))
ax.set_yticklabels(importance_df['feature'].values, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (Random Forest)', fontsize=12)
ax.set_title('Top 15 TDA Features for AD Classification', fontsize=14, fontweight='bold')

legend_elements = [
    mpatches.Patch(facecolor='#e74c3c', label='Bifiltration TDA'),
    mpatches.Patch(facecolor='#3498db', label='Perturbation TDA'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig('figures/tda_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/tda_feature_importance.png")

# ============================================================================
# FIGURE 3: Bifiltration Across PTM Thresholds (H1, H2, H3)
# ============================================================================

print("\nCreating Figure 3: Bifiltration analysis...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
thresholds = [10, 25, 50, 75, 90, 100]

for idx, dim in enumerate([1, 2, 3]):
    ax = axes[idx]
    
    ad_vals = [ad[f'delta_H{dim}_ptm{t}'].mean() for t in thresholds]
    bg_vals = [bg[f'delta_H{dim}_ptm{t}'].mean() for t in thresholds]
    ad_err = [ad[f'delta_H{dim}_ptm{t}'].std() / np.sqrt(len(ad)) for t in thresholds]
    bg_err = [bg[f'delta_H{dim}_ptm{t}'].std() / np.sqrt(len(bg)) for t in thresholds]
    
    ax.errorbar(thresholds, ad_vals, yerr=ad_err, marker='o', capsize=3, 
                color='red', label='AD genes', linewidth=2, markersize=8)
    ax.errorbar(thresholds, bg_vals, yerr=bg_err, marker='s', capsize=3, 
                color='blue', label='Background', linewidth=2, markersize=8)
    ax.set_xlabel('PTM Percentile', fontsize=11)
    ax.set_ylabel(f'ΔH{dim} (perturbation impact)', fontsize=11)
    ax.set_title(f'H{dim} Perturbation vs PTM Level', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/bifiltration_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/bifiltration_analysis.png")

# ============================================================================
# FIGURE 4: Slope Comparison (H1, H2, H3)
# ============================================================================

print("\nCreating Figure 4: Slope comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['AD genes', 'Background']
x = np.arange(len(categories))
width = 0.25

colors = ['#e74c3c', '#3498db', '#2ecc71']
for i, dim in enumerate([1, 2, 3]):
    slopes = [ad[f'delta_H{dim}_ptm_slope'].mean(), bg[f'delta_H{dim}_ptm_slope'].mean()]
    ax.bar(x + (i - 1) * width, slopes, width, label=f'ΔH{dim} slope', color=colors[i])

ax.set_ylabel('Slope (change per PTM percentile)', fontsize=11)
ax.set_title('PTM-Response Slopes by Dimension', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=10)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/slope_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/slope_comparison.png")

print("\n" + "="*60)
print("✅ All visualizations created!")
print("="*60)
