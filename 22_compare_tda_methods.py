#!/usr/bin/env python3
"""
Direct comparison: Perturbation TDA vs Bifiltration TDA

Is bifiltration actually better, or just more features?
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*70)
print("COMPARING TDA METHODS: PERTURBATION VS BIFILTRATION")
print("="*70)

# Load bifiltration data
bifilt_df = pd.read_csv('computed_data/tda_bifiltration_features.csv')
print(f"\nBifiltration data: {len(bifilt_df)} genes")

# Load perturbation TDA data (split across multiple files)
ad_perturb = pd.read_csv('computed_data/tda_perturbation_alzheimers.csv')
cand_perturb = pd.read_csv('computed_data/tda_perturbation_top_candidates.csv')

# Get the background gene IDs from bifiltration data
bg_gene_ids = set(bifilt_df[bifilt_df['is_ad'] == False]['node_id'].values)

# Filter candidates to only those in our background set
cand_filtered = cand_perturb[cand_perturb['node_id'].isin(bg_gene_ids)]

# Combine AD and background perturbation data
perturb_df = pd.concat([ad_perturb, cand_filtered], ignore_index=True)
print(f"Perturbation data: {len(perturb_df)} genes ({len(ad_perturb)} AD + {len(cand_filtered)} BG)")

# Merge on node_id to get genes with both
merged = bifilt_df.merge(perturb_df[['node_id', 'delta_H0', 'delta_H1', 'delta_H2', 'delta_H3']], 
                          on='node_id', how='inner', suffixes=('', '_perturb'))
print(f"Genes with both: {len(merged)}")

y = merged['is_ad'].astype(int).values

# ============================================================================
# FEATURE SETS
# ============================================================================

GRAPH = ['degree', 'ego_original_size', 'ego_actual_size']

PERTURB_TDA = ['delta_H0', 'delta_H1', 'delta_H2', 'delta_H3']

BIFILT_DELTA = [
    'delta_H1_ptm10', 'delta_H2_ptm10',
    'delta_H1_ptm25', 'delta_H2_ptm25', 
    'delta_H1_ptm50', 'delta_H2_ptm50',
    'delta_H1_ptm75', 'delta_H2_ptm75',
    'delta_H1_ptm90', 'delta_H2_ptm90',
    'delta_H1_ptm100', 'delta_H2_ptm100',
]

BIFILT_DERIVED = [
    'delta_H1_ptm_slope', 'delta_H2_ptm_slope',
    'delta_H1_ptm_range', 'delta_H2_ptm_range',
]

BIFILT_ALL = BIFILT_DELTA + BIFILT_DERIVED

# ============================================================================
# TEST EACH FEATURE SET
# ============================================================================

def evaluate(X, y, name):
    """Evaluate features with repeated CV."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')
    
    return np.mean(scores), np.std(scores)

print("\n" + "="*70)
print("AUROC COMPARISON (10x10 CV)")
print("="*70)
print(f"\n{'Feature Set':<40} {'AUROC':>12} {'Improvement':>15}")
print("-"*70)

baseline = None
results = {}

feature_sets = [
    ('Graph only', GRAPH),
    ('Perturbation TDA only', PERTURB_TDA),
    ('Bifiltration TDA only', BIFILT_ALL),
    ('Graph + Perturbation', GRAPH + PERTURB_TDA),
    ('Graph + Bifiltration', GRAPH + BIFILT_ALL),
    ('Graph + Perturb + Bifilt', GRAPH + PERTURB_TDA + BIFILT_ALL),
    ('Perturb + Bifilt (no graph)', PERTURB_TDA + BIFILT_ALL),
]

for name, features in feature_sets:
    available = [f for f in features if f in merged.columns]
    X = merged[available].fillna(0).values
    mean, std = evaluate(X, y, name)
    
    if baseline is None:
        baseline = mean
        improvement = "baseline"
    else:
        improvement = f"+{(mean - baseline)*100:.1f}%"
    
    results[name] = (mean, std)
    print(f"{name:<40} {mean:.4f} ± {std:.4f}  {improvement:>10}")

# ============================================================================
# KEY QUESTION: Does bifiltration add value BEYOND perturbation?
# ============================================================================

print("\n" + "="*70)
print("KEY QUESTION: Does bifiltration ADD VALUE beyond perturbation?")
print("="*70)

perturb_only = results['Graph + Perturbation'][0]
bifilt_only = results['Graph + Bifiltration'][0]
both = results['Graph + Perturb + Bifilt'][0]

print(f"""
Graph + Perturbation:        {perturb_only:.4f}
Graph + Bifiltration:        {bifilt_only:.4f}
Graph + Perturb + Bifilt:    {both:.4f}

Bifiltration vs Perturbation:  {'+' if bifilt_only > perturb_only else ''}{(bifilt_only - perturb_only)*100:.1f}% AUROC
Adding Bifilt to Perturb:      {'+' if both > perturb_only else ''}{(both - perturb_only)*100:.1f}% AUROC
""")

# ============================================================================
# FEATURE IMPORTANCE FOR PURE TDA FEATURES (no graph)
# ============================================================================

print("="*70)
print("FEATURE IMPORTANCE: TDA FEATURES ONLY")
print("="*70)

tda_features = [f for f in PERTURB_TDA + BIFILT_ALL if f in merged.columns]
X = merged[tda_features].fillna(0).values

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X, y)

importance = pd.DataFrame({
    'feature': tda_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 TDA Features (no graph features):")
print("-"*50)
for _, row in importance.head(10).iterrows():
    feat = row['feature']
    imp = row['importance']
    source = "perturb" if feat in PERTURB_TDA else "bifilt"
    print(f"  {feat:<25} {imp:.4f}  [{source}]")

# ============================================================================
# STATISTICAL TEST: Perturbation vs Bifiltration
# ============================================================================

print("\n" + "="*70)
print("STATISTICAL TEST: Is bifiltration significantly better?")
print("="*70)

from scipy import stats

# Get full CV scores for comparison
def get_cv_scores(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    return cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')

X_perturb = merged[GRAPH + PERTURB_TDA].fillna(0).values
X_bifilt = merged[GRAPH + [f for f in BIFILT_ALL if f in merged.columns]].fillna(0).values

scores_perturb = get_cv_scores(X_perturb, y)
scores_bifilt = get_cv_scores(X_bifilt, y)

t_stat, p_val = stats.ttest_rel(scores_bifilt, scores_perturb)

print(f"""
Paired t-test (same CV folds):
  Perturbation mean: {np.mean(scores_perturb):.4f}
  Bifiltration mean: {np.mean(scores_bifilt):.4f}
  Difference:        {np.mean(scores_bifilt) - np.mean(scores_perturb):.4f}
  t-statistic:       {t_stat:.3f}
  p-value:           {p_val:.2e}
""")

if p_val < 0.001:
    print("  → HIGHLY SIGNIFICANT (p < 0.001)")
elif p_val < 0.05:
    print("  → Significant (p < 0.05)")
else:
    print("  → NOT significant")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
1. Perturbation TDA alone:  {results['Perturbation TDA only'][0]:.3f} AUROC
2. Bifiltration TDA alone:  {results['Bifiltration TDA only'][0]:.3f} AUROC
3. Combined:                {results['Perturb + Bifilt (no graph)'][0]:.3f} AUROC

Bifiltration IS {'better' if bifilt_only > perturb_only else 'NOT better'} than perturbation by {abs(bifilt_only - perturb_only)*100:.1f}% AUROC
The improvement is {'statistically significant' if p_val < 0.05 else 'NOT statistically significant'} (p={p_val:.2e})
""")

