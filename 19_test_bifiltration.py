#!/usr/bin/env python3
"""
Test if Bifiltration Features Improve Classification

Compares:
1. Graph features only
2. Graph + original TDA (perturbation)
3. Graph + bifiltration TDA
4. Graph + both TDA types
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_features():
    """Load and merge all feature sets."""
    print("Loading data...")
    
    # Bifiltration features
    bifilt_df = pd.read_csv('computed_data/tda_bifiltration_features.csv')
    print(f"  Bifiltration: {len(bifilt_df)} genes, {len(bifilt_df.columns)} columns")
    
    # Original perturbation features (need to match genes)
    ad_df = pd.read_csv('computed_data/tda_perturbation_alzheimers.csv')
    bg_df = pd.read_csv('computed_data/tda_perturbation_top_candidates.csv')
    
    # Get same degree-matched background
    ad_ids = set(ad_df['node_id'].values)
    bg_df = bg_df[~bg_df['node_id'].isin(ad_ids)]
    
    np.random.seed(42)
    matched_bg = []
    for degree in ad_df['degree'].values:
        tolerance = max(5, degree * 0.2)
        candidates = bg_df[(bg_df['degree'] >= degree - tolerance) & 
                          (bg_df['degree'] <= degree + tolerance)]
        if len(candidates) > 0:
            sampled = candidates.sample(n=1, random_state=len(matched_bg))
            matched_bg.append(sampled)
    
    matched_bg_df = pd.concat(matched_bg, ignore_index=True).drop_duplicates(subset='node_id')
    
    # Combine perturbation data
    ad_df['is_ad'] = 1
    matched_bg_df['is_ad'] = 0
    perturb_df = pd.concat([ad_df, matched_bg_df], ignore_index=True)
    print(f"  Perturbation: {len(perturb_df)} genes")
    
    # Merge on node_id
    merged = bifilt_df.merge(
        perturb_df[['node_id', 'delta_H0', 'delta_H1', 'delta_H2', 'delta_H3',
                    'H0_with', 'H1_with', 'H2_with', 'H3_with',
                    'H0_without', 'H1_without', 'H2_without', 'H3_without']],
        on='node_id',
        how='inner',
        suffixes=('', '_perturb')
    )
    print(f"  Merged: {len(merged)} genes")
    
    return merged


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Clean graph features (no ego_edges or n_simplices - those are biased)
GRAPH_FEATURES = ['degree', 'ego_original_size', 'ego_actual_size']

# Original perturbation TDA (from 07_)
PERTURB_TDA = [
    'delta_H0', 'delta_H1', 'delta_H2', 'delta_H3',
    'H0_with', 'H1_with', 'H2_with', 'H3_with',
    'H0_without', 'H1_without', 'H2_without', 'H3_without',
]

# Bifiltration features (from 17_)
BIFILT_DELTA = [
    'delta_H1_ptm10', 'delta_H2_ptm10',
    'delta_H1_ptm25', 'delta_H2_ptm25',
    'delta_H1_ptm50', 'delta_H2_ptm50',
    'delta_H1_ptm75', 'delta_H2_ptm75',
    'delta_H1_ptm90', 'delta_H2_ptm90',
    'delta_H1_ptm100', 'delta_H2_ptm100',
]

BIFILT_BETTI = [
    'H1_ptm10', 'H2_ptm10',
    'H1_ptm25', 'H2_ptm25',
    'H1_ptm50', 'H2_ptm50',
    'H1_ptm75', 'H2_ptm75',
    'H1_ptm90', 'H2_ptm90',
    'H1_ptm100', 'H2_ptm100',
]

BIFILT_DERIVED = [
    'H1_ptm_slope', 'H2_ptm_slope',
    'H1_ptm_range', 'H2_ptm_range',
    'delta_H1_ptm_slope', 'delta_H2_ptm_slope',
    'delta_H1_ptm_range', 'delta_H2_ptm_range',
]

BIFILT_ALL = BIFILT_DELTA + BIFILT_BETTI + BIFILT_DERIVED

# PTM metadata
PTM_META = ['center_ptm', 'ego_mean_ptm', 'ego_max_ptm', 'ego_ptm_std']


# ============================================================================
# CLASSIFICATION
# ============================================================================

def run_cv(df, feature_cols, n_repeats=10):
    """Run repeated CV and return scores."""
    available = [c for c in feature_cols if c in df.columns]
    if len(available) == 0:
        return np.array([0.5] * 100)
    
    X = df[available].fillna(0).values
    y = df['is_ad'].astype(int).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    
    all_scores = []
    for repeat in range(n_repeats):
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42 + repeat)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')
        all_scores.extend(scores)
    
    return np.array(all_scores)


def compare_to_baseline(baseline_scores, test_scores, name):
    """Compare test scores to baseline with t-test."""
    n = min(len(baseline_scores), len(test_scores))
    t_stat, p_val = stats.ttest_rel(test_scores[:n], baseline_scores[:n])
    improvement = np.mean(test_scores) - np.mean(baseline_scores)
    
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    
    return improvement, p_val, sig


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("TESTING BIFILTRATION FEATURES")
    print("="*70)
    
    df = load_all_features()
    
    # Define feature sets to test
    feature_sets = {
        'Graph only': GRAPH_FEATURES,
        'Graph + Perturb TDA': GRAPH_FEATURES + PERTURB_TDA,
        'Graph + Bifilt deltas': GRAPH_FEATURES + BIFILT_DELTA,
        'Graph + Bifilt all': GRAPH_FEATURES + BIFILT_ALL,
        'Graph + Bifilt + PTM meta': GRAPH_FEATURES + BIFILT_ALL + PTM_META,
        'Graph + Perturb + Bifilt': GRAPH_FEATURES + PERTURB_TDA + BIFILT_ALL,
        'Everything': GRAPH_FEATURES + PERTURB_TDA + BIFILT_ALL + PTM_META,
    }
    
    print(f"\nRunning 10-fold CV × 10 repeats...")
    print("-"*70)
    
    results = {}
    for name, features in feature_sets.items():
        available = [f for f in features if f in df.columns]
        scores = run_cv(df, available)
        results[name] = scores
        print(f"{name:<30}: AUROC = {np.mean(scores):.4f} ± {np.std(scores):.4f}  (n={len(available)} features)")
    
    # Compare to baselines
    print("\n" + "="*70)
    print("COMPARISONS")
    print("="*70)
    
    baseline = results['Graph only']
    print(f"\nCompared to Graph only ({np.mean(baseline):.4f}):")
    print("-"*70)
    
    for name, scores in results.items():
        if name == 'Graph only':
            continue
        imp, p, sig = compare_to_baseline(baseline, scores, name)
        print(f"{name:<30}: {imp:+.4f}  (p={p:.4f}) {sig}")
    
    # Compare bifilt to perturb
    print("\n" + "-"*70)
    perturb_scores = results['Graph + Perturb TDA']
    print(f"\nCompared to Graph + Perturb TDA ({np.mean(perturb_scores):.4f}):")
    print("-"*70)
    
    for name in ['Graph + Bifilt all', 'Graph + Perturb + Bifilt', 'Everything']:
        scores = results[name]
        imp, p, sig = compare_to_baseline(perturb_scores, scores, name)
        print(f"{name:<30}: {imp:+.4f}  (p={p:.4f}) {sig}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    best_name = max(results.keys(), key=lambda x: np.mean(results[x]))
    best_score = np.mean(results[best_name])
    
    graph_score = np.mean(results['Graph only'])
    perturb_score = np.mean(results['Graph + Perturb TDA'])
    bifilt_score = np.mean(results['Graph + Bifilt all'])
    combined_score = np.mean(results['Graph + Perturb + Bifilt'])
    
    print(f"""
Baseline (Graph only):           {graph_score:.4f}
+ Perturbation TDA:              {perturb_score:.4f}  (+{perturb_score - graph_score:.4f})
+ Bifiltration TDA:              {bifilt_score:.4f}  (+{bifilt_score - graph_score:.4f})
+ Both TDA types:                {combined_score:.4f}  (+{combined_score - graph_score:.4f})

Best overall: {best_name} ({best_score:.4f})

Key question: Does bifiltration add value beyond perturbation TDA?
  Perturb alone:  {perturb_score:.4f}
  Combined:       {combined_score:.4f}
  Improvement:    {combined_score - perturb_score:+.4f}
""")
    
    # Check if combined is better than perturb
    imp, p, sig = compare_to_baseline(perturb_scores, results['Graph + Perturb + Bifilt'], 'combined')
    
    if p < 0.05 and imp > 0:
        print(f"✅ YES! Bifiltration adds significant value (p={p:.4f})")
    else:
        print(f"❌ Bifiltration doesn't significantly improve over perturbation alone (p={p:.4f})")


if __name__ == "__main__":
    main()

