#!/usr/bin/env python3
"""
Test classification WITHOUT n_simplices

See if TDA features still help when we exclude the potentially biased feature.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Feature groups
GRAPH_FEATURES = ['degree', 'ego_original_size', 'ego_actual_size', 'ego_edges']

TDA_FEATURES_ALL = [
    'H0_with', 'H1_with', 'H2_with', 'H3_with',
    'H0_without', 'H1_without', 'H2_without', 'H3_without',
    'delta_H0', 'delta_H1', 'delta_H2', 'delta_H3',
    'n_simplices'
]

TDA_FEATURES_NO_SIMPLICES = [
    'H0_with', 'H1_with', 'H2_with', 'H3_with',
    'H0_without', 'H1_without', 'H2_without', 'H3_without',
    'delta_H0', 'delta_H1', 'delta_H2', 'delta_H3',
]

def load_matched_data():
    """Load AD vs degree-matched background."""
    ad_df = pd.read_csv('computed_data/tda_perturbation_alzheimers.csv')
    bg_df = pd.read_csv('computed_data/tda_perturbation_top_candidates.csv')
    
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
    
    matched_bg_df = pd.concat(matched_bg, ignore_index=True)
    matched_bg_df = matched_bg_df.drop_duplicates(subset='node_id')
    
    ad_df['is_ad'] = 1
    matched_bg_df['is_ad'] = 0
    df = pd.concat([ad_df, matched_bg_df], ignore_index=True)
    
    return df


def run_cv(df, feature_cols, n_repeats=10):
    """Run repeated CV and return scores."""
    X = df[[c for c in feature_cols if c in df.columns]].fillna(0).values
    y = df['is_ad'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    
    all_scores = []
    for repeat in range(n_repeats):
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42 + repeat)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')
        all_scores.extend(scores)
    
    return np.array(all_scores)


def main():
    print("="*70)
    print("TESTING CLASSIFICATION WITHOUT N_SIMPLICES")
    print("="*70)
    
    df = load_matched_data()
    print(f"\nDataset: {len(df)} samples ({(df['is_ad']==1).sum()} AD, {(df['is_ad']==0).sum()} BG)")
    
    # Define feature sets to test
    feature_sets = {
        'Graph only': GRAPH_FEATURES,
        'Graph + n_simplices only': GRAPH_FEATURES + ['n_simplices'],
        'Graph + TDA (no simplices)': GRAPH_FEATURES + TDA_FEATURES_NO_SIMPLICES,
        'Graph + ALL TDA': GRAPH_FEATURES + TDA_FEATURES_ALL,
        'TDA only (no simplices)': TDA_FEATURES_NO_SIMPLICES,
        'n_simplices only': ['n_simplices'],
    }
    
    print(f"\nRunning 10-fold CV × 10 repeats for each feature set...")
    print("-"*70)
    
    results = {}
    for name, features in feature_sets.items():
        available = [f for f in features if f in df.columns]
        scores = run_cv(df, available, n_repeats=10)
        results[name] = scores
        print(f"{name:<30}: AUROC = {np.mean(scores):.4f} ± {np.std(scores):.4f}  (n_features={len(available)})")
    
    # Statistical comparisons
    print("\n" + "="*70)
    print("STATISTICAL COMPARISONS")
    print("="*70)
    
    baseline = results['Graph only']
    
    print(f"\nCompared to Graph only ({np.mean(baseline):.4f}):")
    print("-"*70)
    
    for name, scores in results.items():
        if name == 'Graph only':
            continue
        
        # Paired t-test
        n = min(len(baseline), len(scores))
        t_stat, p_val = stats.ttest_rel(scores[:n], baseline[:n])
        improvement = np.mean(scores) - np.mean(baseline)
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        
        print(f"{name:<30}: {improvement:+.4f}  (p={p_val:.4f}) {sig}")
    
    # Key comparison: with vs without n_simplices
    print("\n" + "="*70)
    print("KEY QUESTION: Does TDA help WITHOUT n_simplices?")
    print("="*70)
    
    with_simp = results['Graph + ALL TDA']
    without_simp = results['Graph + TDA (no simplices)']
    graph_only = results['Graph only']
    
    imp_with = np.mean(with_simp) - np.mean(graph_only)
    imp_without = np.mean(without_simp) - np.mean(graph_only)
    
    print(f"\nGraph only baseline:            {np.mean(graph_only):.4f}")
    print(f"Graph + ALL TDA:                {np.mean(with_simp):.4f}  (+{imp_with:.4f})")
    print(f"Graph + TDA (no n_simplices):   {np.mean(without_simp):.4f}  (+{imp_without:.4f})")
    
    # t-test for without simplices vs graph
    n = min(len(without_simp), len(graph_only))
    t_stat, p_val = stats.ttest_rel(without_simp[:n], graph_only[:n])
    
    print(f"\nIs +{imp_without:.4f} significant? p = {p_val:.6f}")
    
    if p_val < 0.05 and imp_without > 0:
        print("\n✓ YES! TDA features help even WITHOUT n_simplices!")
        print("  The improvement is not just from study bias.")
    else:
        print("\n✗ NO - Without n_simplices, TDA doesn't significantly help.")
        print("  Most of the signal was from n_simplices (likely study bias).")
    
    # How much of the improvement came from n_simplices?
    print("\n" + "="*70)
    print("DECOMPOSING THE IMPROVEMENT")
    print("="*70)
    
    simp_only = results['n_simplices only']
    simp_contribution = np.mean(simp_only) - np.mean(graph_only)
    
    print(f"\nn_simplices alone:     +{simp_contribution:.4f} over graph baseline")
    print(f"Other TDA features:    +{imp_without:.4f} over graph baseline")
    print(f"All TDA together:      +{imp_with:.4f} over graph baseline")
    
    pct_from_simplices = (simp_contribution / imp_with * 100) if imp_with > 0 else 0
    print(f"\n→ {pct_from_simplices:.0f}% of TDA improvement comes from n_simplices")


if __name__ == "__main__":
    main()

