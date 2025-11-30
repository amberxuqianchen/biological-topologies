#!/usr/bin/env python3
"""
Final sanity check: Does TDA still help without ego_edges?

ego_edges has 0.548 correlation with n_simplices, so let's exclude it.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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
    return pd.concat([ad_df, matched_bg_df], ignore_index=True)


def run_cv(df, feature_cols, n_repeats=10):
    """Run repeated CV."""
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
    print("FINAL CHECK: EXCLUDING ego_edges")
    print("="*70)
    
    df = load_matched_data()
    
    # Feature sets
    GRAPH_CLEAN = ['degree', 'ego_original_size', 'ego_actual_size']  # NO ego_edges
    TDA_CLEAN = [
        'H0_with', 'H1_with', 'H2_with', 'H3_with',
        'H0_without', 'H1_without', 'H2_without', 'H3_without',
        'delta_H0', 'delta_H1', 'delta_H2', 'delta_H3',
    ]  # NO n_simplices
    
    print(f"\nExcluding: ego_edges (corr=0.548 with n_simplices) and n_simplices")
    print(f"Graph features: {GRAPH_CLEAN}")
    print(f"TDA features: {len(TDA_CLEAN)} features\n")
    
    # Run tests
    results = {}
    feature_sets = {
        'Graph (clean)': GRAPH_CLEAN,
        'Graph + TDA (clean)': GRAPH_CLEAN + TDA_CLEAN,
        'TDA only (clean)': TDA_CLEAN,
    }
    
    for name, features in feature_sets.items():
        scores = run_cv(df, features)
        results[name] = scores
        print(f"{name:<25}: AUROC = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # Statistical test
    baseline = results['Graph (clean)']
    combined = results['Graph + TDA (clean)']
    
    n = min(len(baseline), len(combined))
    t_stat, p_val = stats.ttest_rel(combined[:n], baseline[:n])
    improvement = np.mean(combined) - np.mean(baseline)
    
    print(f"\n{'='*70}")
    print("RESULT")
    print("="*70)
    print(f"\nImprovement from TDA: +{improvement:.4f} ({improvement*100:.1f}%)")
    print(f"p-value: {p_val:.2e}")
    
    if p_val < 0.001:
        print(f"\n✅ CONFIRMED: TDA helps even without ego_edges!")
        print(f"   The {improvement*100:.1f}% improvement is robust and not from biased features.")
    elif p_val < 0.05:
        print(f"\n✅ TDA still helps (p < 0.05)")
    else:
        print(f"\n⚠️ Effect weakened without ego_edges")


if __name__ == "__main__":
    main()

