#!/usr/bin/env python3
"""
Statistical Significance Testing for TDA Improvement

Tests whether the improvement from adding TDA features is
statistically significant using:
1. Paired t-test on CV folds
2. Permutation testing
3. Bootstrap confidence intervals
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
# DATA PREPARATION (same as exploration script)
# ============================================================================

def load_data():
    """Load and prepare AD vs Matched Background data."""
    ad_df = pd.read_csv('computed_data/tda_perturbation_alzheimers.csv')
    bg_df = pd.read_csv('computed_data/tda_perturbation_top_candidates.csv')
    
    # Remove AD genes from background
    ad_ids = set(ad_df['node_id'].values)
    bg_df = bg_df[~bg_df['node_id'].isin(ad_ids)]
    
    # Degree matching
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
    
    # Combine
    ad_df['is_ad'] = 1
    matched_bg_df['is_ad'] = 0
    combined = pd.concat([ad_df, matched_bg_df], ignore_index=True)
    
    return combined


GRAPH_FEATURES = ['degree', 'ego_original_size', 'ego_actual_size', 'ego_edges']
TDA_FEATURES = ['H0_with', 'H1_with', 'H2_with', 'H3_with',
                'H0_without', 'H1_without', 'H2_without', 'H3_without',
                'delta_H0', 'delta_H1', 'delta_H2', 'delta_H3', 'n_simplices']
ALL_FEATURES = GRAPH_FEATURES + TDA_FEATURES


def get_fold_scores(df, feature_cols, n_splits=10, n_repeats=10):
    """Get AUROC scores for each CV fold, repeated multiple times."""
    X = df[feature_cols].fillna(0).values
    y = df['is_ad'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    
    all_scores = []
    for repeat in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')
        all_scores.extend(scores)
    
    return np.array(all_scores)


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def paired_t_test(scores1, scores2, name1, name2):
    """Paired t-test comparing two sets of scores."""
    # Use minimum length if different
    n = min(len(scores1), len(scores2))
    s1, s2 = scores1[:n], scores2[:n]
    
    t_stat, p_value = stats.ttest_rel(s2, s1)  # s2 - s1 (improvement)
    
    diff = s2 - s1
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    
    # 95% CI for the difference
    ci_low = mean_diff - 1.96 * se_diff
    ci_high = mean_diff + 1.96 * se_diff
    
    print(f"\nPaired t-test: {name2} vs {name1}")
    print(f"  Mean improvement: {mean_diff:.4f}")
    print(f"  Std of differences: {std_diff:.4f}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    
    if p_value < 0.001:
        print(f"  → HIGHLY SIGNIFICANT (p < 0.001) ✓✓✓")
    elif p_value < 0.01:
        print(f"  → SIGNIFICANT (p < 0.01) ✓✓")
    elif p_value < 0.05:
        print(f"  → SIGNIFICANT (p < 0.05) ✓")
    else:
        print(f"  → NOT SIGNIFICANT (p >= 0.05)")
    
    return p_value, mean_diff, (ci_low, ci_high)


def permutation_test(df, graph_cols, all_cols, n_permutations=1000):
    """Permutation test for TDA improvement."""
    print(f"\nPermutation test ({n_permutations} permutations)...")
    
    X_graph = df[graph_cols].fillna(0).values
    X_all = df[all_cols].fillna(0).values
    y = df['is_ad'].values
    
    scaler_g = StandardScaler()
    scaler_a = StandardScaler()
    X_graph_scaled = scaler_g.fit_transform(X_graph)
    X_all_scaled = scaler_a.fit_transform(X_all)
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Observed difference
    scores_graph = cross_val_score(clf, X_graph_scaled, y, cv=cv, scoring='roc_auc')
    scores_all = cross_val_score(clf, X_all_scaled, y, cv=cv, scoring='roc_auc')
    observed_diff = np.mean(scores_all) - np.mean(scores_graph)
    
    # Permutation distribution
    perm_diffs = []
    for i in range(n_permutations):
        # Shuffle the TDA columns (break the relationship with labels)
        tda_cols = [c for c in all_cols if c not in graph_cols]
        X_perm = X_all_scaled.copy()
        
        # Shuffle TDA features independently
        np.random.seed(i)
        for j in range(len(graph_cols), X_perm.shape[1]):
            np.random.shuffle(X_perm[:, j])
        
        scores_perm = cross_val_score(clf, X_perm, y, cv=cv, scoring='roc_auc')
        perm_diff = np.mean(scores_perm) - np.mean(scores_graph)
        perm_diffs.append(perm_diff)
        
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_permutations} permutations...")
    
    perm_diffs = np.array(perm_diffs)
    
    # P-value: proportion of permutations >= observed
    p_value = np.mean(perm_diffs >= observed_diff)
    
    print(f"\n  Observed improvement: {observed_diff:.4f}")
    print(f"  Mean permuted improvement: {np.mean(perm_diffs):.4f}")
    print(f"  Std permuted improvement: {np.std(perm_diffs):.4f}")
    print(f"  Permutation p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  → SIGNIFICANT by permutation test ✓")
    else:
        print(f"  → Not significant by permutation test")
    
    return p_value, observed_diff, perm_diffs


def bootstrap_ci(df, graph_cols, all_cols, n_bootstrap=1000):
    """Bootstrap confidence interval for improvement."""
    print(f"\nBootstrap confidence interval ({n_bootstrap} samples)...")
    
    improvements = []
    n = len(df)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        df_boot = df.iloc[idx]
        
        X_graph = df_boot[graph_cols].fillna(0).values
        X_all = df_boot[all_cols].fillna(0).values
        y = df_boot['is_ad'].values
        
        scaler_g = StandardScaler()
        scaler_a = StandardScaler()
        X_graph_scaled = scaler_g.fit_transform(X_graph)
        X_all_scaled = scaler_a.fit_transform(X_all)
        
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        try:
            scores_graph = cross_val_score(clf, X_graph_scaled, y, cv=cv, scoring='roc_auc')
            scores_all = cross_val_score(clf, X_all_scaled, y, cv=cv, scoring='roc_auc')
            improvement = np.mean(scores_all) - np.mean(scores_graph)
            improvements.append(improvement)
        except:
            continue
        
        if (i + 1) % 200 == 0:
            print(f"  Completed {i + 1}/{n_bootstrap} bootstrap samples...")
    
    improvements = np.array(improvements)
    
    ci_low = np.percentile(improvements, 2.5)
    ci_high = np.percentile(improvements, 97.5)
    mean_imp = np.mean(improvements)
    
    print(f"\n  Mean improvement: {mean_imp:.4f}")
    print(f"  Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    
    if ci_low > 0:
        print(f"  → CI does not include 0 - SIGNIFICANT ✓")
    else:
        print(f"  → CI includes 0 - not significant")
    
    return mean_imp, (ci_low, ci_high), improvements


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("Scenario: AD vs Degree-Matched Background")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"  AD genes: {(df['is_ad'] == 1).sum()}")
    print(f"  Background genes: {(df['is_ad'] == 0).sum()}")
    
    # Get available features
    graph_cols = [c for c in GRAPH_FEATURES if c in df.columns]
    all_cols = [c for c in ALL_FEATURES if c in df.columns]
    
    print(f"\n  Graph features: {len(graph_cols)}")
    print(f"  All features: {len(all_cols)}")
    
    # Test 1: Repeated CV with paired t-test
    print("\n" + "="*60)
    print("TEST 1: Repeated Cross-Validation (10 repeats × 10 folds)")
    print("="*60)
    
    graph_scores = get_fold_scores(df, graph_cols, n_splits=10, n_repeats=10)
    all_scores = get_fold_scores(df, all_cols, n_splits=10, n_repeats=10)
    
    print(f"\nGraph-only scores: {np.mean(graph_scores):.4f} ± {np.std(graph_scores):.4f}")
    print(f"All features scores: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
    
    p_ttest, mean_diff, ci = paired_t_test(graph_scores, all_scores, "Graph", "Graph+TDA")
    
    # Test 2: Permutation test
    print("\n" + "="*60)
    print("TEST 2: Permutation Test")
    print("="*60)
    
    p_perm, obs_diff, perm_diffs = permutation_test(df, graph_cols, all_cols, n_permutations=500)
    
    # Test 3: Bootstrap CI
    print("\n" + "="*60)
    print("TEST 3: Bootstrap Confidence Interval")
    print("="*60)
    
    mean_boot, ci_boot, boot_imps = bootstrap_ci(df, graph_cols, all_cols, n_bootstrap=500)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"""
TDA Improvement: +{mean_diff:.4f} ({mean_diff*100:.1f}%)

Statistical Tests:
  1. Paired t-test p-value:     {p_ttest:.6f}  {'✓' if p_ttest < 0.05 else '✗'}
  2. Permutation test p-value:  {p_perm:.4f}  {'✓' if p_perm < 0.05 else '✗'}
  3. Bootstrap 95% CI:          [{ci_boot[0]:.4f}, {ci_boot[1]:.4f}]  {'✓' if ci_boot[0] > 0 else '✗'}

VERDICT: {"The improvement is STATISTICALLY SIGNIFICANT! ✓✓✓" if (p_ttest < 0.05 and ci_boot[0] > 0) else "The improvement may not be statistically significant."}

Interpretation:
  - The {mean_diff*100:.1f}% improvement is {'not just noise - TDA genuinely helps!' if p_ttest < 0.05 else 'within the margin of error.'}
  - Even controlling for degree, TDA features capture {'meaningful' if p_ttest < 0.05 else 'limited'} additional signal.
""")


if __name__ == "__main__":
    main()

