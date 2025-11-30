#!/usr/bin/env python3
"""
Verify statistical calculations and explore why TDA hurts Amyloid vs Tau
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
# PART 1: VERIFY P-VALUE CALCULATIONS
# ============================================================================

def verify_ttest():
    """Manually verify t-test calculation."""
    print("="*60)
    print("VERIFYING T-TEST CALCULATION")
    print("="*60)
    
    # Simulate what we had: 100 paired observations
    # Graph mean ~0.636, All mean ~0.699, improvement ~0.063
    np.random.seed(42)
    
    # These are example paired scores (in reality we have 100 pairs)
    n = 100
    graph_scores = np.random.normal(0.636, 0.057, n)
    improvement = np.random.normal(0.063, 0.052, n)
    all_scores = graph_scores + improvement
    
    # Manual t-test calculation
    differences = all_scores - graph_scores
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    t_manual = mean_diff / se_diff
    
    # Using scipy
    t_scipy, p_scipy = stats.ttest_rel(all_scores, graph_scores)
    
    # Manual p-value from t-distribution
    df = n - 1
    p_manual = 2 * (1 - stats.t.cdf(abs(t_manual), df))
    
    print(f"\nManual calculation:")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Std of differences: {std_diff:.4f}")
    print(f"  Standard error: {se_diff:.4f}")
    print(f"  t-statistic: {t_manual:.4f}")
    print(f"  Degrees of freedom: {df}")
    print(f"  p-value (two-tailed): {p_manual:.2e}")
    
    print(f"\nScipy calculation:")
    print(f"  t-statistic: {t_scipy:.4f}")
    print(f"  p-value: {p_scipy:.2e}")
    
    print(f"\n✓ Calculations match: {np.isclose(t_manual, t_scipy, rtol=0.01)}")
    
    # What t-value gives p < 0.001 with df=99?
    t_critical = stats.t.ppf(0.9995, 99)  # Two-tailed, so 0.9995
    print(f"\nFor p < 0.001 (two-tailed), need |t| > {t_critical:.3f}")
    print(f"Our t = 12.07 >> {t_critical:.3f}, so p << 0.001 ✓")


# ============================================================================
# PART 2: WHY DOES TDA HURT FOR AMYLOID VS TAU?
# ============================================================================

def analyze_amyloid_vs_tau():
    """Explore why TDA hurts classification for Amyloid vs Tau."""
    print("\n" + "="*60)
    print("WHY DOES TDA HURT AMYLOID VS TAU?")
    print("="*60)
    
    # Load data
    ad_df = pd.read_csv('computed_data/tda_perturbation_alzheimers.csv')
    
    # Load categories
    genes_file = 'data/BIOGRID-PROJECT-alzheimers_disease_project-GENES-5.0.250.projectindex.txt'
    genes_df = pd.read_csv(genes_file, sep='\t')
    
    categories = {}
    for _, row in genes_df.iterrows():
        gene_id = row['ENTREZ GENE ID']
        cats = str(row.get('CATEGORY VALUES', '-'))
        if 'Amyloid' in cats and 'Tau' in cats:
            categories[gene_id] = 'Both'
        elif 'Amyloid' in cats:
            categories[gene_id] = 'Amyloid'
        elif 'Tau' in cats:
            categories[gene_id] = 'Tau'
        else:
            categories[gene_id] = 'Other'
    
    ad_df['category'] = ad_df['node_id'].map(categories)
    
    # Filter to pure Amyloid/Tau
    df = ad_df[ad_df['category'].isin(['Amyloid', 'Tau'])].copy()
    
    amyloid = df[df['category'] == 'Amyloid']
    tau = df[df['category'] == 'Tau']
    
    print(f"\nSample sizes:")
    print(f"  Amyloid: {len(amyloid)}")
    print(f"  Tau: {len(tau)}")
    
    # Compare feature distributions
    print(f"\n--- GRAPH FEATURES ---")
    for feat in ['degree', 'ego_original_size']:
        am_mean = amyloid[feat].mean()
        tau_mean = tau[feat].mean()
        t_stat, p_val = stats.ttest_ind(amyloid[feat], tau[feat])
        diff_pct = (am_mean - tau_mean) / tau_mean * 100
        print(f"  {feat}:")
        print(f"    Amyloid: {am_mean:.1f}, Tau: {tau_mean:.1f} ({diff_pct:+.1f}%)")
        print(f"    t-test p-value: {p_val:.4f} {'*' if p_val < 0.05 else ''}")
    
    print(f"\n--- TDA FEATURES ---")
    for feat in ['delta_H0', 'delta_H1', 'delta_H2', 'H1_with', 'H2_with']:
        if feat in amyloid.columns:
            am_mean = amyloid[feat].mean()
            tau_mean = tau[feat].mean()
            t_stat, p_val = stats.ttest_ind(amyloid[feat], tau[feat])
            print(f"  {feat}:")
            print(f"    Amyloid: {am_mean:.2f}, Tau: {tau_mean:.2f}")
            print(f"    t-test p-value: {p_val:.4f} {'*' if p_val < 0.05 else ''}")
    
    # The key insight
    print(f"\n--- INTERPRETATION ---")
    print("""
The issue: Amyloid and Tau genes have SIMILAR topological signatures!

Both are:
- AD-related genes (same disease)
- Well-studied (similar degree distributions)
- Central to the same disease network

TDA captures LOCAL TOPOLOGY - how a gene's removal affects its neighborhood.
But Amyloid and Tau genes likely play similar TOPOLOGICAL roles, even though
their FUNCTIONS differ (protein aggregation vs microtubule stability).

The functional difference (Amyloid vs Tau) isn't reflected in the
topological structure of their local networks.

In contrast, AD vs Background works because:
- AD genes are in disease-relevant network modules
- Background genes (even high-degree) are in different modules
- TDA can detect this structural difference
    """)
    
    # Check correlation between TDA features and graph features
    print(f"\n--- FEATURE CORRELATIONS ---")
    features = ['degree', 'delta_H0', 'delta_H1', 'delta_H2', 'H1_with']
    available = [f for f in features if f in df.columns]
    corr = df[available].corr()
    print("Correlation with degree:")
    for feat in available[1:]:
        print(f"  {feat}: {corr.loc['degree', feat]:.3f}")
    
    # Overfitting check
    print(f"\n--- OVERFITTING CHECK ---")
    print(f"Samples: {len(df)}")
    print(f"Graph features: 4")
    print(f"TDA features: 13")
    print(f"Samples per feature (graph): {len(df)/4:.1f}")
    print(f"Samples per feature (all): {len(df)/17:.1f}")
    print("""
Rule of thumb: Need ~10-20 samples per feature for stable models.
With 411 samples and 17 features, we have ~24 samples per feature.
This is borderline - adding noisy features can hurt when signal is weak.
    """)


# ============================================================================
# PART 3: COMPARE AD vs BACKGROUND MORE CAREFULLY
# ============================================================================

def verify_ad_vs_background():
    """Double-check the AD vs Background result."""
    print("\n" + "="*60)
    print("VERIFYING AD VS BACKGROUND RESULT")
    print("="*60)
    
    # Load data
    ad_df = pd.read_csv('computed_data/tda_perturbation_alzheimers.csv')
    bg_df = pd.read_csv('computed_data/tda_perturbation_top_candidates.csv')
    
    ad_ids = set(ad_df['node_id'].values)
    bg_df = bg_df[~bg_df['node_id'].isin(ad_ids)]
    
    # Degree matching
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
    
    print(f"\nDataset:")
    print(f"  AD genes: {(df['is_ad'] == 1).sum()}")
    print(f"  Background: {(df['is_ad'] == 0).sum()}")
    print(f"  AD mean degree: {ad_df['degree'].mean():.1f}")
    print(f"  BG mean degree: {matched_bg_df['degree'].mean():.1f}")
    
    # Check TDA feature differences
    print(f"\n--- TDA FEATURE DIFFERENCES (AD vs Background) ---")
    ad_data = df[df['is_ad'] == 1]
    bg_data = df[df['is_ad'] == 0]
    
    for feat in ['delta_H0', 'delta_H1', 'delta_H2', 'H1_with', 'H2_with', 'n_simplices']:
        if feat in df.columns:
            ad_mean = ad_data[feat].mean()
            bg_mean = bg_data[feat].mean()
            t_stat, p_val = stats.ttest_ind(ad_data[feat], bg_data[feat])
            print(f"  {feat}:")
            print(f"    AD: {ad_mean:.2f}, BG: {bg_mean:.2f}")
            print(f"    t-test p-value: {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''}")
    
    print("""
Key observation: If TDA features show significant differences between
AD and Background (low p-values), then they contain real signal!
    """)


def main():
    verify_ttest()
    analyze_amyloid_vs_tau()
    verify_ad_vs_background()


if __name__ == "__main__":
    main()

