#!/usr/bin/env python3
"""
Analyze which TDA features drive classification and check for bias propagation.

1. Feature importance - which TDA features matter?
2. Correlation with n_simplices - are other features also biased?
3. Effect of subsampling on bias
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
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
    df = pd.concat([ad_df, matched_bg_df], ignore_index=True)
    
    return df


# Feature definitions
GRAPH_FEATURES = ['degree', 'ego_original_size', 'ego_actual_size', 'ego_edges']

TDA_FEATURES = [
    'H0_with', 'H1_with', 'H2_with', 'H3_with',
    'H0_without', 'H1_without', 'H2_without', 'H3_without',
    'delta_H0', 'delta_H1', 'delta_H2', 'delta_H3',
]

ALL_FEATURES = GRAPH_FEATURES + TDA_FEATURES


def analyze_feature_importance(df):
    """Which features drive the classification?"""
    print("="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    features = [f for f in GRAPH_FEATURES + TDA_FEATURES if f in df.columns]
    X = df[features].fillna(0).values
    y = df['is_ad'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Logistic regression coefficients
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_scaled, y)
    
    lr_importance = pd.DataFrame({
        'feature': features,
        'coefficient': lr.coef_[0],
        'abs_coefficient': np.abs(lr.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\nLogistic Regression Coefficients (top 15):")
    print("-"*50)
    for _, row in lr_importance.head(15).iterrows():
        feat = row['feature']
        coef = row['coefficient']
        is_tda = feat in TDA_FEATURES
        marker = "[TDA]" if is_tda else "[Graph]"
        direction = "→ AD" if coef > 0 else "→ BG"
        print(f"  {feat:<20} {coef:+.4f} {direction:<8} {marker}")
    
    # Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_scaled, y)
    
    rf_importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n\nRandom Forest Feature Importance (top 15):")
    print("-"*50)
    for _, row in rf_importance.head(15).iterrows():
        feat = row['feature']
        imp = row['importance']
        is_tda = feat in TDA_FEATURES
        marker = "[TDA]" if is_tda else "[Graph]"
        print(f"  {feat:<20} {imp:.4f} {marker}")
    
    # Summary
    print("\n\nTOP 5 MOST IMPORTANT FEATURES:")
    print("-"*50)
    top5 = rf_importance.head(5)['feature'].tolist()
    for i, feat in enumerate(top5, 1):
        is_tda = feat in TDA_FEATURES
        print(f"  {i}. {feat} {'(TDA)' if is_tda else '(Graph)'}")
    
    return lr_importance, rf_importance


def check_bias_propagation(df):
    """Check if other TDA features are correlated with n_simplices (potential bias)."""
    print("\n" + "="*70)
    print("BIAS PROPAGATION CHECK")
    print("="*70)
    print("\nAre other TDA features correlated with n_simplices?")
    print("High correlation → might also be affected by study bias\n")
    
    if 'n_simplices' not in df.columns:
        print("n_simplices not in dataframe")
        return
    
    features_to_check = TDA_FEATURES + ['ego_original_size', 'ego_edges']
    
    print(f"{'Feature':<20} {'Corr w/ n_simplices':>20} {'Interpretation':<30}")
    print("-"*70)
    
    for feat in features_to_check:
        if feat in df.columns:
            corr = df[feat].corr(df['n_simplices'])
            
            if abs(corr) > 0.7:
                interp = "⚠️  HIGH - might be biased"
            elif abs(corr) > 0.4:
                interp = "⚡ MODERATE - check carefully"
            else:
                interp = "✓ LOW - likely independent"
            
            print(f"  {feat:<18} {corr:>+18.3f}   {interp}")
    
    # Key insight
    print("\n" + "-"*70)
    print("""
KEY INSIGHT:
- delta_H features (perturbation impact) should be LESS correlated with 
  n_simplices than absolute Betti numbers (H*_with, H*_without)
- If delta features have low correlation → they're measuring something 
  different from neighborhood density
    """)


def check_subsampling_effect(df):
    """Check if subsampling equalizes the neighborhoods."""
    print("\n" + "="*70)
    print("SUBSAMPLING EFFECT ANALYSIS")
    print("="*70)
    
    ad = df[df['is_ad'] == 1]
    bg = df[df['is_ad'] == 0]
    
    print("\nBefore vs After Subsampling:")
    print("-"*70)
    print(f"{'Metric':<25} {'AD Mean':>12} {'BG Mean':>12} {'Ratio':>10}")
    print("-"*70)
    
    # Original size (before subsampling)
    ad_orig = ad['ego_original_size'].mean()
    bg_orig = bg['ego_original_size'].mean()
    print(f"{'ego_original_size':<25} {ad_orig:>12.0f} {bg_orig:>12.0f} {ad_orig/bg_orig:>10.2f}")
    
    # Actual size (after subsampling to max 300)
    ad_actual = ad['ego_actual_size'].mean()
    bg_actual = bg['ego_actual_size'].mean()
    print(f"{'ego_actual_size':<25} {ad_actual:>12.0f} {bg_actual:>12.0f} {ad_actual/bg_actual:>10.2f}")
    
    # Edges
    ad_edges = ad['ego_edges'].mean()
    bg_edges = bg['ego_edges'].mean()
    print(f"{'ego_edges':<25} {ad_edges:>12.0f} {bg_edges:>12.0f} {ad_edges/bg_edges:>10.2f}")
    
    # Was subsampled?
    if 'was_subsampled' in df.columns:
        ad_sub = ad['was_subsampled'].mean() * 100
        bg_sub = bg['was_subsampled'].mean() * 100
        print(f"{'% subsampled':<25} {ad_sub:>11.1f}% {bg_sub:>11.1f}%")
    
    print("\n" + "-"*70)
    print("""
INTERPRETATION:
- If ego_actual_size is similar for AD and BG → subsampling equalizes size
- But ego_edges might still differ → captures density WITHIN the sample
- delta features capture how topology CHANGES, not absolute structure
    """)


def analyze_what_deltas_capture(df):
    """Explain what the delta features are measuring."""
    print("\n" + "="*70)
    print("WHAT DO DELTA FEATURES CAPTURE?")
    print("="*70)
    
    ad = df[df['is_ad'] == 1]
    bg = df[df['is_ad'] == 0]
    
    print("\nDelta = Betti(with node) - Betti(without node)")
    print("  Positive delta → node is PART OF topological features")
    print("  Negative delta → node FILLS IN/DESTROYS features\n")
    
    print(f"{'Feature':<15} {'AD Mean':>12} {'BG Mean':>12} {'Diff':>10} {'Meaning':<30}")
    print("-"*80)
    
    explanations = {
        'delta_H0': "Components created/destroyed",
        'delta_H1': "1D holes (loops) affected", 
        'delta_H2': "2D voids affected",
        'delta_H3': "3D cavities affected"
    }
    
    for feat in ['delta_H0', 'delta_H1', 'delta_H2', 'delta_H3']:
        if feat in df.columns:
            ad_mean = ad[feat].mean()
            bg_mean = bg[feat].mean()
            diff = ad_mean - bg_mean
            meaning = explanations.get(feat, "")
            print(f"  {feat:<13} {ad_mean:>+12.2f} {bg_mean:>+12.2f} {diff:>+10.2f}   {meaning}")
    
    print("\n" + "-"*80)
    print("""
INTERPRETATION:
- All deltas are NEGATIVE → removing a gene creates more features (fills holes)
- If AD has MORE negative delta → AD genes are more "hole-filling"
- If BG has MORE negative delta → BG genes are more "hole-filling"
- The DIFFERENCE tells us about structural role, not study bias!
    """)


def main():
    print("="*70)
    print("DEEP DIVE: WHAT TDA FEATURES DRIVE CLASSIFICATION?")
    print("="*70)
    
    df = load_matched_data()
    print(f"\nDataset: {len(df)} samples")
    
    # 1. Feature importance
    lr_imp, rf_imp = analyze_feature_importance(df)
    
    # 2. Bias propagation
    check_bias_propagation(df)
    
    # 3. Subsampling effect
    check_subsampling_effect(df)
    
    # 4. What deltas measure
    analyze_what_deltas_capture(df)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
1. TOP FEATURES: Check which TDA features are most important
   → If delta features dominate, signal is about PERTURBATION IMPACT
   → If H*_with features dominate, signal might include bias
   
2. BIAS CHECK: Low correlation with n_simplices is good
   → delta features typically less correlated than absolute Bettis
   
3. SUBSAMPLING: Equalizes neighborhood SIZE but not STRUCTURE
   → BFS subsampling preserves local topology near center
   → delta captures how center node affects this local structure
   
4. DELTA MEANING: Measures "topological importance" of each gene
   → Independent of absolute neighborhood size
   → Captures functional role in network topology
    """)


if __name__ == "__main__":
    main()
