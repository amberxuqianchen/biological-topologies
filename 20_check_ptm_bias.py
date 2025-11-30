#!/usr/bin/env python3
"""
Check if PTM meta features are biased (like n_simplices was).
Also clarify what the PTM features actually are.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def main():
    print("="*70)
    print("CHECKING PTM FEATURE BIAS")
    print("="*70)
    
    # Load bifiltration data
    df = pd.read_csv('computed_data/tda_bifiltration_features.csv')
    
    ad = df[df['is_ad'] == True]
    bg = df[df['is_ad'] == False]
    
    print(f"\nDataset: {len(ad)} AD genes, {len(bg)} background genes")
    
    # ================================================================
    # WHAT ARE THE PTM FEATURES?
    # ================================================================
    print("\n" + "="*70)
    print("WHAT ARE THE PTM META FEATURES?")
    print("="*70)
    
    print("""
Feature          | What it is                           | How calculated
-----------------|--------------------------------------|---------------------------
center_ptm       | PTM COUNT of the center gene         | # of PTM records in BioGRID
ego_mean_ptm     | Mean PTM count in neighborhood       | mean(PTM counts of neighbors)
ego_max_ptm      | Max PTM count in neighborhood        | max(PTM counts of neighbors)
ego_ptm_std      | Std of PTM counts in neighborhood    | std(PTM counts of neighbors)

NOTE: These are RAW COUNTS, not densities or 0-1 normalized values!
      center_ptm could be 0, 5, 50, 500, etc.
    """)
    
    # ================================================================
    # COMPARE AD vs BACKGROUND
    # ================================================================
    print("\n" + "="*70)
    print("AD vs BACKGROUND: PTM META FEATURES")
    print("="*70)
    
    features_to_check = ['center_ptm', 'ego_mean_ptm', 'ego_max_ptm', 'ego_ptm_std']
    
    print(f"\n{'Feature':<20} {'AD Mean':>12} {'BG Mean':>12} {'Ratio':>10} {'p-value':>12}")
    print("-"*70)
    
    for feat in features_to_check:
        if feat not in df.columns:
            continue
        ad_mean = ad[feat].mean()
        bg_mean = bg[feat].mean()
        ratio = ad_mean / bg_mean if bg_mean > 0 else float('inf')
        
        t_stat, p_val = stats.ttest_ind(ad[feat], bg[feat])
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        
        print(f"{feat:<20} {ad_mean:>12.2f} {bg_mean:>12.2f} {ratio:>10.2f}x {p_val:>11.2e} {sig}")
    
    # ================================================================
    # DISTRIBUTION PLOTS
    # ================================================================
    print("\n" + "="*70)
    print("DISTRIBUTION ANALYSIS")
    print("="*70)
    
    for feat in ['center_ptm', 'ego_mean_ptm']:
        ad_vals = ad[feat]
        bg_vals = bg[feat]
        
        print(f"\n{feat}:")
        print(f"  AD:  min={ad_vals.min():.0f}, median={ad_vals.median():.0f}, "
              f"max={ad_vals.max():.0f}, mean={ad_vals.mean():.1f}")
        print(f"  BG:  min={bg_vals.min():.0f}, median={bg_vals.median():.0f}, "
              f"max={bg_vals.max():.0f}, mean={bg_vals.mean():.1f}")
    
    # ================================================================
    # VERDICT
    # ================================================================
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    ad_center_ptm = ad['center_ptm'].mean()
    bg_center_ptm = bg['center_ptm'].mean()
    ratio = ad_center_ptm / bg_center_ptm if bg_center_ptm > 0 else float('inf')
    
    if ratio > 1.5:
        print(f"""
⚠️  BIAS CONFIRMED!

AD genes have {ratio:.1f}x more PTM records than background genes.

This is almost certainly because:
- AD genes are well-studied → more experiments → more PTM records
- BioGRID curates what's published → publication bias

The 0.926 AUROC is mostly detecting "how well-studied is this gene"
rather than real biological differences.

RECOMMENDATION: Exclude PTM meta features from your final model.
Use only the bifiltration TDA features (0.709 AUROC) which are robust.
        """)
    else:
        print(f"PTM counts are similar between AD and background (ratio={ratio:.2f})")
    
    # ================================================================
    # WHAT ABOUT BIFILTRATION FEATURES?
    # ================================================================
    print("\n" + "="*70)
    print("ARE BIFILTRATION TDA FEATURES BIASED?")
    print("="*70)
    
    # Check correlation of bifilt features with center_ptm
    bifilt_features = [c for c in df.columns if 'delta_H' in c and 'ptm' in c][:6]
    
    print(f"\nCorrelation of bifiltration features with center_ptm:")
    print("-"*50)
    
    for feat in bifilt_features:
        corr = df[feat].corr(df['center_ptm'])
        print(f"  {feat:<25}: {corr:+.3f}")
    
    print("""
If correlations are LOW (< 0.3) → bifiltration features are independent of study bias
If correlations are HIGH (> 0.5) → might be affected by bias
    """)


if __name__ == "__main__":
    main()

