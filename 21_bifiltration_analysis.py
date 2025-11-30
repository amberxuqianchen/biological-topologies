#!/usr/bin/env python3
"""
Comprehensive Bifiltration Analysis

1. Feature importance - what's driving the 0.709 AUROC?
2. Nice visualizations
3. Test Amyloid vs Tau with bifiltration
4. Explain how the classifier works
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING
# ============================================================================

def load_bifiltration_data():
    """Load bifiltration features."""
    df = pd.read_csv('computed_data/tda_bifiltration_features.csv')
    return df


def load_ad_categories():
    """Load Amyloid/Tau categories."""
    filepath = 'data/BIOGRID-PROJECT-alzheimers_disease_project-GENES-5.0.250.projectindex.txt'
    genes_df = pd.read_csv(filepath, sep='\t')
    
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
    return categories


# ============================================================================
# FEATURE DEFINITIONS  
# ============================================================================

GRAPH_FEATURES = ['degree', 'ego_original_size', 'ego_actual_size']

BIFILT_DELTA = [
    'delta_H1_ptm10', 'delta_H2_ptm10', 'delta_H3_ptm10',
    'delta_H1_ptm25', 'delta_H2_ptm25', 'delta_H3_ptm25',
    'delta_H1_ptm50', 'delta_H2_ptm50', 'delta_H3_ptm50',
    'delta_H1_ptm75', 'delta_H2_ptm75', 'delta_H3_ptm75',
    'delta_H1_ptm90', 'delta_H2_ptm90', 'delta_H3_ptm90',
    'delta_H1_ptm100', 'delta_H2_ptm100', 'delta_H3_ptm100',
]

BIFILT_BETTI = [f'H{d}_ptm{p}' for d in [1,2,3] for p in [10,25,50,75,90,100]]

BIFILT_DERIVED = [
    'H1_ptm_slope', 'H2_ptm_slope', 'H3_ptm_slope',
    'H1_ptm_range', 'H2_ptm_range', 'H3_ptm_range',
    'delta_H1_ptm_slope', 'delta_H2_ptm_slope', 'delta_H3_ptm_slope',
    'delta_H1_ptm_range', 'delta_H2_ptm_range', 'delta_H3_ptm_range',
]

BIFILT_ALL = BIFILT_DELTA + BIFILT_BETTI + BIFILT_DERIVED


# ============================================================================
# 1. FEATURE IMPORTANCE
# ============================================================================

def analyze_feature_importance(df):
    """Which bifiltration features matter most?"""
    print("\n" + "="*70)
    print("1. FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    features = [f for f in GRAPH_FEATURES + BIFILT_ALL if f in df.columns]
    X = df[features].fillna(0).values
    y = df['is_ad'].astype(int).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_scaled, y)
    
    importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print("-"*50)
    for i, row in importance.head(15).iterrows():
        feat = row['feature']
        imp = row['importance']
        feat_type = "bifilt" if feat in BIFILT_ALL else "graph"
        print(f"  {feat:<25} {imp:.4f}  [{feat_type}]")
    
    return importance


# ============================================================================
# 2. VISUALIZATIONS
# ============================================================================

def create_visualizations(df, importance):
    """Create publication-quality figures."""
    print("\n" + "="*70)
    print("2. CREATING VISUALIZATIONS")
    print("="*70)
    
    os.makedirs('figures', exist_ok=True)
    
    ad = df[df['is_ad'] == True]
    bg = df[df['is_ad'] == False]
    
    # Figure 1: Feature Importance Bar Chart
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = importance.head(15)
    colors = ['coral' if f in BIFILT_ALL else 'steelblue' for f in top_features['feature']]
    
    bars = ax.barh(range(len(top_features)), top_features['importance'].values, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (Random Forest)')
    ax.set_title('Top 15 Features for AD Gene Classification')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='coral', label='Bifiltration TDA'),
                       Patch(facecolor='steelblue', label='Graph')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('figures/bifilt_feature_importance.png', dpi=150)
    plt.close()
    print("  Saved: figures/bifilt_feature_importance.png")
    
    # Figure 2: Delta H2 across PTM thresholds
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    thresholds = [10, 25, 50, 75, 90, 100]
    
    for idx, dim in enumerate(['H1', 'H2']):
        ax = axes[idx]
        
        ad_means = [ad[f'delta_{dim}_ptm{t}'].mean() for t in thresholds]
        bg_means = [bg[f'delta_{dim}_ptm{t}'].mean() for t in thresholds]
        ad_stds = [ad[f'delta_{dim}_ptm{t}'].std() / np.sqrt(len(ad)) for t in thresholds]
        bg_stds = [bg[f'delta_{dim}_ptm{t}'].std() / np.sqrt(len(bg)) for t in thresholds]
        
        ax.errorbar(thresholds, ad_means, yerr=ad_stds, label='AD genes', 
                    marker='o', capsize=3, color='red')
        ax.errorbar(thresholds, bg_means, yerr=bg_stds, label='Background', 
                    marker='s', capsize=3, color='blue')
        
        ax.set_xlabel('PTM Percentile Threshold')
        ax.set_ylabel(f'delta_{dim} (perturbation impact)')
        ax.set_title(f'{dim} Perturbation Across PTM Levels')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/bifilt_delta_across_ptm.png', dpi=150)
    plt.close()
    print("  Saved: figures/bifilt_delta_across_ptm.png")
    
    # Figure 3: Classification performance comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    results = {
        'Graph only': 0.514,
        'Graph + Perturb': 0.590,
        'Graph + Bifilt': 0.709,
    }
    
    colors = ['gray', 'steelblue', 'coral']
    bars = ax.bar(results.keys(), results.values(), color=colors)
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random')
    ax.set_ylabel('AUROC')
    ax.set_title('Classification Performance: AD vs Background')
    ax.set_ylim(0.4, 0.8)
    
    # Add value labels
    for bar, val in zip(bars, results.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('figures/bifilt_performance_comparison.png', dpi=150)
    plt.close()
    print("  Saved: figures/bifilt_performance_comparison.png")


# ============================================================================
# 3. AMYLOID VS TAU WITH BIFILTRATION
# ============================================================================

def test_amyloid_vs_tau(df, categories):
    """Test if bifiltration can distinguish Amyloid vs Tau."""
    print("\n" + "="*70)
    print("3. AMYLOID VS TAU CLASSIFICATION (WITH BIFILTRATION)")
    print("="*70)
    
    # Add categories to AD genes
    ad_df = df[df['is_ad'] == True].copy()
    ad_df['category'] = ad_df['node_id'].map(categories)
    
    # Filter to pure Amyloid or Tau
    filtered = ad_df[ad_df['category'].isin(['Amyloid', 'Tau'])]
    
    print(f"\n  Amyloid genes: {(filtered['category'] == 'Amyloid').sum()}")
    print(f"  Tau genes: {(filtered['category'] == 'Tau').sum()}")
    
    y = (filtered['category'] == 'Amyloid').astype(int).values
    
    # Test different feature sets
    feature_sets = {
        'Graph only': GRAPH_FEATURES,
        'Graph + Bifilt': GRAPH_FEATURES + BIFILT_ALL,
        'Bifilt only': BIFILT_ALL,
    }
    
    print(f"\n  {'Feature Set':<25} {'AUROC':>10}")
    print("  " + "-"*40)
    
    for name, features in feature_sets.items():
        available = [f for f in features if f in filtered.columns]
        X = filtered[available].fillna(0).values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')
        
        print(f"  {name:<25} {np.mean(scores):>10.4f} ± {np.std(scores):.4f}")
    
    print("""
  Interpretation:
  - If AUROC ≈ 0.5 → Amyloid and Tau have similar bifiltration signatures
  - If AUROC > 0.6 → Bifiltration captures pathway-specific topology!
    """)


# ============================================================================
# 4. HOW THE CLASSIFIER WORKS
# ============================================================================

def explain_classifier():
    """Explain how our classifier works."""
    print("\n" + "="*70)
    print("4. HOW THE CLASSIFIER WORKS")
    print("="*70)
    
    print("""
  We use STRATIFIED K-FOLD CROSS-VALIDATION:
  
  ┌─────────────────────────────────────────────────────────────┐
  │  Total: 841 genes (464 AD + 377 Background)                 │
  └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Split into 10 folds (stratified = same AD/BG ratio each)   │
  │                                                             │
  │  Fold 1: [████████████████████████████████████████] ████    │
  │  Fold 2: [████████████████████████████████████████] ████    │
  │  ...                                                        │
  │  Fold 10: ████ [████████████████████████████████████████]   │
  │           ↑                ↑                                │
  │         Test            Train                               │
  └─────────────────────────────────────────────────────────────┘
  
  For each fold:
    - Train on ~757 genes (90%)
    - Test on ~84 genes (10%)
    - Record AUROC
  
  Repeat 10 times with different random splits → 100 total scores
  
  Final AUROC = mean of all 100 scores
  
  This is UNBIASED because:
    - Every gene gets tested exactly once per repeat
    - No data leakage between train and test
    - Cross-validation variance gives us confidence intervals
    """)


# ============================================================================
# 5. SHOULD WE COMPUTE MORE BACKGROUND GENES?
# ============================================================================

def discuss_more_candidates():
    """Discuss whether to compute more background genes."""
    print("\n" + "="*70)
    print("5. SHOULD WE COMPUTE MORE BACKGROUND GENES?")
    print("="*70)
    
    print("""
  Current situation:
    - AD genes: 464 (all computed)
    - Background: 377 (degree-matched subset)
    - Top candidates available: 10,000 (from perturbation TDA)
  
  Pros of computing more bifiltration:
    ✓ Could discover novel AD candidates
    ✓ Better calibrated probability estimates
    ✓ More robust model
  
  Cons:
    ✗ Time: ~10-15 sec/gene × 10,000 = ~28-42 hours
    ✗ Diminishing returns for classification (already have enough for training)
  
  RECOMMENDATION:
    For classification: Current 841 genes is sufficient
    For candidate discovery: Compute top 1000-2000 candidates with highest
                            perturbation TDA scores (most "interesting" genes)
  
  If you want to run overnight:
    python 18_overnight_bifiltration.py --custom  # (need to add this option)
    """)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("COMPREHENSIVE BIFILTRATION ANALYSIS")
    print("="*70)
    
    # Load data
    df = load_bifiltration_data()
    categories = load_ad_categories()
    
    print(f"\nLoaded {len(df)} genes ({(df['is_ad']==True).sum()} AD, {(df['is_ad']==False).sum()} BG)")
    
    # 1. Feature importance
    importance = analyze_feature_importance(df)
    
    # 2. Visualizations
    create_visualizations(df, importance)
    
    # 3. Amyloid vs Tau
    test_amyloid_vs_tau(df, categories)
    
    # 4. Explain classifier
    explain_classifier()
    
    # 5. Discuss more candidates
    discuss_more_candidates()
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print("\nFigures saved to figures/")


if __name__ == "__main__":
    main()

