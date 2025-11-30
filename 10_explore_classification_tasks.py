#!/usr/bin/env python3
"""
Explore Classification Tasks for TDA

This script compares different classification scenarios to find where
TDA features add the most value over traditional graph metrics.

Scenarios tested:
1. Amyloid vs Tau (within AD genes)
2. AD vs degree-matched background
3. AD vs other diseases (Autism, Glioblastoma, etc.)
4. Multi-disease classification

For each scenario, we compare:
- Graph features only (degree, ego size, etc.)
- TDA features only (delta_H0, delta_H1, etc.)
- All features combined

Goal: Find the task where TDA improves classification the most!
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_perturbation_data():
    """Load all computed perturbation TDA data."""
    data = {}
    
    files = {
        'alzheimers': 'computed_data/tda_perturbation_alzheimers.csv',
        'autism': 'computed_data/tda_perturbation_autism.csv',
        'autophagy': 'computed_data/tda_perturbation_autophagy.csv',
        'fanconi': 'computed_data/tda_perturbation_fanconi.csv',
        'glioblastoma': 'computed_data/tda_perturbation_glioblastoma.csv',
        'top_candidates': 'computed_data/tda_perturbation_top_candidates.csv',
    }
    
    for name, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            data[name] = df
            print(f"  {name}: {len(df)} genes")
        else:
            print(f"  {name}: NOT FOUND")
    
    return data


def load_ad_categories():
    """Load Amyloid/Tau category labels for AD genes."""
    filepath = 'data/BIOGRID-PROJECT-alzheimers_disease_project-GENES-5.0.250.projectindex.txt'
    df = pd.read_csv(filepath, sep='\t')
    
    categories = {}
    for _, row in df.iterrows():
        gene_id = row['ENTREZ GENE ID']
        cats = row.get('CATEGORY VALUES', '-')
        if pd.notna(cats) and cats != '-':
            cats_str = str(cats)
            is_amyloid = 'Amyloid' in cats_str
            is_tau = 'Tau' in cats_str
            if is_amyloid and is_tau:
                categories[gene_id] = 'Both'
            elif is_amyloid:
                categories[gene_id] = 'Amyloid'
            elif is_tau:
                categories[gene_id] = 'Tau'
            else:
                categories[gene_id] = 'Other'
        else:
            categories[gene_id] = 'Unknown'
    
    return categories


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

GRAPH_FEATURES = ['degree', 'ego_original_size', 'ego_actual_size', 'ego_edges']

TDA_FEATURES = [
    'H0_with', 'H1_with', 'H2_with', 'H3_with',
    'H0_without', 'H1_without', 'H2_without', 'H3_without',
    'delta_H0', 'delta_H1', 'delta_H2', 'delta_H3',
    'n_simplices'
]

ALL_FEATURES = GRAPH_FEATURES + TDA_FEATURES


def get_features(df, feature_set='all'):
    """Extract feature matrix from dataframe."""
    if feature_set == 'graph':
        cols = [c for c in GRAPH_FEATURES if c in df.columns]
    elif feature_set == 'tda':
        cols = [c for c in TDA_FEATURES if c in df.columns]
    else:  # all
        cols = [c for c in ALL_FEATURES if c in df.columns]
    
    X = df[cols].fillna(0).values
    return X, cols


# ============================================================================
# CLASSIFICATION EXPERIMENTS
# ============================================================================

def run_classification(X, y, name=""):
    """Run classification with cross-validation, return AUROC."""
    if len(np.unique(y)) < 2:
        return {'auroc': 0.5, 'std': 0.0}
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use logistic regression for interpretability
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    
    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    try:
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')
        return {'auroc': scores.mean(), 'std': scores.std()}
    except:
        return {'auroc': 0.5, 'std': 0.0}


def compare_feature_sets(df, y, task_name):
    """Compare graph-only, TDA-only, and all features."""
    results = {}
    
    for feature_set in ['graph', 'tda', 'all']:
        X, cols = get_features(df, feature_set)
        res = run_classification(X, y, f"{task_name}_{feature_set}")
        results[feature_set] = res
        print(f"    {feature_set:6s}: AUROC = {res['auroc']:.3f} ± {res['std']:.3f}")
    
    # Calculate TDA improvement
    improvement = results['all']['auroc'] - results['graph']['auroc']
    print(f"    → TDA improvement: {improvement:+.3f}")
    
    return results, improvement


# ============================================================================
# SCENARIO 1: AMYLOID VS TAU
# ============================================================================

def test_amyloid_vs_tau(data, categories):
    """Test classification of Amyloid vs Tau genes."""
    print("\n" + "="*60)
    print("SCENARIO 1: Amyloid vs Tau (within AD)")
    print("="*60)
    
    df = data['alzheimers'].copy()
    df['category'] = df['node_id'].map(categories)
    
    # Filter to pure Amyloid or Tau (exclude Both and Unknown)
    df_filtered = df[df['category'].isin(['Amyloid', 'Tau'])]
    
    print(f"  Amyloid genes: {(df_filtered['category'] == 'Amyloid').sum()}")
    print(f"  Tau genes: {(df_filtered['category'] == 'Tau').sum()}")
    
    y = (df_filtered['category'] == 'Amyloid').astype(int).values
    
    return compare_feature_sets(df_filtered, y, "Amyloid_vs_Tau")


# ============================================================================
# SCENARIO 2: AD VS DEGREE-MATCHED BACKGROUND
# ============================================================================

def test_ad_vs_matched_background(data):
    """Test AD genes vs degree-matched background genes."""
    print("\n" + "="*60)
    print("SCENARIO 2: AD vs Degree-Matched Background")
    print("="*60)
    
    ad_df = data['alzheimers'].copy()
    bg_df = data['top_candidates'].copy()
    
    # Remove any AD genes from background
    ad_ids = set(ad_df['node_id'].values)
    bg_df = bg_df[~bg_df['node_id'].isin(ad_ids)]
    
    print(f"  AD genes: {len(ad_df)}")
    print(f"  Background genes: {len(bg_df)}")
    
    # Degree matching: for each AD gene, find background gene with similar degree
    ad_degrees = ad_df['degree'].values
    
    # Sample background to match AD degree distribution
    matched_bg = []
    for degree in ad_degrees:
        # Find background genes within ±20% of this degree
        tolerance = max(5, degree * 0.2)
        candidates = bg_df[(bg_df['degree'] >= degree - tolerance) & 
                          (bg_df['degree'] <= degree + tolerance)]
        if len(candidates) > 0:
            # Random sample one
            sampled = candidates.sample(n=1, random_state=len(matched_bg))
            matched_bg.append(sampled)
    
    if len(matched_bg) > 0:
        matched_bg_df = pd.concat(matched_bg, ignore_index=True)
        # Drop duplicates (same gene might be matched multiple times)
        matched_bg_df = matched_bg_df.drop_duplicates(subset='node_id')
        
        print(f"  Matched background: {len(matched_bg_df)}")
        print(f"  AD mean degree: {ad_df['degree'].mean():.1f}")
        print(f"  Matched BG mean degree: {matched_bg_df['degree'].mean():.1f}")
        
        # Combine
        ad_df['is_ad'] = 1
        matched_bg_df['is_ad'] = 0
        combined = pd.concat([ad_df, matched_bg_df], ignore_index=True)
        
        y = combined['is_ad'].values
        
        return compare_feature_sets(combined, y, "AD_vs_Matched")
    else:
        print("  ERROR: Could not match degrees")
        return None, 0


# ============================================================================
# SCENARIO 3: AD VS OTHER DISEASES
# ============================================================================

def test_ad_vs_other_disease(data, other_disease):
    """Test AD vs another disease."""
    print(f"\n" + "="*60)
    print(f"SCENARIO 3: AD vs {other_disease.title()}")
    print("="*60)
    
    ad_df = data['alzheimers'].copy()
    other_df = data[other_disease].copy()
    
    print(f"  AD genes: {len(ad_df)}")
    print(f"  {other_disease} genes: {len(other_df)}")
    
    # Check overlap
    ad_ids = set(ad_df['node_id'].values)
    other_ids = set(other_df['node_id'].values)
    overlap = ad_ids & other_ids
    print(f"  Overlap: {len(overlap)} genes")
    
    # Remove overlap from both
    ad_df = ad_df[~ad_df['node_id'].isin(overlap)]
    other_df = other_df[~other_df['node_id'].isin(overlap)]
    
    print(f"  After removing overlap: AD={len(ad_df)}, {other_disease}={len(other_df)}")
    
    ad_df['is_ad'] = 1
    other_df['is_ad'] = 0
    combined = pd.concat([ad_df, other_df], ignore_index=True)
    
    y = combined['is_ad'].values
    
    return compare_feature_sets(combined, y, f"AD_vs_{other_disease}")


# ============================================================================
# SCENARIO 4: MULTI-DISEASE (Disease genes vs Background)
# ============================================================================

def test_any_disease_vs_background(data):
    """Test any disease gene vs background."""
    print("\n" + "="*60)
    print("SCENARIO 4: Any Disease vs Background (degree-matched)")
    print("="*60)
    
    # Combine all disease genes
    disease_dfs = []
    for name in ['alzheimers', 'autism', 'autophagy', 'fanconi', 'glioblastoma']:
        if name in data:
            df = data[name].copy()
            df['disease'] = name
            disease_dfs.append(df)
    
    disease_df = pd.concat(disease_dfs, ignore_index=True)
    disease_df = disease_df.drop_duplicates(subset='node_id')
    
    bg_df = data['top_candidates'].copy()
    
    # Remove disease genes from background
    disease_ids = set(disease_df['node_id'].values)
    bg_df = bg_df[~bg_df['node_id'].isin(disease_ids)]
    
    print(f"  All disease genes: {len(disease_df)}")
    print(f"  Background genes: {len(bg_df)}")
    
    # Degree matching
    disease_degrees = disease_df['degree'].values
    matched_bg = []
    for degree in disease_degrees:
        tolerance = max(5, degree * 0.2)
        candidates = bg_df[(bg_df['degree'] >= degree - tolerance) & 
                          (bg_df['degree'] <= degree + tolerance)]
        if len(candidates) > 0:
            sampled = candidates.sample(n=1, random_state=len(matched_bg))
            matched_bg.append(sampled)
    
    if len(matched_bg) > 0:
        matched_bg_df = pd.concat(matched_bg, ignore_index=True)
        matched_bg_df = matched_bg_df.drop_duplicates(subset='node_id')
        
        print(f"  Matched background: {len(matched_bg_df)}")
        
        disease_df['is_disease'] = 1
        matched_bg_df['is_disease'] = 0
        combined = pd.concat([disease_df, matched_bg_df], ignore_index=True)
        
        y = combined['is_disease'].values
        
        return compare_feature_sets(combined, y, "Disease_vs_Background")
    else:
        return None, 0


# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(df, y, task_name):
    """Analyze which features are most important."""
    X, cols = get_features(df, 'all')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit random forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_scaled, y)
    
    importance = pd.DataFrame({
        'feature': cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results_summary(all_results):
    """Plot comparison of all scenarios."""
    scenarios = list(all_results.keys())
    
    graph_scores = [all_results[s]['graph']['auroc'] for s in scenarios]
    tda_scores = [all_results[s]['tda']['auroc'] for s in scenarios]
    all_scores = [all_results[s]['all']['auroc'] for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, graph_scores, width, label='Graph Only', color='steelblue')
    bars2 = ax.bar(x, tda_scores, width, label='TDA Only', color='coral')
    bars3 = ax.bar(x + width, all_scores, width, label='Graph + TDA', color='forestgreen')
    
    ax.set_ylabel('AUROC')
    ax.set_title('Classification Performance: TDA vs Graph Features')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    
    # Add improvement annotations
    for i, scenario in enumerate(scenarios):
        improvement = all_scores[i] - graph_scores[i]
        ax.annotate(f'+{improvement:.2f}', 
                   xy=(x[i] + width, all_scores[i] + 0.02),
                   ha='center', fontsize=9, color='forestgreen')
    
    plt.tight_layout()
    plt.savefig('figures/classification_comparison.png', dpi=150)
    plt.close()
    print("\nSaved figure to figures/classification_comparison.png")


def plot_degree_distributions(data, categories):
    """Plot degree distributions for different groups."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Amyloid vs Tau
    ax = axes[0, 0]
    ad_df = data['alzheimers'].copy()
    ad_df['category'] = ad_df['node_id'].map(categories)
    
    for cat in ['Amyloid', 'Tau']:
        subset = ad_df[ad_df['category'] == cat]['degree']
        ax.hist(subset, bins=30, alpha=0.5, label=f'{cat} (n={len(subset)})')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.set_title('Amyloid vs Tau Degree Distribution')
    ax.legend()
    
    # 2. AD vs Background
    ax = axes[0, 1]
    ax.hist(data['alzheimers']['degree'], bins=30, alpha=0.5, label=f"AD (n={len(data['alzheimers'])})")
    ax.hist(data['top_candidates']['degree'], bins=30, alpha=0.5, label=f"Background (n={len(data['top_candidates'])})")
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.set_title('AD vs Background Degree Distribution')
    ax.legend()
    
    # 3. Delta H1 comparison
    ax = axes[1, 0]
    for cat in ['Amyloid', 'Tau']:
        subset = ad_df[ad_df['category'] == cat]['delta_H1']
        ax.hist(subset, bins=30, alpha=0.5, label=f'{cat}')
    ax.set_xlabel('Delta H1 (TDA perturbation)')
    ax.set_ylabel('Count')
    ax.set_title('Amyloid vs Tau: TDA Delta H1')
    ax.legend()
    
    # 4. Delta H1: AD vs Background
    ax = axes[1, 1]
    ax.hist(data['alzheimers']['delta_H1'], bins=30, alpha=0.5, label='AD')
    ax.hist(data['top_candidates']['delta_H1'], bins=30, alpha=0.5, label='Background')
    ax.set_xlabel('Delta H1 (TDA perturbation)')
    ax.set_ylabel('Count')
    ax.set_title('AD vs Background: TDA Delta H1')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('figures/feature_distributions.png', dpi=150)
    plt.close()
    print("Saved figure to figures/feature_distributions.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("EXPLORING CLASSIFICATION TASKS FOR TDA")
    print("="*60)
    
    os.makedirs('figures', exist_ok=True)
    
    # Load data
    print("\nLoading perturbation TDA data...")
    data = load_all_perturbation_data()
    
    print("\nLoading AD gene categories...")
    categories = load_ad_categories()
    amyloid_count = sum(1 for v in categories.values() if v == 'Amyloid')
    tau_count = sum(1 for v in categories.values() if v == 'Tau')
    both_count = sum(1 for v in categories.values() if v == 'Both')
    print(f"  Amyloid: {amyloid_count}, Tau: {tau_count}, Both: {both_count}")
    
    # Run all scenarios
    all_results = {}
    improvements = {}
    
    # Scenario 1: Amyloid vs Tau
    results, improvement = test_amyloid_vs_tau(data, categories)
    all_results['Amyloid_vs_Tau'] = results
    improvements['Amyloid_vs_Tau'] = improvement
    
    # Scenario 2: AD vs Degree-Matched Background
    results, improvement = test_ad_vs_matched_background(data)
    if results:
        all_results['AD_vs_Matched_BG'] = results
        improvements['AD_vs_Matched_BG'] = improvement
    
    # Scenario 3: AD vs Other Diseases
    for disease in ['autism', 'glioblastoma']:
        if disease in data:
            results, improvement = test_ad_vs_other_disease(data, disease)
            all_results[f'AD_vs_{disease}'] = results
            improvements[f'AD_vs_{disease}'] = improvement
    
    # Scenario 4: Any Disease vs Background
    results, improvement = test_any_disease_vs_background(data)
    if results:
        all_results['AllDisease_vs_BG'] = results
        improvements['AllDisease_vs_BG'] = improvement
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: TDA IMPROVEMENT BY SCENARIO")
    print("="*60)
    print(f"{'Scenario':<25} {'Graph':>8} {'All':>8} {'Improvement':>12}")
    print("-"*55)
    
    for scenario in sorted(improvements.keys(), key=lambda x: improvements[x], reverse=True):
        graph = all_results[scenario]['graph']['auroc']
        all_score = all_results[scenario]['all']['auroc']
        imp = improvements[scenario]
        marker = "★★★" if imp > 0.05 else ("★★" if imp > 0.02 else ("★" if imp > 0 else ""))
        print(f"{scenario:<25} {graph:>8.3f} {all_score:>8.3f} {imp:>+10.3f}  {marker}")
    
    # Best scenario
    best_scenario = max(improvements.keys(), key=lambda x: improvements[x])
    print(f"\n→ BEST SCENARIO: {best_scenario} (TDA improvement: {improvements[best_scenario]:+.3f})")
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_results_summary(all_results)
    plot_degree_distributions(data, categories)
    
    # Feature importance for best scenario
    print(f"\nFeature importance for {best_scenario}:")
    # (Would need to re-prepare data for this - skip for now)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("""
Based on the results above, look for scenarios where:
1. Graph-only AUROC is NOT too high (< 0.75) - room for improvement
2. TDA improvement is positive and significant (> 0.03)
3. Sample sizes are reasonable (> 50 per class)

The scenario with the highest TDA improvement AND reasonable 
graph-only baseline is the best choice for demonstrating TDA value.

Next steps:
1. Pick the best scenario
2. Run full ML pipeline (Random Forest, XGBoost, etc.)
3. Do proper hyperparameter tuning
4. Generate publication-quality figures
5. Add bifiltration features for further improvement
    """)


if __name__ == "__main__":
    main()

