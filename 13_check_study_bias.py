#!/usr/bin/env python3
"""
Check for Study Bias in TDA Features

If ALL disease projects show fewer simplices than background,
it's likely study bias (how disease genes are researched).

If only specific diseases show the pattern, it might be real biology.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_all_data():
    """Load all perturbation TDA data."""
    data = {}
    files = {
        'Alzheimer\'s': 'computed_data/tda_perturbation_alzheimers.csv',
        'Autism': 'computed_data/tda_perturbation_autism.csv',
        'Autophagy': 'computed_data/tda_perturbation_autophagy.csv',
        'Fanconi': 'computed_data/tda_perturbation_fanconi.csv',
        'Glioblastoma': 'computed_data/tda_perturbation_glioblastoma.csv',
        'Background': 'computed_data/tda_perturbation_top_candidates.csv',
    }
    
    for name, path in files.items():
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
            print(f"  {name}: {len(data[name])} genes")
    
    return data


def compare_simplices_by_disease(data):
    """Compare n_simplices across all disease projects vs background."""
    print("\n" + "="*70)
    print("COMPARING N_SIMPLICES ACROSS ALL DISEASE PROJECTS")
    print("="*70)
    
    bg = data['Background']
    
    # Remove disease genes from background for fair comparison
    all_disease_ids = set()
    for name, df in data.items():
        if name != 'Background':
            all_disease_ids.update(df['node_id'].values)
    
    bg_clean = bg[~bg['node_id'].isin(all_disease_ids)]
    print(f"\nBackground (excluding disease genes): {len(bg_clean)}")
    
    results = []
    
    print(f"\n{'Disease':<15} {'N':>6} {'Mean Degree':>12} {'Mean Simplices':>15} {'BG Simplices':>15} {'Ratio':>8} {'p-value':>10}")
    print("-" * 90)
    
    for name, df in data.items():
        if name == 'Background':
            continue
        
        # Degree-match background to this disease
        matched_bg = []
        for degree in df['degree'].values:
            tolerance = max(5, degree * 0.2)
            candidates = bg_clean[(bg_clean['degree'] >= degree - tolerance) & 
                                  (bg_clean['degree'] <= degree + tolerance)]
            if len(candidates) > 0:
                sampled = candidates.sample(n=1, random_state=len(matched_bg))
                matched_bg.append(sampled)
        
        if len(matched_bg) < 10:
            print(f"{name:<15} {'Too few matches'}")
            continue
            
        matched_bg_df = pd.concat(matched_bg, ignore_index=True)
        matched_bg_df = matched_bg_df.drop_duplicates(subset='node_id')
        
        # Compare n_simplices
        disease_simplices = df['n_simplices'].values
        bg_simplices = matched_bg_df['n_simplices'].values
        
        disease_mean = np.mean(disease_simplices)
        bg_mean = np.mean(bg_simplices)
        ratio = disease_mean / bg_mean if bg_mean > 0 else 0
        
        # t-test
        t_stat, p_val = stats.ttest_ind(disease_simplices, bg_simplices)
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        
        print(f"{name:<15} {len(df):>6} {df['degree'].mean():>12.1f} {disease_mean:>15.0f} {bg_mean:>15.0f} {ratio:>8.2f} {p_val:>9.4f} {sig}")
        
        results.append({
            'disease': name,
            'n': len(df),
            'mean_degree': df['degree'].mean(),
            'disease_simplices': disease_mean,
            'bg_simplices': bg_mean,
            'ratio': ratio,
            'p_value': p_val,
            'significant': p_val < 0.05
        })
    
    return pd.DataFrame(results)


def compare_other_tda_features(data):
    """Compare other TDA features to see if pattern is consistent."""
    print("\n" + "="*70)
    print("COMPARING OTHER TDA FEATURES (delta_H1, delta_H2, H1_with)")
    print("="*70)
    
    bg = data['Background']
    all_disease_ids = set()
    for name, df in data.items():
        if name != 'Background':
            all_disease_ids.update(df['node_id'].values)
    bg_clean = bg[~bg['node_id'].isin(all_disease_ids)]
    
    features_to_check = ['delta_H1', 'delta_H2', 'H1_with', 'H2_with']
    
    for feature in features_to_check:
        print(f"\n--- {feature} ---")
        print(f"{'Disease':<15} {'Disease Mean':>12} {'BG Mean':>12} {'Diff':>10} {'p-value':>10}")
        print("-" * 60)
        
        for name, df in data.items():
            if name == 'Background':
                continue
            if feature not in df.columns:
                continue
            
            # Degree-match
            matched_bg = []
            for degree in df['degree'].values:
                tolerance = max(5, degree * 0.2)
                candidates = bg_clean[(bg_clean['degree'] >= degree - tolerance) & 
                                      (bg_clean['degree'] <= degree + tolerance)]
                if len(candidates) > 0:
                    sampled = candidates.sample(n=1, random_state=len(matched_bg) + hash(feature) % 1000)
                    matched_bg.append(sampled)
            
            if len(matched_bg) < 10:
                continue
                
            matched_bg_df = pd.concat(matched_bg, ignore_index=True)
            
            disease_vals = df[feature].values
            bg_vals = matched_bg_df[feature].values
            
            disease_mean = np.mean(disease_vals)
            bg_mean = np.mean(bg_vals)
            diff = disease_mean - bg_mean
            
            t_stat, p_val = stats.ttest_ind(disease_vals, bg_vals)
            sig = "*" if p_val < 0.05 else ""
            
            print(f"{name:<15} {disease_mean:>12.2f} {bg_mean:>12.2f} {diff:>+10.2f} {p_val:>9.4f} {sig}")


def check_degree_distribution(data):
    """Check if degree distributions differ between diseases."""
    print("\n" + "="*70)
    print("DEGREE DISTRIBUTION COMPARISON")
    print("="*70)
    
    print(f"\n{'Disease':<15} {'Mean':>10} {'Median':>10} {'Std':>10} {'Max':>10}")
    print("-" * 60)
    
    for name, df in data.items():
        mean_d = df['degree'].mean()
        median_d = df['degree'].median()
        std_d = df['degree'].std()
        max_d = df['degree'].max()
        print(f"{name:<15} {mean_d:>10.1f} {median_d:>10.1f} {std_d:>10.1f} {max_d:>10.0f}")


def plot_simplices_comparison(data, results_df):
    """Visualize the comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Ratio of disease/background simplices
    ax = axes[0]
    diseases = results_df['disease'].values
    ratios = results_df['ratio'].values
    colors = ['red' if r < 1 else 'green' for r in ratios]
    
    bars = ax.barh(diseases, ratios, color=colors, alpha=0.7)
    ax.axvline(x=1, color='black', linestyle='--', label='Equal to background')
    ax.set_xlabel('Ratio (Disease / Background n_simplices)')
    ax.set_title('n_simplices: Disease vs Degree-Matched Background')
    
    # Add significance markers
    for i, (ratio, pval) in enumerate(zip(ratios, results_df['p_value'])):
        marker = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        ax.text(ratio + 0.02, i, marker, va='center', fontsize=12)
    
    # Plot 2: Box plots of n_simplices by disease
    ax = axes[1]
    plot_data = []
    labels = []
    
    for name, df in data.items():
        if 'n_simplices' in df.columns:
            # Log transform for better visualization
            values = np.log10(df['n_simplices'].values + 1)
            plot_data.append(values)
            labels.append(name)
    
    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
    
    # Color boxes
    colors = ['coral' if l != 'Background' else 'steelblue' for l in labels]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('log10(n_simplices)')
    ax.set_title('Distribution of n_simplices by Project')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('figures/study_bias_check.png', dpi=150)
    plt.close()
    print("\nSaved figure to figures/study_bias_check.png")


def main():
    print("="*70)
    print("CHECKING FOR STUDY BIAS IN TDA FEATURES")
    print("="*70)
    print("""
If ALL disease projects have fewer simplices than background → STUDY BIAS
If only some diseases differ → might be REAL BIOLOGY
    """)
    
    os.makedirs('figures', exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_all_data()
    
    # Check degree distributions first
    check_degree_distribution(data)
    
    # Compare n_simplices
    results_df = compare_simplices_by_disease(data)
    
    # Compare other features
    compare_other_tda_features(data)
    
    # Visualization
    if len(results_df) > 0:
        plot_simplices_comparison(data, results_df)
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if len(results_df) > 0:
        all_lower = all(results_df['ratio'] < 1)
        all_significant = all(results_df['significant'])
        
        if all_lower and all_significant:
            print("""
⚠️  ALL disease projects have significantly fewer simplices than background!

This strongly suggests STUDY BIAS:
- Disease genes are studied with "hub-centric" approaches
- Researchers focus on the disease gene, not its neighbors' interactions  
- This creates artificially sparse local topologies

The n_simplices feature is likely capturing HOW genes are studied,
not their true biological network structure.

RECOMMENDATION: 
- Be cautious about interpreting n_simplices as biology
- Focus on features less correlated with study attention (delta_H2?)
- Consider this a methodological finding for your paper
            """)
        elif all_lower:
            print("""
Most disease projects have fewer simplices, but not all significant.
This suggests PARTIAL study bias - the effect exists but varies.
            """)
        else:
            print("""
Mixed results - some diseases differ, others don't.
This could indicate REAL BIOLOGICAL differences between diseases,
or varying levels of study bias per disease.
            """)
    
    # What features might still be valid?
    print("\n" + "="*70)
    print("WHICH FEATURES MIGHT STILL BE VALID?")
    print("="*70)
    print("""
Features to investigate further:
- delta_H2: Showed significance for AD, might capture real topology
- Features normalized by n_simplices or degree
- Relative metrics (ratios) rather than absolute counts
    
The classification improvement might still be real, but driven by
study bias rather than biology. This is still publishable as a
methodological finding!
    """)


if __name__ == "__main__":
    main()

