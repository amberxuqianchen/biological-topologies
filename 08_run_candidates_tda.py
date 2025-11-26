#!/usr/bin/env python3
"""
Run TDA Feature Extraction on Candidate Genes

This script uses 07_perturbation_tda.py as a module to compute TDA features
for candidate AD genes (non-AD genes with strong AD connections).

Usage:
    python 08_run_candidates_tda.py                 # Run on top 100 candidates
    python 08_run_candidates_tda.py --n 500         # Run on top 500 candidates
    python 08_run_candidates_tda.py --random 1000   # Run on 1000 random non-AD genes
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os

# Import from our perturbation TDA module
from importlib import import_module

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perturbation_tda_module import (
    load_human_interactome,
    load_ad_genes,
    compute_features_for_genes
)


def load_candidate_genes(filepath='computed_data/top_ad_candidates.csv', n=None):
    """Load top candidate genes."""
    df = pd.read_csv(filepath)
    
    if n is not None:
        df = df.head(n)
    
    return df['gene_id'].tolist(), df


def get_random_non_ad_genes(G, ad_genes, n=1000, seed=42):
    """Get random non-AD genes from the network."""
    np.random.seed(seed)
    
    all_genes = set(G.nodes())
    non_ad_genes = list(all_genes - ad_genes)
    
    if n > len(non_ad_genes):
        n = len(non_ad_genes)
    
    return list(np.random.choice(non_ad_genes, size=n, replace=False))


def main():
    parser = argparse.ArgumentParser(description='Run TDA on candidate genes')
    parser.add_argument('--n', type=int, default=100, 
                        help='Number of top candidates to process')
    parser.add_argument('--random', type=int, default=None,
                        help='Use N random non-AD genes instead of candidates')
    parser.add_argument('--max-ego-size', type=int, default=300,
                        help='Max ego graph size')
    parser.add_argument('--all-non-ad', action='store_true',
                        help='Run on ALL non-AD genes (warning: slow!)')
    args = parser.parse_args()
    
    print("="*70)
    print("TDA FEATURE EXTRACTION FOR CANDIDATE GENES")
    print("="*70)
    
    # Load network and AD genes
    G = load_human_interactome()
    ad_genes, gene_categories = load_ad_genes()
    
    # Determine which genes to process
    if args.all_non_ad:
        all_genes = set(G.nodes())
        genes_to_process = list(all_genes - ad_genes)
        output_file = 'computed_data/tda_features_all_non_ad.csv'
        print(f"\nProcessing ALL {len(genes_to_process):,} non-AD genes...")
        
    elif args.random:
        genes_to_process = get_random_non_ad_genes(G, ad_genes, n=args.random)
        output_file = f'computed_data/tda_features_random_{args.random}_non_ad.csv'
        print(f"\nProcessing {len(genes_to_process)} random non-AD genes...")
        
    else:
        genes_to_process, candidate_df = load_candidate_genes(n=args.n)
        output_file = f'computed_data/tda_features_top_{args.n}_candidates.csv'
        print(f"\nProcessing top {len(genes_to_process)} candidate genes...")
        print(f"  (Non-AD genes with most AD connections)")
    
    # Filter to genes in network
    genes_in_network = [g for g in genes_to_process if g in G]
    print(f"Genes in network: {len(genes_in_network)}/{len(genes_to_process)}")
    
    # Run TDA
    df = compute_features_for_genes(
        G, genes_in_network,
        ad_genes=ad_genes,
        gene_categories=gene_categories,
        max_ego_size=args.max_ego_size,
        output_file=output_file
    )
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Genes processed: {len(df)}")
    print(f"Output saved to: {output_file}")
    
    print(f"\nDelta H1 distribution:")
    print(f"  Mean: {df['delta_H1'].mean():.2f}")
    print(f"  Std:  {df['delta_H1'].std():.2f}")
    print(f"  Min:  {df['delta_H1'].min()}")
    print(f"  Max:  {df['delta_H1'].max()}")
    
    # Compare with AD genes if available
    ad_features_file = 'computed_data/tda_features_ad_genes.csv'
    if os.path.exists(ad_features_file):
        ad_df = pd.read_csv(ad_features_file)
        print("\n" + "-"*70)
        print("COMPARISON: AD genes vs Candidates")
        print("-"*70)
        print(f"{'Metric':<20} {'AD Genes':>15} {'Candidates':>15}")
        print("-"*50)
        for col in ['delta_H1', 'delta_H2', 'delta_H3']:
            print(f"{col:<20} {ad_df[col].mean():>15.2f} {df[col].mean():>15.2f}")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()

