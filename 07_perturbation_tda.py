#!/usr/bin/env python3
"""
Perturbation TDA Analysis

Computes how removing a gene affects the local network topology.
For each gene, we:
1. Extract its ego graph (2-hop neighborhood, subsampled if needed)
2. Compute Betti numbers using clique complex (H0, H1, H2, H3)
3. Remove the gene, recompute Betti numbers
4. Return delta values (change when node is removed)

This captures "topological importance" - how critical is this gene
to the structure of its local neighborhood?

Usage:
    python 07_perturbation_tda.py                    # Run on AD genes
    python 07_perturbation_tda.py --all              # Run on all genes
    python 07_perturbation_tda.py --sample 1000      # Run on 1000 random genes
"""

import numpy as np
import networkx as nx
import pandas as pd
import gudhi
import time
import argparse
from tqdm import tqdm
import os


# ============================================================================
# DATA LOADING
# ============================================================================

def load_human_interactome(filepath='computed_data/BIOGRID-HUMAN-5.0.251.tab3.txt'):
    """Load the full human protein interaction network."""
    print("Loading human interactome...")
    df = pd.read_csv(filepath, sep='\t', low_memory=False)
    
    # Filter to physical interactions
    physical = df[df['Experimental System Type'] == 'physical']
    
    # Build graph
    G = nx.Graph()
    for _, row in physical.iterrows():
        try:
            a = int(row['Entrez Gene Interactor A'])
            b = int(row['Entrez Gene Interactor B'])
            if a != b:
                G.add_edge(a, b)
        except:
            continue
    
    # Get largest connected component
    lcc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc).copy()
    
    print(f"  Nodes: {G_lcc.number_of_nodes():,}")
    print(f"  Edges: {G_lcc.number_of_edges():,}")
    
    return G_lcc


def load_ad_genes(filepath='data/BIOGRID-PROJECT-alzheimers_disease_project-GENES-5.0.250.projectindex.txt'):
    """Load AD gene list with category information."""
    df = pd.read_csv(filepath, sep='\t')
    
    ad_genes = set(df['ENTREZ GENE ID'].values)
    
    # Parse categories (Amyloid vs Tau)
    gene_categories = {}
    for _, row in df.iterrows():
        gene_id = row['ENTREZ GENE ID']
        cats = row.get('CATEGORY VALUES', '-')
        if pd.notna(cats) and cats != '-':
            if 'Amyloid' in str(cats) and 'Tau' in str(cats):
                gene_categories[gene_id] = 'Both'
            elif 'Amyloid' in str(cats):
                gene_categories[gene_id] = 'Amyloid'
            elif 'Tau' in str(cats):
                gene_categories[gene_id] = 'Tau'
            else:
                gene_categories[gene_id] = 'Other'
        else:
            gene_categories[gene_id] = 'Unknown'
    
    return ad_genes, gene_categories


# ============================================================================
# CORE TDA FUNCTIONS
# ============================================================================

def extract_ego_graph(G, node, radius=2, max_size=300):
    """
    Extract ego graph around a node with BFS subsampling.
    
    Parameters:
    -----------
    G : networkx.Graph
        The full network
    node : int
        Center node
    radius : int
        Hop distance for ego graph
    max_size : int
        Maximum number of nodes (uses BFS subsampling if exceeded)
    
    Returns:
    --------
    ego : networkx.Graph
        The ego graph
    original_size : int
        Original size before subsampling
    was_subsampled : bool
        Whether subsampling was applied
    """
    ego = nx.ego_graph(G, node, radius=radius)
    original_size = len(ego.nodes())
    was_subsampled = False
    
    if original_size > max_size:
        was_subsampled = True
        # BFS-based sampling to ensure connected subgraph centered on node
        visited = set([node])
        queue = [node]
        while len(visited) < max_size and queue:
            current = queue.pop(0)
            for neighbor in ego.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(visited) >= max_size:
                        break
        ego = ego.subgraph(list(visited)).copy()
    
    return ego, original_size, was_subsampled


def compute_betti_clique(G, max_dim=3):
    """
    Compute Betti numbers using GUDHI clique complex.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph
    max_dim : int
        Maximum homology dimension (3 for H0, H1, H2, H3)
    
    Returns:
    --------
    betti : list
        [β₀, β₁, β₂, β₃] Betti numbers
    n_simplices : int
        Number of simplices in the complex
    """
    if len(G.nodes()) < 2:
        return [1] + [0] * max_dim, 0
    
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Build simplex tree from edges
    st = gudhi.SimplexTree()
    for u, v in G.edges():
        st.insert([node_to_idx[u], node_to_idx[v]])
    
    # Expand to clique complex
    st.expansion(max_dim + 1)
    
    # Compute persistence and Betti numbers
    st.compute_persistence()
    betti = st.betti_numbers()
    
    # Pad with zeros if needed
    while len(betti) <= max_dim:
        betti.append(0)
    
    return betti[:max_dim+1], st.num_simplices()


def compute_perturbation_features(G, node, radius=2, max_ego_size=300, max_dim=3):
    """
    Compute perturbation TDA features for a single node.
    
    This is the main "black box" function.
    
    Parameters:
    -----------
    G : networkx.Graph
        The full network
    node : int
        The node to analyze
    radius : int
        Ego graph radius (default 2)
    max_ego_size : int
        Maximum ego graph size (default 300)
    max_dim : int
        Maximum homology dimension (default 3 for H0-H3)
    
    Returns:
    --------
    dict with all features
    """
    start_time = time.time()
    
    # Check if node exists
    if node not in G:
        return None
    
    # Extract ego graph
    ego, original_size, was_subsampled = extract_ego_graph(
        G, node, radius=radius, max_size=max_ego_size
    )
    
    # Baseline: Betti with center node
    betti_with, n_simp_with = compute_betti_clique(ego, max_dim=max_dim)
    
    # Perturbed: Betti without center node
    ego_without = ego.copy()
    ego_without.remove_node(node)
    
    if len(ego_without.nodes()) >= 2:
        betti_without, n_simp_without = compute_betti_clique(ego_without, max_dim=max_dim)
    else:
        betti_without = [0] * (max_dim + 1)
        n_simp_without = 0
    
    total_time = time.time() - start_time
    
    # Build feature dictionary
    features = {
        # Metadata
        'node_id': node,
        'degree': G.degree(node),
        'ego_original_size': original_size,
        'ego_actual_size': len(ego.nodes()),
        'ego_edges': ego.number_of_edges(),
        'was_subsampled': was_subsampled,
        'n_simplices': n_simp_with,
        'time_seconds': total_time,
        
        # Baseline Betti (with node)
        'H0_with': betti_with[0],
        'H1_with': betti_with[1],
        'H2_with': betti_with[2],
        'H3_with': betti_with[3] if max_dim >= 3 else 0,
        
        # Perturbed Betti (without node)
        'H0_without': betti_without[0],
        'H1_without': betti_without[1],
        'H2_without': betti_without[2],
        'H3_without': betti_without[3] if max_dim >= 3 else 0,
        
        # Delta (with - without)
        # Positive = removing node destroyed features (node was part of them)
        # Negative = removing node created features (node was filling holes)
        'delta_H0': betti_with[0] - betti_without[0],
        'delta_H1': betti_with[1] - betti_without[1],
        'delta_H2': betti_with[2] - betti_without[2],
        'delta_H3': (betti_with[3] if max_dim >= 3 else 0) - (betti_without[3] if max_dim >= 3 else 0),
    }
    
    return features


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def compute_features_for_genes(G, genes, ad_genes=None, gene_categories=None, 
                                max_ego_size=300, output_file=None):
    """
    Compute perturbation TDA features for a list of genes.
    
    Parameters:
    -----------
    G : networkx.Graph
        The network
    genes : list
        List of gene IDs to analyze
    ad_genes : set or None
        Set of AD gene IDs (for labeling)
    gene_categories : dict or None
        Dict mapping gene ID to category (Amyloid/Tau)
    max_ego_size : int
        Maximum ego graph size
    output_file : str or None
        Path to save CSV results
    
    Returns:
    --------
    pd.DataFrame with all features
    """
    results = []
    skipped = 0
    
    print(f"\nComputing TDA features for {len(genes)} genes...")
    print(f"Settings: max_ego_size={max_ego_size}, max_dim=3 (H0-H3)")
    
    start_time = time.time()
    
    for gene in tqdm(genes, desc="Processing genes"):
        try:
            features = compute_perturbation_features(
                G, gene, radius=2, max_ego_size=max_ego_size, max_dim=3
            )
            
            if features is None:
                skipped += 1
                continue
            
            # Add labels
            if ad_genes is not None:
                features['is_ad'] = gene in ad_genes
            if gene_categories is not None:
                features['ad_category'] = gene_categories.get(gene, 'Non-AD')
            
            results.append(features)
            
        except Exception as e:
            print(f"\n  Error on gene {gene}: {e}")
            skipped += 1
            continue
    
    total_time = time.time() - start_time
    
    print(f"\nCompleted: {len(results)} genes processed, {skipped} skipped")
    print(f"Total time: {total_time:.1f}s ({total_time/max(1,len(results)):.3f}s per gene)")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
    
    return df


# ============================================================================
# MAIN ENTRY POINTS
# ============================================================================

def run_ad_genes(G, ad_genes, gene_categories, max_ego_size=300):
    """Run analysis on all AD genes."""
    # Filter to AD genes that are in the network
    genes_in_network = [g for g in ad_genes if g in G]
    print(f"\nAD genes in network: {len(genes_in_network)} / {len(ad_genes)}")
    
    output_file = 'computed_data/tda_features_ad_genes.csv'
    
    df = compute_features_for_genes(
        G, genes_in_network, 
        ad_genes=ad_genes,
        gene_categories=gene_categories,
        max_ego_size=max_ego_size,
        output_file=output_file
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: AD Genes TDA Features")
    print("="*60)
    print(f"Total genes: {len(df)}")
    print(f"\nCategory breakdown:")
    if 'ad_category' in df.columns:
        print(df['ad_category'].value_counts())
    print(f"\nDelta H1 distribution:")
    print(f"  Mean: {df['delta_H1'].mean():.2f}")
    print(f"  Std:  {df['delta_H1'].std():.2f}")
    print(f"  Min:  {df['delta_H1'].min()}")
    print(f"  Max:  {df['delta_H1'].max()}")
    
    return df


def run_all_genes(G, ad_genes, gene_categories, max_ego_size=300, sample=None):
    """Run analysis on all genes (or a sample)."""
    all_genes = list(G.nodes())
    
    if sample is not None:
        np.random.seed(42)
        all_genes = list(np.random.choice(all_genes, size=min(sample, len(all_genes)), replace=False))
        print(f"\nSampling {len(all_genes)} genes...")
    
    output_file = f'computed_data/tda_features_all_genes{"_sample" + str(sample) if sample else ""}.csv'
    
    df = compute_features_for_genes(
        G, all_genes,
        ad_genes=ad_genes,
        gene_categories=gene_categories,
        max_ego_size=max_ego_size,
        output_file=output_file
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: All Genes TDA Features")
    print("="*60)
    print(f"Total genes: {len(df)}")
    print(f"AD genes: {df['is_ad'].sum()}")
    print(f"Non-AD genes: {(~df['is_ad']).sum()}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Compute perturbation TDA features')
    parser.add_argument('--all', action='store_true', help='Run on all genes')
    parser.add_argument('--sample', type=int, default=None, help='Sample N random genes')
    parser.add_argument('--max-ego-size', type=int, default=300, help='Max ego graph size')
    args = parser.parse_args()
    
    print("="*60)
    print("PERTURBATION TDA FEATURE EXTRACTION")
    print("="*60)
    
    # Load data
    G = load_human_interactome()
    ad_genes, gene_categories = load_ad_genes()
    
    print(f"\nAD genes loaded: {len(ad_genes)}")
    
    # Run analysis
    if args.all or args.sample:
        df = run_all_genes(G, ad_genes, gene_categories, 
                          max_ego_size=args.max_ego_size, 
                          sample=args.sample)
    else:
        df = run_ad_genes(G, ad_genes, gene_categories,
                         max_ego_size=args.max_ego_size)
    
    print("\n" + "="*60)
    print("DONE! Features saved to computed_data/")
    print("="*60)


if __name__ == "__main__":
    main()
