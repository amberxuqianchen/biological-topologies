#!/usr/bin/env python3
"""
Bifiltration TDA Analysis (Core Module)

Computes TDA features at different PTM thresholds to capture
how topology changes across biochemical activity levels.

Usage:
    # Compute features for a single gene
    from 17_bifiltration_tda import compute_bifiltration_features, load_ptm_counts
    
    ptm_counts = load_ptm_counts()
    features = compute_bifiltration_features(G, gene_id, ptm_counts)
    
    # Or run on AD genes directly
    python 17_bifiltration_tda.py
"""

import numpy as np
import networkx as nx
import pandas as pd
import gudhi
import time
import os
from tqdm import tqdm

# ============================================================================
# PTM DATA LOADING
# ============================================================================

def load_ptm_counts(filepaths=None):
    """
    Load PTM counts per gene from BioGRID PTM files.
    
    Parameters:
    -----------
    filepaths : list or None
        List of PTM file paths. If None, uses default AD and other project files.
    
    Returns:
    --------
    dict : {gene_id: ptm_count}
    """
    if filepaths is None:
        filepaths = [
            'data/BIOGRID-PROJECT-alzheimers_disease_project-PTM-5.0.250.ptmtab.txt',
            'data/BIOGRID-PROJECT-autism_spectrum_disorder_asd_project-LATEST/BIOGRID-PROJECT-autism_spectrum_disorder_asd_project-PTM-5.0.251.ptmtab.txt',
            'data/BIOGRID-PROJECT-glioblastoma_project-LATEST/BIOGRID-PROJECT-glioblastoma_project-PTM-5.0.251.ptmtab.txt',
            'data/BIOGRID-PROJECT-autophagy_project-LATEST/BIOGRID-PROJECT-autophagy_project-PTM-5.0.251.ptmtab.txt',
            'data/BIOGRID-PROJECT-fanconi_anemia_project-LATEST/BIOGRID-PROJECT-fanconi_anemia_project-PTM-5.0.251.ptmtab.txt',
        ]
    
    print("Loading PTM data...")
    all_ptm = {}
    
    for filepath in filepaths:
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, sep='\t', low_memory=False)
                df = df[df['Organism ID'] == 9606]  # Human only
                for gene_id, count in df.groupby('Entrez Gene ID').size().items():
                    all_ptm[gene_id] = all_ptm.get(gene_id, 0) + count
                print(f"  Loaded: {filepath.split('/')[-1]}")
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")
                continue
    
    print(f"  Total: PTM counts for {len(all_ptm)} genes")
    return all_ptm


# ============================================================================
# CORE TDA FUNCTIONS
# ============================================================================

def extract_ego_graph(G, node, radius=2, max_size=300):
    """
    Extract ego graph around a node with BFS subsampling.
    """
    ego = nx.ego_graph(G, node, radius=radius)
    original_size = len(ego.nodes())
    was_subsampled = False
    
    if original_size > max_size:
        was_subsampled = True
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


def compute_betti_clique(G, max_dim=2):
    """
    Compute Betti numbers using GUDHI clique complex.
    """
    if len(G.nodes()) < 2:
        return [1] + [0] * max_dim
    
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    st = gudhi.SimplexTree()
    for u, v in G.edges():
        st.insert([node_to_idx[u], node_to_idx[v]])
    
    st.expansion(max_dim + 1)
    st.compute_persistence()
    betti = st.betti_numbers()
    
    while len(betti) <= max_dim:
        betti.append(0)
    
    return betti[:max_dim + 1]


# ============================================================================
# BIFILTRATION FEATURE COMPUTATION
# ============================================================================

def compute_bifiltration_features(G, node, ptm_counts, 
                                   radius=2,
                                   max_ego_size=300,
                                   percentiles=[10, 25, 50, 75, 90, 100],
                                   max_dim=2):
    """
    Compute bifiltration TDA features for a single node.
    
    This is the main "black box" function for bifiltration.
    
    Parameters:
    -----------
    G : networkx.Graph
        The full network
    node : int
        The node to analyze
    ptm_counts : dict
        {gene_id: ptm_count} mapping
    radius : int
        Ego graph radius (default 2)
    max_ego_size : int
        Maximum ego graph size (default 300)
    percentiles : list
        PTM percentile thresholds for bifiltration
    max_dim : int
        Maximum homology dimension (default 2 for H0, H1, H2)
    
    Returns:
    --------
    dict with all bifiltration features, or None if node not in G
    """
    start_time = time.time()
    
    if node not in G:
        return None
    
    # Extract ego graph
    ego, original_size, was_subsampled = extract_ego_graph(
        G, node, radius=radius, max_size=max_ego_size
    )
    
    # Get PTM counts for nodes in ego graph (0 if not in PTM data)
    ego_ptm = {n: ptm_counts.get(n, 0) for n in ego.nodes()}
    ptm_values = list(ego_ptm.values())
    
    if len(ptm_values) == 0:
        return None
    
    # Basic features
    features = {
        'node_id': node,
        'degree': G.degree(node),
        'center_ptm': ptm_counts.get(node, 0),
        'ego_original_size': original_size,
        'ego_actual_size': len(ego.nodes()),
        'ego_edges': ego.number_of_edges(),
        'was_subsampled': was_subsampled,
        'ego_mean_ptm': np.mean(ptm_values),
        'ego_max_ptm': np.max(ptm_values),
        'ego_ptm_std': np.std(ptm_values) if len(ptm_values) > 1 else 0,
    }
    
    # Compute Betti at each percentile threshold
    betti_series = {f'H{d}': [] for d in range(max_dim + 1)}
    delta_series = {f'delta_H{d}': [] for d in range(1, max_dim + 1)}
    
    for pct in percentiles:
        threshold = np.percentile(ptm_values, pct)
        
        # Filter to nodes with PTM <= threshold
        nodes_to_include = [n for n in ego.nodes() if ego_ptm[n] <= threshold]
        
        # Always include center node
        if node not in nodes_to_include:
            nodes_to_include.append(node)
        
        # Compute Betti WITH center node
        if len(nodes_to_include) < 2:
            betti_with = [1] + [0] * max_dim
        else:
            subgraph = ego.subgraph(nodes_to_include)
            betti_with = compute_betti_clique(subgraph, max_dim=max_dim)
        
        # Store Betti numbers
        for d in range(max_dim + 1):
            features[f'H{d}_ptm{pct}'] = betti_with[d]
            betti_series[f'H{d}'].append(betti_with[d])
        
        # Compute Betti WITHOUT center node (perturbation)
        if node in nodes_to_include and len(nodes_to_include) > 2:
            nodes_without = [n for n in nodes_to_include if n != node]
            subgraph_without = ego.subgraph(nodes_without)
            betti_without = compute_betti_clique(subgraph_without, max_dim=max_dim)
            
            for d in range(1, max_dim + 1):
                delta = betti_with[d] - betti_without[d]
                features[f'delta_H{d}_ptm{pct}'] = delta
                delta_series[f'delta_H{d}'].append(delta)
        else:
            for d in range(1, max_dim + 1):
                features[f'delta_H{d}_ptm{pct}'] = 0
                delta_series[f'delta_H{d}'].append(0)
    
    # Compute slopes and ranges
    x = np.arange(len(percentiles))
    
    for dim in range(max_dim + 1):
        series = betti_series[f'H{dim}']
        if len(series) > 1:
            features[f'H{dim}_ptm_slope'] = np.polyfit(x, series, 1)[0]
            features[f'H{dim}_ptm_range'] = max(series) - min(series)
        else:
            features[f'H{dim}_ptm_slope'] = 0
            features[f'H{dim}_ptm_range'] = 0
    
    for dim in range(1, max_dim + 1):
        series = delta_series[f'delta_H{dim}']
        if len(series) > 1:
            features[f'delta_H{dim}_ptm_slope'] = np.polyfit(x, series, 1)[0]
            features[f'delta_H{dim}_ptm_range'] = max(series) - min(series)
        else:
            features[f'delta_H{dim}_ptm_slope'] = 0
            features[f'delta_H{dim}_ptm_range'] = 0
    
    features['time_seconds'] = time.time() - start_time
    
    return features


# ============================================================================
# BATCH PROCESSING (for running directly)
# ============================================================================

def load_human_interactome(filepath='data/BIOGRID-HUMAN-5.0.251.tab3.txt'):
    """Load the full human protein interaction network."""
    print("Loading human interactome...")
    df = pd.read_csv(filepath, sep='\t', low_memory=False)
    
    physical = df[df['Experimental System Type'] == 'physical']
    
    G = nx.Graph()
    for _, row in physical.iterrows():
        try:
            a = int(row['Entrez Gene Interactor A'])
            b = int(row['Entrez Gene Interactor B'])
            if a != b:
                G.add_edge(a, b)
        except:
            continue
    
    lcc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc).copy()
    
    print(f"  Nodes: {G_lcc.number_of_nodes():,}")
    print(f"  Edges: {G_lcc.number_of_edges():,}")
    
    return G_lcc


def load_ad_genes(filepath='data/BIOGRID-PROJECT-alzheimers_disease_project-GENES-5.0.250.projectindex.txt'):
    """Load AD gene list."""
    df = pd.read_csv(filepath, sep='\t')
    return set(df['ENTREZ GENE ID'].values)


def run_on_ad_genes(output_file='computed_data/tda_bifiltration_ad_genes.csv'):
    """Run bifiltration analysis on AD genes."""
    print("="*60)
    print("BIFILTRATION TDA - AD GENES")
    print("="*60)
    
    G = load_human_interactome()
    ptm_counts = load_ptm_counts()
    ad_genes = load_ad_genes()
    
    genes_in_network = [g for g in ad_genes if g in G]
    print(f"\nAD genes in network: {len(genes_in_network)}/{len(ad_genes)}")
    
    results = []
    for gene in tqdm(genes_in_network, desc="Processing AD genes"):
        features = compute_bifiltration_features(G, gene, ptm_counts)
        if features:
            features['is_ad'] = True
            results.append(features)
    
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
    
    return df


def main():
    """Run on AD genes when called directly."""
    run_on_ad_genes()


if __name__ == "__main__":
    main()

