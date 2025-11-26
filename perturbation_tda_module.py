#!/usr/bin/env python3
"""
Perturbation TDA Module

This module exports the core TDA functions for use by other scripts.
It wraps the functionality from 07_perturbation_tda.py.
"""

import numpy as np
import networkx as nx
import pandas as pd
import gudhi
import time
import os
from tqdm import tqdm


# ============================================================================
# DATA LOADING
# ============================================================================

def load_human_interactome(filepath='data/BIOGRID-HUMAN-5.0.251.tab3.txt'):
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


def compute_betti_clique(G, max_dim=3):
    """
    Compute Betti numbers using GUDHI clique complex.
    """
    if len(G.nodes()) < 2:
        return [1] + [0] * max_dim, 0
    
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
    
    return betti[:max_dim+1], st.num_simplices()


def compute_perturbation_features(G, node, radius=2, max_ego_size=300, max_dim=3):
    """
    Compute perturbation TDA features for a single node.
    """
    start_time = time.time()
    
    if node not in G:
        return None
    
    ego, original_size, was_subsampled = extract_ego_graph(
        G, node, radius=radius, max_size=max_ego_size
    )
    
    betti_with, n_simp_with = compute_betti_clique(ego, max_dim=max_dim)
    
    ego_without = ego.copy()
    ego_without.remove_node(node)
    
    if len(ego_without.nodes()) >= 2:
        betti_without, n_simp_without = compute_betti_clique(ego_without, max_dim=max_dim)
    else:
        betti_without = [0] * (max_dim + 1)
        n_simp_without = 0
    
    total_time = time.time() - start_time
    
    features = {
        'node_id': node,
        'degree': G.degree(node),
        'ego_original_size': original_size,
        'ego_actual_size': len(ego.nodes()),
        'ego_edges': ego.number_of_edges(),
        'was_subsampled': was_subsampled,
        'n_simplices': n_simp_with,
        'time_seconds': total_time,
        
        'H0_with': betti_with[0],
        'H1_with': betti_with[1],
        'H2_with': betti_with[2],
        'H3_with': betti_with[3] if max_dim >= 3 else 0,
        
        'H0_without': betti_without[0],
        'H1_without': betti_without[1],
        'H2_without': betti_without[2],
        'H3_without': betti_without[3] if max_dim >= 3 else 0,
        
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
    
    df = pd.DataFrame(results)
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
    
    return df

