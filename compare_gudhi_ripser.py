#!/usr/bin/env python3
"""
Compare GUDHI vs Ripser for computing Betti numbers on ego graphs.

This script:
1. Loads a sample gene from the AD network
2. Extracts its ego graph
3. Computes H0, H1, H2 using both GUDHI and Ripser
4. Compares results for correctness
5. Times both methods
"""

import numpy as np
import networkx as nx
import pandas as pd
import time
from scipy.sparse.csgraph import shortest_path

# Import both libraries
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("GUDHI not available. Install with: pip install gudhi")

try:
    from ripser import ripser as ripser_compute
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    print("Ripser not available. Install with: pip install ripser")


def load_ad_network():
    """Load the AD protein interaction network."""
    interactions = pd.read_csv(
        'data/BIOGRID-PROJECT-alzheimers_disease_project-INTERACTIONS-5.0.250.tab3.txt',
        sep='\t', low_memory=False
    )
    physical = interactions[interactions['Experimental System Type'] == 'physical']
    
    G = nx.Graph()
    for _, row in physical.iterrows():
        try:
            a = int(row['Entrez Gene Interactor A'])
            b = int(row['Entrez Gene Interactor B'])
            if a != b:
                G.add_edge(a, b)
        except:
            continue
    
    # Get LCC
    lcc = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc).copy()


def extract_ego_graph(G, node, radius=2, max_size=200):
    """Extract ego graph around a node."""
    ego = nx.ego_graph(G, node, radius=radius)
    
    if len(ego.nodes()) > max_size:
        # BFS subsampling
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
    
    return ego


def compute_distance_matrix(G, max_dist=3):
    """Compute shortest path distance matrix."""
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Build adjacency matrix
    adj = np.zeros((n, n))
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj[i, j] = adj[j, i] = 1
    
    # Compute shortest paths
    dist = shortest_path(adj, method='D', unweighted=True)
    
    # Cap at max_dist (for Rips complex threshold)
    dist = np.minimum(dist, max_dist + 1)
    
    return dist, nodes


def compute_betti_ripser(dist_matrix, max_dim=2, threshold=3):
    """Compute Betti numbers using Ripser."""
    result = ripser_compute(
        dist_matrix,
        maxdim=max_dim,
        distance_matrix=True,
        thresh=threshold
    )
    
    # Count features alive at the threshold
    # Ripser convention: essential (infinite) features have death <= birth (typically death=0)
    betti = []
    for dim in range(max_dim + 1):
        dgm = result['dgms'][dim]
        count = 0
        for b, d in dgm:
            # Essential features: death <= birth (always alive)
            if d <= b:
                count += 1
            # Non-essential: born before threshold and dies after threshold
            elif b <= threshold and d > threshold:
                count += 1
        betti.append(count)
    
    return betti, result


def compute_betti_gudhi(dist_matrix, max_dim=2, threshold=3):
    """Compute Betti numbers using GUDHI."""
    # GUDHI RipsComplex expects the distance matrix
    rips = gudhi.RipsComplex(
        distance_matrix=dist_matrix.tolist(),
        max_edge_length=threshold
    )
    
    # Create simplex tree (need max_dimension = max_dim + 1 for H_max_dim)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dim + 1)
    
    # Must compute persistence before getting Betti numbers
    simplex_tree.compute_persistence()
    
    # Get Betti numbers
    betti = simplex_tree.betti_numbers()
    
    # Pad with zeros if needed
    while len(betti) <= max_dim:
        betti.append(0)
    
    return betti[:max_dim + 1], simplex_tree


def run_comparison(ego_graph, threshold=3, max_dim=2, n_trials=5):
    """Run comparison between GUDHI and Ripser."""
    
    print(f"\nEgo graph: {ego_graph.number_of_nodes()} nodes, {ego_graph.number_of_edges()} edges")
    print(f"Threshold: {threshold}, Max dimension: {max_dim}")
    print("=" * 60)
    
    # Compute distance matrix (shared)
    print("\nComputing distance matrix...")
    t0 = time.time()
    dist_matrix, nodes = compute_distance_matrix(ego_graph, max_dist=threshold)
    dist_time = time.time() - t0
    print(f"Distance matrix: {dist_time:.4f}s")
    
    # RIPSER
    if RIPSER_AVAILABLE:
        print("\n--- RIPSER ---")
        ripser_times = []
        for i in range(n_trials):
            t0 = time.time()
            betti_ripser, result = compute_betti_ripser(dist_matrix, max_dim, threshold)
            ripser_times.append(time.time() - t0)
        
        print(f"Betti numbers: {betti_ripser}")
        print(f"Time: {np.mean(ripser_times):.4f}s ± {np.std(ripser_times):.4f}s (n={n_trials})")
        
        # Show persistence diagram info
        for dim in range(max_dim + 1):
            dgm = result['dgms'][dim]
            n_features = len(dgm)
            print(f"  H{dim}: {n_features} total features in diagram")
    else:
        betti_ripser = None
    
    # GUDHI
    if GUDHI_AVAILABLE:
        print("\n--- GUDHI ---")
        gudhi_times = []
        for i in range(n_trials):
            t0 = time.time()
            betti_gudhi, simplex_tree = compute_betti_gudhi(dist_matrix, max_dim, threshold)
            gudhi_times.append(time.time() - t0)
        
        print(f"Betti numbers: {betti_gudhi}")
        print(f"Time: {np.mean(gudhi_times):.4f}s ± {np.std(gudhi_times):.4f}s (n={n_trials})")
        print(f"  Simplex tree: {simplex_tree.num_simplices()} simplices")
    else:
        betti_gudhi = None
    
    # Comparison
    print("\n--- COMPARISON ---")
    if betti_ripser is not None and betti_gudhi is not None:
        match = betti_ripser == betti_gudhi
        if match:
            print("✅ Results MATCH!")
        else:
            print("❌ Results DIFFER!")
            print(f"  Ripser: {betti_ripser}")
            print(f"  GUDHI:  {betti_gudhi}")
        
        speedup = np.mean(ripser_times) / np.mean(gudhi_times)
        if speedup > 1:
            print(f"\n⚡ GUDHI is {speedup:.2f}x faster")
        else:
            print(f"\n⚡ Ripser is {1/speedup:.2f}x faster")
    
    return betti_ripser, betti_gudhi


def main():
    print("=" * 60)
    print("GUDHI vs RIPSER COMPARISON")
    print("=" * 60)
    
    if not GUDHI_AVAILABLE or not RIPSER_AVAILABLE:
        print("\nBoth GUDHI and Ripser are required for comparison.")
        return
    
    # Load network
    print("\nLoading AD network...")
    G = load_ad_network()
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Get a high-degree node for testing
    degrees = dict(G.degree())
    test_nodes = sorted(degrees.keys(), key=lambda x: -degrees[x])[:3]
    
    for test_node in test_nodes:
        print("\n" + "=" * 60)
        print(f"TEST NODE: {test_node} (degree={degrees[test_node]})")
        print("=" * 60)
        
        # Extract ego graph
        ego = extract_ego_graph(G, test_node, radius=2, max_size=200)
        
        # Test at different thresholds
        for threshold in [2, 3]:
            print(f"\n{'='*30}")
            print(f"THRESHOLD = {threshold}")
            print(f"{'='*30}")
            run_comparison(ego, threshold=threshold, max_dim=2, n_trials=5)
    
    # Also test perturbation (removing center node)
    print("\n" + "=" * 60)
    print("PERTURBATION TEST (removing center node)")
    print("=" * 60)
    
    test_node = test_nodes[0]
    ego = extract_ego_graph(G, test_node, radius=2, max_size=200)
    
    print(f"\nWith center node {test_node}:")
    dist_with, _ = compute_distance_matrix(ego)
    betti_with_ripser, _ = compute_betti_ripser(dist_with, max_dim=2, threshold=3)
    betti_with_gudhi, _ = compute_betti_gudhi(dist_with, max_dim=2, threshold=3)
    print(f"  Ripser: {betti_with_ripser}")
    print(f"  GUDHI:  {betti_with_gudhi}")
    
    # Remove center node
    ego_without = ego.copy()
    ego_without.remove_node(test_node)
    
    print(f"\nWithout center node {test_node}:")
    if ego_without.number_of_nodes() > 0:
        dist_without, _ = compute_distance_matrix(ego_without)
        betti_without_ripser, _ = compute_betti_ripser(dist_without, max_dim=2, threshold=3)
        betti_without_gudhi, _ = compute_betti_gudhi(dist_without, max_dim=2, threshold=3)
        print(f"  Ripser: {betti_without_ripser}")
        print(f"  GUDHI:  {betti_without_gudhi}")
        
        print(f"\nDelta (with - without):")
        delta_ripser = [betti_with_ripser[i] - betti_without_ripser[i] for i in range(3)]
        delta_gudhi = [betti_with_gudhi[i] - betti_without_gudhi[i] for i in range(3)]
        print(f"  Ripser: ΔH0={delta_ripser[0]}, ΔH1={delta_ripser[1]}, ΔH2={delta_ripser[2]}")
        print(f"  GUDHI:  ΔH0={delta_gudhi[0]}, ΔH1={delta_gudhi[1]}, ΔH2={delta_gudhi[2]}")
    else:
        print("  (Graph is empty after removal)")


if __name__ == "__main__":
    main()

