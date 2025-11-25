#!/usr/bin/env python3
"""
Bifiltration Analysis for AD Networks
Two-parameter persistence using graph distance + biological properties (PTM count)

Approaches implemented:
1. RIVET (if available): True 2-parameter persistent homology
2. Separate filtrations: Compare distance-based vs PTM-based topology
3. Combined filtration: Mix both parameters into single filtration
4. Fibered barcodes: Persistence along slices of 2D parameter space
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Check for RIVET
RIVET_AVAILABLE = False
try:
    import rivet
    RIVET_AVAILABLE = True
    print("✓ RIVET available (true 2-parameter persistence!)")
except ImportError:
    try:
        # Alternative import name
        import pyrivet as rivet
        RIVET_AVAILABLE = True
        print("✓ pyrivet available (true 2-parameter persistence!)")
    except ImportError:
        print("⚠️  RIVET not available - install with: pip install rivet_python")
        print("   Will use approximation methods instead")

try:
    from ripser import ripser as ripser_compute
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    print("⚠️  Ripser not available")


class BifiltrationAnalyzer:
    """
    Analyze AD networks using two-parameter filtration:
    - Parameter 1: Graph distance (shortest path)
    - Parameter 2: PTM density (biological property)
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.genes_df = None
        self.interactions_df = None
        self.ptm_df = None
        self.network = None
        self.lcc = None
        self.ptm_counts = {}
        
    def load_data(self):
        """Load all necessary data"""
        print("="*80)
        print("LOADING DATA FOR BIFILTRATION")
        print("="*80)
        
        # Load genes
        genes_file = f"{self.data_dir}/BIOGRID-PROJECT-alzheimers_disease_project-GENES-5.0.250.projectindex.txt"
        self.genes_df = pd.read_csv(genes_file, sep='\t', header=0)
        print(f"✓ Loaded {len(self.genes_df)} genes")
        
        # Load PTM counts from genes file
        for _, row in self.genes_df.iterrows():
            self.ptm_counts[row['ENTREZ GENE ID']] = row['PTM COUNT']
        
        # Load interactions
        interactions_file = f"{self.data_dir}/BIOGRID-PROJECT-alzheimers_disease_project-INTERACTIONS-5.0.250.tab3.txt"
        self.interactions_df = pd.read_csv(interactions_file, sep='\t', header=0)
        print(f"✓ Loaded {len(self.interactions_df)} interactions")
        
        # Try to load PTM file for more detailed info
        try:
            ptm_file = f"{self.data_dir}/BIOGRID-PROJECT-alzheimers_disease_project-PTM-5.0.250.ptmtab.txt"
            self.ptm_df = pd.read_csv(ptm_file, sep='\t', header=0)
            print(f"✓ Loaded {len(self.ptm_df)} PTM records")
        except:
            print("  Note: Detailed PTM file not found, using PTM counts from genes file")
            
    def build_network(self):
        """Build PPI network"""
        print("\n" + "="*80)
        print("BUILDING NETWORK")
        print("="*80)
        
        self.network = nx.Graph()
        
        for _, interaction in self.interactions_df.iterrows():
            try:
                gene_a = int(interaction['Entrez Gene Interactor A'])
                gene_b = int(interaction['Entrez Gene Interactor B'])
                if gene_a != gene_b:
                    self.network.add_edge(gene_a, gene_b)
            except:
                continue
        
        # Get LCC
        largest_cc = max(nx.connected_components(self.network), key=len)
        self.lcc = self.network.subgraph(largest_cc).copy()
        
        print(f"✓ Network: {self.lcc.number_of_nodes()} nodes, {self.lcc.number_of_edges()} edges")
        
    def get_node_ptm(self, node):
        """Get PTM count for a node (normalized)"""
        ptm = self.ptm_counts.get(node, 0)
        return ptm
    
    def extract_ego_graph(self, node, radius=2, max_size=200):
        """Extract and subsample ego graph"""
        ego = nx.ego_graph(self.lcc, node, radius=radius)
        
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
    
    # =========================================================================
    # APPROACH 0: RIVET (True 2-Parameter Persistence) - PREFERRED IF AVAILABLE
    # =========================================================================
    
    def compute_rivet_bifiltration(self, ego_graph, max_dim=1):
        """
        Compute true 2-parameter persistent homology using RIVET.
        
        Parameters:
        -----------
        ego_graph : nx.Graph
            The ego graph to analyze
        max_dim : int
            Maximum homology dimension
            
        Returns:
        --------
        result : dict
            Contains fibered barcodes, Betti numbers, etc.
        """
        if not RIVET_AVAILABLE:
            return None
        
        try:
            nodes = list(ego_graph.nodes())
            n = len(nodes)
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            # Compute distances (parameter 1)
            distances = dict(nx.all_pairs_shortest_path_length(ego_graph))
            max_dist = max(max(d.values()) for d in distances.values())
            
            # Get PTM values (parameter 2)
            ptm_values = {node: self.get_node_ptm(node) for node in nodes}
            max_ptm = max(max(ptm_values.values()), 1)
            
            # Build bifiltration data for RIVET
            # Format: list of simplices with (simplex, filtration_value_1, filtration_value_2)
            simplices = []
            
            # Add vertices with their PTM as second parameter
            for node in nodes:
                idx = node_to_idx[node]
                ptm_norm = ptm_values[node] / max_ptm
                simplices.append(([idx], 0.0, ptm_norm))  # Vertex appears at distance 0
            
            # Add edges
            for u, v in ego_graph.edges():
                i, j = node_to_idx[u], node_to_idx[v]
                dist = distances[u].get(v, 1) / max_dist  # Should be 1/max_dist for direct edges
                ptm_max = max(ptm_values[u], ptm_values[v]) / max_ptm
                simplices.append(([i, j], dist, ptm_max))
            
            # Add triangles (2-simplices)
            for node in nodes:
                neighbors = list(ego_graph.neighbors(node))
                idx_node = node_to_idx[node]
                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i+1:]:
                        if ego_graph.has_edge(n1, n2):
                            # Triangle found
                            idx1, idx2 = node_to_idx[n1], node_to_idx[n2]
                            triangle = sorted([idx_node, idx1, idx2])
                            
                            # Filtration: max distance and max PTM among vertices
                            all_nodes = [nodes[idx_node], n1, n2]
                            dist_val = max(
                                distances[all_nodes[0]].get(all_nodes[1], 1),
                                distances[all_nodes[0]].get(all_nodes[2], 1),
                                distances[all_nodes[1]].get(all_nodes[2], 1)
                            ) / max_dist
                            ptm_val = max(ptm_values[n] for n in all_nodes) / max_ptm
                            
                            simplices.append((triangle, dist_val, ptm_val))
            
            # Remove duplicates
            seen = set()
            unique_simplices = []
            for simplex, f1, f2 in simplices:
                key = tuple(simplex)
                if key not in seen:
                    seen.add(key)
                    unique_simplices.append((simplex, f1, f2))
            
            # Call RIVET
            # Note: RIVET API varies by version, this is a general approach
            result = rivet.compute_bifiltration(
                unique_simplices,
                x_label="Distance",
                y_label="PTM",
                homology_dimension=max_dim
            )
            
            return result
            
        except Exception as e:
            print(f"  Warning: RIVET computation failed - {str(e)}")
            return None
    
    def extract_rivet_features(self, rivet_result, prefix='rivet_'):
        """Extract features from RIVET result"""
        features = {}
        
        if rivet_result is None:
            # Return empty features
            for dim in [0, 1]:
                features[f'{prefix}H{dim}_fibered_count'] = 0
                features[f'{prefix}H{dim}_total_rank'] = 0.0
            return features
        
        try:
            # Extract fibered barcodes along different lines
            # Line 1: Fix PTM = 0.5, vary distance
            fb1 = rivet_result.get_fibered_barcode(angle=0, offset=0.5)
            features[f'{prefix}H1_fiber_horizontal'] = len(fb1) if fb1 else 0
            
            # Line 2: Fix distance = 0.5, vary PTM
            fb2 = rivet_result.get_fibered_barcode(angle=90, offset=0.5)
            features[f'{prefix}H1_fiber_vertical'] = len(fb2) if fb2 else 0
            
            # Line 3: Diagonal
            fb3 = rivet_result.get_fibered_barcode(angle=45, offset=0.0)
            features[f'{prefix}H1_fiber_diagonal'] = len(fb3) if fb3 else 0
            
            # Betti numbers at grid points
            for x in [0.25, 0.5, 0.75]:
                for y in [0.25, 0.5, 0.75]:
                    betti = rivet_result.get_betti(x, y, dim=1)
                    features[f'{prefix}betti1_at_{int(x*100)}_{int(y*100)}'] = betti
            
        except Exception as e:
            print(f"  Warning: Feature extraction failed - {str(e)}")
        
        return features
    
    # =========================================================================
    # APPROACH 1: Separate Filtrations
    # =========================================================================
    
    def compute_distance_filtration(self, ego_graph, max_dim=1):
        """
        Standard Vietoris-Rips filtration using graph distance.
        (Same as what we do in single-parameter analysis)
        """
        nodes = list(ego_graph.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Compute distance matrix
        distances = dict(nx.all_pairs_shortest_path_length(ego_graph))
        dist_matrix = np.zeros((n, n))
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i != j:
                    dist_matrix[i, j] = distances[node_i].get(node_j, np.inf)
        
        # Run Ripser
        result = ripser_compute(dist_matrix, maxdim=max_dim, distance_matrix=True, thresh=5.0)
        return result, 'distance'
    
    def compute_ptm_filtration(self, ego_graph, max_dim=1):
        """
        Filtration using PTM density.
        Nodes with LOW PTM appear first, HIGH PTM appear last.
        """
        nodes = list(ego_graph.nodes())
        n = len(nodes)
        
        # Get PTM values for each node
        ptm_values = np.array([self.get_node_ptm(node) for node in nodes])
        max_ptm = max(ptm_values.max(), 1)  # Avoid division by zero
        
        # Create "distance" matrix based on PTM difference
        # Higher PTM difference = larger "distance"
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Filtration: edge appears when BOTH nodes have been added
                    # Node added at filtration value = its PTM (normalized)
                    filt_i = ptm_values[i] / max_ptm
                    filt_j = ptm_values[j] / max_ptm
                    # Edge appears at max of two endpoint filtration values
                    dist_matrix[i, j] = max(filt_i, filt_j)
        
        # Run Ripser
        result = ripser_compute(dist_matrix, maxdim=max_dim, distance_matrix=True, thresh=1.5)
        return result, 'ptm'
    
    # =========================================================================
    # APPROACH 2: Combined Filtration
    # =========================================================================
    
    def compute_combined_filtration(self, ego_graph, alpha=0.5, max_dim=1):
        """
        Combined filtration mixing distance and PTM.
        f(edge) = alpha * distance + (1-alpha) * PTM_score
        
        Parameters:
        -----------
        alpha : float
            Weight for distance (0 = PTM only, 1 = distance only, 0.5 = equal)
        """
        nodes = list(ego_graph.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Get distances
        distances = dict(nx.all_pairs_shortest_path_length(ego_graph))
        max_dist = max(max(d.values()) for d in distances.values())
        
        # Get PTM values
        ptm_values = np.array([self.get_node_ptm(node) for node in nodes])
        max_ptm = max(ptm_values.max(), 1)
        
        # Combined distance matrix
        dist_matrix = np.zeros((n, n))
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i != j:
                    # Normalize distance to [0, 1]
                    d = distances[node_i].get(node_j, max_dist) / max_dist
                    
                    # PTM component: max of endpoint PTMs (normalized)
                    p = max(ptm_values[i], ptm_values[j]) / max_ptm
                    
                    # Combined filtration value
                    dist_matrix[i, j] = alpha * d + (1 - alpha) * p
        
        # Run Ripser
        result = ripser_compute(dist_matrix, maxdim=max_dim, distance_matrix=True, thresh=2.0)
        return result, f'combined_alpha{alpha}'
    
    # =========================================================================
    # APPROACH 3: Fibered Barcodes (Slices through 2D space)
    # =========================================================================
    
    def compute_fibered_barcodes(self, ego_graph, ptm_thresholds=[0.0, 0.25, 0.5, 0.75, 1.0], max_dim=1):
        """
        Compute persistence along "fibers" - fixing PTM threshold, varying distance.
        
        For each PTM threshold:
        - Include only nodes with PTM <= threshold
        - Compute distance-based persistence on this subgraph
        
        This gives a family of barcodes parameterized by PTM threshold.
        """
        nodes = list(ego_graph.nodes())
        ptm_values = {node: self.get_node_ptm(node) for node in nodes}
        max_ptm = max(max(ptm_values.values()), 1)
        
        # Normalize PTM values
        normalized_ptm = {node: ptm / max_ptm for node, ptm in ptm_values.items()}
        
        fibered_results = {}
        
        for threshold in ptm_thresholds:
            # Get nodes with PTM <= threshold
            included_nodes = [n for n in nodes if normalized_ptm[n] <= threshold]
            
            if len(included_nodes) < 3:
                fibered_results[threshold] = None
                continue
            
            # Create subgraph
            subgraph = ego_graph.subgraph(included_nodes).copy()
            
            # Skip if disconnected
            if not nx.is_connected(subgraph):
                largest_cc = max(nx.connected_components(subgraph), key=len)
                subgraph = subgraph.subgraph(largest_cc).copy()
            
            if len(subgraph.nodes()) < 3:
                fibered_results[threshold] = None
                continue
            
            # Compute distance-based persistence on subgraph
            result, _ = self.compute_distance_filtration(subgraph, max_dim=max_dim)
            fibered_results[threshold] = result
        
        return fibered_results
    
    # =========================================================================
    # Feature Extraction
    # =========================================================================
    
    def extract_persistence_features(self, result, prefix=''):
        """Extract features from a persistence result"""
        features = {}
        
        if result is None:
            for dim in [0, 1]:
                features[f'{prefix}H{dim}_count'] = 0
                features[f'{prefix}H{dim}_total_persistence'] = 0.0
                features[f'{prefix}H{dim}_max_persistence'] = 0.0
            return features
        
        for dim in range(len(result['dgms'])):
            dgm = result['dgms'][dim]
            finite_dgm = dgm[dgm[:, 1] != np.inf]
            
            if len(finite_dgm) > 0:
                lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
                valid = lifetimes[np.isfinite(lifetimes) & (lifetimes >= 0)]
                
                if len(valid) > 0:
                    features[f'{prefix}H{dim}_count'] = len(valid)
                    features[f'{prefix}H{dim}_total_persistence'] = float(np.sum(valid))
                    features[f'{prefix}H{dim}_max_persistence'] = float(np.max(valid))
                    features[f'{prefix}H{dim}_mean_persistence'] = float(np.mean(valid))
                else:
                    features[f'{prefix}H{dim}_count'] = 0
                    features[f'{prefix}H{dim}_total_persistence'] = 0.0
                    features[f'{prefix}H{dim}_max_persistence'] = 0.0
                    features[f'{prefix}H{dim}_mean_persistence'] = 0.0
            else:
                features[f'{prefix}H{dim}_count'] = 0
                features[f'{prefix}H{dim}_total_persistence'] = 0.0
                features[f'{prefix}H{dim}_max_persistence'] = 0.0
                features[f'{prefix}H{dim}_mean_persistence'] = 0.0
        
        return features
    
    def analyze_node_bifiltration(self, node, radius=2, max_size=200, use_rivet=True):
        """
        Complete bifiltration analysis for a single node.
        Returns features from available approaches.
        
        Priority:
        1. RIVET (true 2-parameter persistence) if available
        2. Fallback methods (separate, combined, fibered)
        """
        # Extract ego graph
        ego = self.extract_ego_graph(node, radius=radius, max_size=max_size)
        
        if len(ego.nodes()) < 5:
            return None
        
        features = {'node_id': node}
        
        # =====================================================================
        # APPROACH 0: RIVET (if available and requested)
        # =====================================================================
        if use_rivet and RIVET_AVAILABLE:
            rivet_result = self.compute_rivet_bifiltration(ego)
            if rivet_result:
                features.update(self.extract_rivet_features(rivet_result))
                features['used_rivet'] = True
            else:
                features['used_rivet'] = False
        else:
            features['used_rivet'] = False
        
        # =====================================================================
        # FALLBACK APPROACHES (always compute for comparison)
        # =====================================================================
        
        # Approach 1: Separate filtrations
        dist_result, _ = self.compute_distance_filtration(ego)
        ptm_result, _ = self.compute_ptm_filtration(ego)
        
        features.update(self.extract_persistence_features(dist_result, prefix='dist_'))
        features.update(self.extract_persistence_features(ptm_result, prefix='ptm_'))
        
        # Approach 2: Combined filtrations with different alphas
        for alpha in [0.25, 0.5, 0.75]:
            combined_result, _ = self.compute_combined_filtration(ego, alpha=alpha)
            features.update(self.extract_persistence_features(
                combined_result, prefix=f'combined{int(alpha*100)}_'
            ))
        
        # Approach 3: Fibered barcodes (summary statistics)
        fibered = self.compute_fibered_barcodes(ego, ptm_thresholds=[0.25, 0.5, 0.75, 1.0])
        
        for threshold, result in fibered.items():
            t_str = f"{int(threshold*100)}"
            fiber_features = self.extract_persistence_features(result, prefix=f'fiber{t_str}_')
            features.update(fiber_features)
        
        return features
    
    def run_bifiltration_analysis(self, target_nodes, output_file='data/bifiltration_features.csv'):
        """Run bifiltration analysis on target nodes"""
        print("\n" + "="*80)
        print("BIFILTRATION ANALYSIS")
        print("="*80)
        print(f"Target nodes: {len(target_nodes)}")
        print("Approaches:")
        print("  1. Separate filtrations (distance vs PTM)")
        print("  2. Combined filtrations (α = 0.25, 0.5, 0.75)")
        print("  3. Fibered barcodes (PTM thresholds: 0.25, 0.5, 0.75, 1.0)")
        
        all_features = []
        
        for i, node in enumerate(target_nodes):
            if (i + 1) % 20 == 0:
                print(f"  Processing {i+1}/{len(target_nodes)}...", end='\r')
            
            features = self.analyze_node_bifiltration(node)
            if features:
                all_features.append(features)
        
        print(f"\n✓ Extracted bifiltration features for {len(all_features)} nodes")
        
        # Save to CSV
        df = pd.DataFrame(all_features)
        df.to_csv(output_file, index=False)
        print(f"✓ Saved to {output_file}")
        
        return df
    
    def visualize_bifiltration(self, node, save_path=None):
        """Visualize bifiltration results for a single node"""
        ego = self.extract_ego_graph(node, radius=2, max_size=200)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Distance filtration persistence diagram
        dist_result, _ = self.compute_distance_filtration(ego)
        ax = axes[0, 0]
        if len(dist_result['dgms'][1]) > 0:
            dgm = dist_result['dgms'][1]
            finite = dgm[dgm[:, 1] != np.inf]
            if len(finite) > 0:
                ax.scatter(finite[:, 0], finite[:, 1], alpha=0.6)
                max_val = max(finite[:, 0].max(), finite[:, 1].max())
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        ax.set_title('Distance Filtration (H1)')
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        
        # 2. PTM filtration persistence diagram
        ptm_result, _ = self.compute_ptm_filtration(ego)
        ax = axes[0, 1]
        if len(ptm_result['dgms'][1]) > 0:
            dgm = ptm_result['dgms'][1]
            finite = dgm[dgm[:, 1] != np.inf]
            if len(finite) > 0:
                ax.scatter(finite[:, 0], finite[:, 1], alpha=0.6, color='orange')
                max_val = max(finite[:, 0].max(), finite[:, 1].max())
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        ax.set_title('PTM Filtration (H1)')
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        
        # 3. Combined filtration (alpha=0.5)
        combined_result, _ = self.compute_combined_filtration(ego, alpha=0.5)
        ax = axes[0, 2]
        if len(combined_result['dgms'][1]) > 0:
            dgm = combined_result['dgms'][1]
            finite = dgm[dgm[:, 1] != np.inf]
            if len(finite) > 0:
                ax.scatter(finite[:, 0], finite[:, 1], alpha=0.6, color='green')
                max_val = max(finite[:, 0].max(), finite[:, 1].max())
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        ax.set_title('Combined Filtration α=0.5 (H1)')
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        
        # 4. Fibered barcodes - H1 count across thresholds
        fibered = self.compute_fibered_barcodes(ego, ptm_thresholds=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax = axes[1, 0]
        thresholds = []
        h1_counts = []
        for t, result in fibered.items():
            thresholds.append(t)
            if result and len(result['dgms'][1]) > 0:
                h1_counts.append(len(result['dgms'][1][result['dgms'][1][:, 1] != np.inf]))
            else:
                h1_counts.append(0)
        ax.plot(thresholds, h1_counts, 'o-', color='purple')
        ax.set_xlabel('PTM Threshold')
        ax.set_ylabel('H1 Count')
        ax.set_title('Fibered Barcodes: H1 vs PTM Threshold')
        
        # 5. Node PTM distribution in ego graph
        ax = axes[1, 1]
        ptms = [self.get_node_ptm(n) for n in ego.nodes()]
        ax.hist(ptms, bins=20, color='teal', alpha=0.7)
        ax.axvline(self.get_node_ptm(node), color='red', linestyle='--', label='Central node')
        ax.set_xlabel('PTM Count')
        ax.set_ylabel('Frequency')
        ax.set_title('PTM Distribution in Ego Graph')
        ax.legend()
        
        # 6. Summary text
        ax = axes[1, 2]
        ax.axis('off')
        summary = f"""
        Node: {node}
        Ego graph size: {len(ego.nodes())}
        Ego graph edges: {len(ego.edges())}
        Central node PTM: {self.get_node_ptm(node)}
        
        Distance filtration H1: {len(dist_result['dgms'][1])} features
        PTM filtration H1: {len(ptm_result['dgms'][1])} features
        Combined filtration H1: {len(combined_result['dgms'][1])} features
        """
        ax.text(0.1, 0.5, summary, fontsize=11, family='monospace',
                verticalalignment='center')
        ax.set_title('Summary')
        
        plt.suptitle(f'Bifiltration Analysis: Node {node}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")
        
        plt.show()
        return fig


def main():
    """Demo bifiltration analysis"""
    print("="*80)
    print("BIFILTRATION ANALYSIS FOR AD NETWORKS")
    print("="*80)
    
    analyzer = BifiltrationAnalyzer(data_dir="data")
    analyzer.load_data()
    analyzer.build_network()
    
    # Get some AD genes to analyze
    ad_genes = set(analyzer.genes_df['ENTREZ GENE ID'].values)
    ad_genes_in_network = list(ad_genes.intersection(set(analyzer.lcc.nodes())))[:10]
    
    print(f"\n🧪 Demo: Analyzing {len(ad_genes_in_network)} AD genes")
    
    # Visualize one node
    if ad_genes_in_network:
        print(f"\nVisualizing node {ad_genes_in_network[0]}...")
        analyzer.visualize_bifiltration(ad_genes_in_network[0])
    
    # Run full analysis on sample
    print("\nRunning full bifiltration analysis...")
    df = analyzer.run_bifiltration_analysis(
        ad_genes_in_network,
        output_file='data/bifiltration_features_demo.csv'
    )
    
    print(f"\n📊 Features extracted: {len(df.columns)}")
    print(df.head())
    
    return analyzer, df


if __name__ == "__main__":
    analyzer, df = main()

