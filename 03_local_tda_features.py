#!/usr/bin/env python3
"""
Local Network TDA Feature Extraction for AD Gene Classification
Computes per-node topological features from local network neighborhoods
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import TDA packages
RIPSER_AVAILABLE = False
GUDHI_AVAILABLE = False
FLAGSER_AVAILABLE = False

try:
    from ripser import ripser as ripser_compute
    RIPSER_AVAILABLE = True
    print("✓ Ripser available")
except ImportError:
    print("⚠️  Ripser not available - install with: pip install ripser")

try:
    import gudhi
    GUDHI_AVAILABLE = True
    print("✓ GUDHI available")
except ImportError:
    print("⚠️  GUDHI not available - install with: pip install gudhi")

try:
    import pyflagser
    FLAGSER_AVAILABLE = True
    print("✓ Flagser available (optimized for graphs!)")
except ImportError:
    print("⚠️  Flagser not available - install with: pip install pyflagser")

class LocalTDAFeatureExtractor:
    """Extract TDA features from local network neighborhoods"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.genes_df = None
        self.interactions_df = None
        self.network = None
        self.lcc = None
        self.ad_genes = set()
        self.target_nodes = []
        self.features_df = None
        
    def load_data(self):
        """Load BioGRID data"""
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        
        # Load genes
        genes_file = f"{self.data_dir}/BIOGRID-PROJECT-alzheimers_disease_project-GENES-5.0.250.projectindex.txt"
        self.genes_df = pd.read_csv(genes_file, sep='\t', header=0)
        print(f"✓ Loaded {len(self.genes_df)} AD-associated genes")
        
        # Store AD genes
        self.ad_genes = set(self.genes_df['ENTREZ GENE ID'].values)
        
        # Load interactions
        interactions_file = f"{self.data_dir}/BIOGRID-PROJECT-alzheimers_disease_project-INTERACTIONS-5.0.250.tab3.txt"
        self.interactions_df = pd.read_csv(interactions_file, sep='\t', header=0)
        print(f"✓ Loaded {len(self.interactions_df)} interactions")
        
    def build_network(self):
        """Build PPI network and extract LCC"""
        print("\n" + "="*80)
        print("BUILDING NETWORK")
        print("="*80)
        
        # Create network from interactions
        self.network = nx.Graph()
        
        # Add edges (skip self-loops)
        for _, interaction in self.interactions_df.iterrows():
            try:
                gene_a = int(interaction['Entrez Gene Interactor A'])
                gene_b = int(interaction['Entrez Gene Interactor B'])
                
                if gene_a != gene_b:  # Skip self-loops
                    self.network.add_edge(gene_a, gene_b)
            except (ValueError, TypeError):
                continue
        
        print(f"✓ Network: {self.network.number_of_nodes():,} nodes, {self.network.number_of_edges():,} edges")
        
        # Extract largest connected component
        connected_components = list(nx.connected_components(self.network))
        largest_cc = max(connected_components, key=len)
        self.lcc = self.network.subgraph(largest_cc).copy()
        
        print(f"✓ LCC: {self.lcc.number_of_nodes():,} nodes ({self.lcc.number_of_nodes()/self.network.number_of_nodes()*100:.1f}%)")
        print(f"✓ LCC edges: {self.lcc.number_of_edges():,}")
        
    def sample_target_nodes(self, neg_pos_ratio=2, exclude_neurological=True):
        """
        Create balanced dataset of AD and non-AD genes
        
        Parameters:
        -----------
        neg_pos_ratio : int
            Ratio of negative to positive samples (default 2:1)
        exclude_neurological : bool
            Whether to exclude genes with neurological keywords from negative class
        """
        print("\n" + "="*80)
        print("SAMPLING TARGET NODES")
        print("="*80)
        
        # Positive class: AD genes in LCC
        ad_genes_in_lcc = self.ad_genes.intersection(set(self.lcc.nodes()))
        positive_nodes = list(ad_genes_in_lcc)
        
        print(f"✓ Positive class (AD genes): {len(positive_nodes)}")
        
        # Negative class: Non-AD genes in LCC
        all_lcc_nodes = set(self.lcc.nodes())
        non_ad_candidates = list(all_lcc_nodes - self.ad_genes)
        
        # Optionally filter out neurological-related genes
        if exclude_neurological:
            # This would require gene names/descriptions - for now we'll skip this filtering
            # In practice, you'd load gene annotations and filter by keywords
            print(f"  Note: Neurological keyword filtering not implemented yet")
        
        # Sample negative class
        n_negative = len(positive_nodes) * neg_pos_ratio
        n_negative = min(n_negative, len(non_ad_candidates))
        
        # np.random.seed(42)  # For reproducibility
        negative_nodes = np.random.choice(non_ad_candidates, size=n_negative, replace=False).tolist()
        
        print(f"✓ Negative class (non-AD genes): {n_negative}")
        print(f"✓ Class ratio (neg:pos): {neg_pos_ratio}:1")
        
        # Combine target nodes
        self.target_nodes = positive_nodes + negative_nodes
        
        # Create labels
        self.labels = np.array([1] * len(positive_nodes) + [0] * len(negative_nodes))
        
        print(f"✓ Total target nodes: {len(self.target_nodes)}")
        print(f"✓ Class distribution: {sum(self.labels)} positive, {len(self.labels) - sum(self.labels)} negative")
        
    def extract_ego_graph(self, node, radius=2):
        """
        Extract local neighborhood (ego graph) around a node
        
        Parameters:
        -----------
        node : int
            Target node
        radius : int
            Neighborhood radius (number of hops)
            
        Returns:
        --------
        ego_graph : nx.Graph
            Subgraph containing node and its neighbors within radius
        """
        return nx.ego_graph(self.lcc, node, radius=radius)
    
    def compute_distance_matrix(self, ego_graph, sparse=True, max_distance=3):
        """
        Compute shortest-path distance matrix for ego graph.
        
        Can return either dense or sparse format. Sparse is MUCH faster
        for Ripser when graph has large diameter.
        
        Parameters:
        -----------
        ego_graph : nx.Graph
            Local neighborhood graph
        sparse : bool
            If True, return sparse format (list of edges with distances)
            If False, return dense NxN matrix
        max_distance : int
            For sparse mode, only include pairs within this distance
            (Ripser will treat larger distances as infinity)
            
        Returns:
        --------
        If sparse=False:
            distance_matrix : np.ndarray
            node_list : list
        If sparse=True:
            sparse_distances : np.ndarray (M x 3) of [i, j, distance]
            n_nodes : int
        """
        node_list = list(ego_graph.nodes())
        n = len(node_list)
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        if sparse:
            # Sparse format: only store edges within max_distance
            # This is MUCH faster for Ripser!
            edges = []
            
            # Compute distances only up to max_distance (more efficient)
            for source in ego_graph.nodes():
                # BFS to find nodes within max_distance
                distances = nx.single_source_shortest_path_length(
                    ego_graph, source, cutoff=max_distance
                )
                i = node_to_idx[source]
                for target, dist in distances.items():
                    j = node_to_idx[target]
                    if i < j:  # Only upper triangle
                        edges.append([i, j, dist])
            
            return np.array(edges) if edges else np.array([]).reshape(0, 3), n
        
        else:
            # Dense format (original behavior)
            distances = dict(nx.all_pairs_shortest_path_length(ego_graph))
            distance_matrix = np.zeros((n, n))
            for i, node_i in enumerate(node_list):
                for j, node_j in enumerate(node_list):
                    if i != j:
                        distance_matrix[i, j] = distances[node_i].get(node_j, np.inf)
            
            return distance_matrix, node_list
    
    def compute_tda_features_flagser(self, graph, max_dim=2):
        """
        Compute topological features using Flagser (optimized for flag complexes).
        Uses DEGREE as filtration - nodes/edges added in order of degree.
        
        Parameters:
        -----------
        graph : nx.Graph
            The ego graph to analyze
        max_dim : int
            Maximum homology dimension
            
        Returns:
        --------
        features : dict
            Dictionary of TDA-derived features
        """
        if not FLAGSER_AVAILABLE:
            return None
        
        try:
            # Create adjacency matrix with DEGREE-based filtration
            # Lower degree = appears earlier in filtration
            nodes = list(graph.nodes())
            n = len(nodes)
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            # Get degrees for filtration values
            degrees = dict(graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            
            # Build adjacency matrix with filtration values
            # Edge filtration = max degree of endpoints (normalized)
            adjacency = np.zeros((n, n))
            for u, v in graph.edges():
                i, j = node_to_idx[u], node_to_idx[v]
                # Filtration value = max degree of endpoints (normalized to [0,1])
                filt = max(degrees[u], degrees[v]) / max_degree
                adjacency[i, j] = filt
                adjacency[j, i] = filt
            
            # Compute persistent homology with Flagser
            result = pyflagser.flagser_weighted(
                adjacency,
                max_dimension=max_dim,
                directed=False
            )
            
            features = {}
            
            # Extract features from each dimension
            for dim in range(max_dim + 1):
                if dim < len(result['dgms']):
                    dgm = np.array(result['dgms'][dim])
                    
                    if len(dgm) > 0:
                        # Filter finite intervals
                        finite_mask = dgm[:, 1] != np.inf
                        finite_dgm = dgm[finite_mask]
                        
                        if len(finite_dgm) > 0:
                            lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
                            valid_mask = (lifetimes >= 0) & np.isfinite(lifetimes)
                            valid_lifetimes = lifetimes[valid_mask]
                            
                            if len(valid_lifetimes) > 0:
                                features[f'H{dim}_count'] = len(valid_lifetimes)
                                features[f'H{dim}_total_persistence'] = float(np.sum(valid_lifetimes))
                                features[f'H{dim}_max_persistence'] = float(np.max(valid_lifetimes))
                                features[f'H{dim}_mean_persistence'] = float(np.mean(valid_lifetimes))
                                features[f'H{dim}_median_persistence'] = float(np.median(valid_lifetimes))
                                features[f'H{dim}_std_persistence'] = float(np.std(valid_lifetimes))
                                
                                if np.sum(valid_lifetimes) > 0:
                                    probs = valid_lifetimes / np.sum(valid_lifetimes)
                                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                                    features[f'H{dim}_entropy'] = float(entropy)
                                else:
                                    features[f'H{dim}_entropy'] = 0.0
                            else:
                                self._set_zero_features(features, dim)
                        else:
                            self._set_zero_features(features, dim)
                        
                        features[f'H{dim}_infinite_count'] = int(np.sum(~finite_mask))
                        features[f'H{dim}_invalid_count'] = 0
                    else:
                        self._set_zero_features(features, dim)
                        features[f'H{dim}_infinite_count'] = 0
                        features[f'H{dim}_invalid_count'] = 0
                else:
                    self._set_zero_features(features, dim)
                    features[f'H{dim}_infinite_count'] = 0
                    features[f'H{dim}_invalid_count'] = 0
            
            # Also add Betti numbers (useful for unweighted analysis)
            if 'betti' in result:
                for dim, betti in enumerate(result['betti']):
                    features[f'betti_{dim}'] = betti
            
            return features
            
        except Exception as e:
            print(f"  Warning: Flagser failed - {str(e)}")
            return None
    
    def _set_zero_features(self, features, dim):
        """Helper to set zero values for a dimension"""
        features[f'H{dim}_count'] = 0
        features[f'H{dim}_total_persistence'] = 0.0
        features[f'H{dim}_max_persistence'] = 0.0
        features[f'H{dim}_mean_persistence'] = 0.0
        features[f'H{dim}_median_persistence'] = 0.0
        features[f'H{dim}_std_persistence'] = 0.0
        features[f'H{dim}_entropy'] = 0.0
    
    def compute_tda_features_from_graph(self, graph, center_node, max_dim=2):
        """
        Compute persistent homology directly from graph using clique complex
        with HOP DISTANCE FILTRATION (distance from center node).
        
        This is faster than Ripser for graphs and gives meaningful persistence!
        
        Parameters:
        -----------
        graph : nx.Graph
            The ego graph to analyze
        center_node : int
            The central node (used for distance-based filtration)
        max_dim : int
            Maximum homology dimension
            
        Returns:
        --------
        features : dict
            Dictionary of TDA-derived features
        """
        if not GUDHI_AVAILABLE:
            return None  # Fall back to Ripser
        
        try:
            # Get hop distances from center node
            hop_distances = nx.single_source_shortest_path_length(graph, center_node)
            
            # Build simplex tree with FILTRATION based on max hop distance
            simplex_tree = gudhi.SimplexTree()
            
            # Add vertices with filtration = their hop distance
            for node in graph.nodes():
                filt_value = float(hop_distances.get(node, 0))
                simplex_tree.insert([node], filtration=filt_value)
            
            # Add edges with filtration = max of endpoint distances
            for u, v in graph.edges():
                filt_value = float(max(hop_distances.get(u, 0), hop_distances.get(v, 0)))
                simplex_tree.insert([u, v], filtration=filt_value)
            
            # Expand to cliques (triangles, tetrahedra, etc.)
            # Filtration of higher simplices = max of vertex filtrations
            simplex_tree.expansion(max_dim + 1)
            
            # Compute persistence
            simplex_tree.compute_persistence()
            
            features = {}
            
            # Extract persistence pairs for each dimension
            for dim in range(max_dim + 1):
                pairs = simplex_tree.persistence_intervals_in_dimension(dim)
                
                if len(pairs) > 0:
                    pairs = np.array(pairs)
                    # Filter finite intervals
                    finite_mask = pairs[:, 1] != np.inf
                    finite_pairs = pairs[finite_mask]
                    
                    if len(finite_pairs) > 0:
                        lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
                        # Filter valid lifetimes
                        valid_mask = (lifetimes >= 0) & (lifetimes < 1e10)
                        valid_lifetimes = lifetimes[valid_mask]
                        
                        if len(valid_lifetimes) > 0:
                            features[f'H{dim}_count'] = len(valid_lifetimes)
                            features[f'H{dim}_total_persistence'] = float(np.sum(valid_lifetimes))
                            features[f'H{dim}_max_persistence'] = float(np.max(valid_lifetimes))
                            features[f'H{dim}_mean_persistence'] = float(np.mean(valid_lifetimes))
                            features[f'H{dim}_median_persistence'] = float(np.median(valid_lifetimes))
                            features[f'H{dim}_std_persistence'] = float(np.std(valid_lifetimes))
                            
                            # Entropy
                            total = np.sum(valid_lifetimes)
                            if total > 0:
                                probs = valid_lifetimes / total
                                entropy = -np.sum(probs * np.log(probs + 1e-10))
                                features[f'H{dim}_entropy'] = float(entropy)
                            else:
                                features[f'H{dim}_entropy'] = 0.0
                        else:
                            features[f'H{dim}_count'] = 0
                            features[f'H{dim}_total_persistence'] = 0.0
                            features[f'H{dim}_max_persistence'] = 0.0
                            features[f'H{dim}_mean_persistence'] = 0.0
                            features[f'H{dim}_median_persistence'] = 0.0
                            features[f'H{dim}_std_persistence'] = 0.0
                            features[f'H{dim}_entropy'] = 0.0
                    else:
                        features[f'H{dim}_count'] = 0
                        features[f'H{dim}_total_persistence'] = 0.0
                        features[f'H{dim}_max_persistence'] = 0.0
                        features[f'H{dim}_mean_persistence'] = 0.0
                        features[f'H{dim}_median_persistence'] = 0.0
                        features[f'H{dim}_std_persistence'] = 0.0
                        features[f'H{dim}_entropy'] = 0.0
                    
                    # Infinite features (important for H0 - the main connected component)
                    infinite_count = int(np.sum(~finite_mask))
                    features[f'H{dim}_infinite_count'] = infinite_count
                    features[f'H{dim}_invalid_count'] = 0
                else:
                    features[f'H{dim}_count'] = 0
                    features[f'H{dim}_total_persistence'] = 0.0
                    features[f'H{dim}_max_persistence'] = 0.0
                    features[f'H{dim}_mean_persistence'] = 0.0
                    features[f'H{dim}_median_persistence'] = 0.0
                    features[f'H{dim}_std_persistence'] = 0.0
                    features[f'H{dim}_entropy'] = 0.0
                    features[f'H{dim}_infinite_count'] = 0
                    features[f'H{dim}_invalid_count'] = 0
            
            return features
            
        except Exception as e:
            print(f"  Warning: GUDHI computation failed - {str(e)}")
            return None  # Fall back to Ripser
    
    def compute_tda_features_sparse(self, sparse_edges, n_nodes, max_dim=2):
        """
        Compute persistent homology using Ripser's SPARSE mode.
        Much faster for graphs than dense distance matrices!
        
        Parameters:
        -----------
        sparse_edges : np.ndarray
            Mx3 array of [i, j, distance] for each edge
        n_nodes : int
            Number of nodes
        max_dim : int
            Maximum homology dimension
            
        Returns:
        --------
        features : dict
        """
        if not RIPSER_AVAILABLE:
            return self._get_empty_tda_features(max_dim=max_dim)
        
        try:
            from scipy.sparse import csr_matrix
            
            if len(sparse_edges) == 0:
                return self._get_empty_tda_features(max_dim=max_dim)
            
            # Build sparse distance matrix
            rows = sparse_edges[:, 0].astype(int)
            cols = sparse_edges[:, 1].astype(int)
            data = sparse_edges[:, 2]
            
            # Create symmetric sparse matrix
            sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
            sparse_matrix = sparse_matrix + sparse_matrix.T  # Make symmetric
            
            # Run Ripser with sparse input
            result = ripser_compute(
                sparse_matrix,
                maxdim=max_dim,
                distance_matrix=True,
                thresh=5.0  # Only compute up to distance 5
            )
            
            return self._extract_persistence_features(result, max_dim)
            
        except Exception as e:
            print(f"  Warning: Sparse Ripser failed - {str(e)}")
            return self._get_empty_tda_features(max_dim=max_dim)
    
    def _extract_persistence_features(self, result, max_dim):
        """Extract features from Ripser result"""
        features = {}
        
        for dim in range(max_dim + 1):
            dgm = result['dgms'][dim]
            finite_dgm = dgm[dgm[:, 1] != np.inf]
            
            if len(finite_dgm) > 0:
                lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
                valid_mask = np.isfinite(lifetimes) & (lifetimes >= 0) & (lifetimes < 1e10)
                valid_lifetimes = lifetimes[valid_mask]
                
                features[f'H{dim}_invalid_count'] = int(len(lifetimes) - len(valid_lifetimes))
                
                if len(valid_lifetimes) > 0:
                    features[f'H{dim}_count'] = len(valid_lifetimes)
                    features[f'H{dim}_total_persistence'] = float(np.sum(valid_lifetimes))
                    features[f'H{dim}_max_persistence'] = float(np.max(valid_lifetimes))
                    features[f'H{dim}_mean_persistence'] = float(np.mean(valid_lifetimes))
                    features[f'H{dim}_median_persistence'] = float(np.median(valid_lifetimes))
                    features[f'H{dim}_std_persistence'] = float(np.std(valid_lifetimes))
                    
                    if np.sum(valid_lifetimes) > 0:
                        probs = valid_lifetimes / np.sum(valid_lifetimes)
                        features[f'H{dim}_entropy'] = float(-np.sum(probs * np.log(probs + 1e-10)))
                    else:
                        features[f'H{dim}_entropy'] = 0.0
                else:
                    self._set_zero_features(features, dim)
            else:
                self._set_zero_features(features, dim)
                features[f'H{dim}_invalid_count'] = 0
            
            features[f'H{dim}_infinite_count'] = int(len(dgm) - len(finite_dgm))
        
        return features
    
    def compute_tda_features(self, distance_matrix, max_dim=1):
        """
        Compute persistent homology on distance matrix
        
        Parameters:
        -----------
        distance_matrix : np.ndarray
            Distance matrix for Vietoris-Rips complex
        max_dim : int
            Maximum homology dimension to compute
            
        Returns:
        --------
        features : dict
            Dictionary of TDA-derived features
        """
        if not RIPSER_AVAILABLE:
            return self._get_empty_tda_features(max_dim=max_dim)
        
        try:
            # Run Ripser with distance matrix
            result = ripser_compute(
                distance_matrix, 
                maxdim=max_dim, 
                distance_matrix=True,
                thresh=10.0  # Limit filtration value
            )
            
            features = {}
            
            # Process each homology dimension
            for dim in range(max_dim + 1):
                dgm = result['dgms'][dim]
                
                # Filter out infinite persistence (for H0)
                finite_dgm = dgm[dgm[:, 1] != np.inf]
                
                # Compute lifetimes
                if len(finite_dgm) > 0:
                    lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
                    
                    # Filter out invalid lifetimes (numerical issues)
                    valid_mask = np.isfinite(lifetimes) & (lifetimes >= 0) & (lifetimes < 1e10)
                    valid_lifetimes = lifetimes[valid_mask]
                    n_invalid = len(lifetimes) - len(valid_lifetimes)
                    
                    # Track invalid count for diagnostics
                    features[f'H{dim}_invalid_count'] = n_invalid
                    
                    if len(valid_lifetimes) > 0:
                        features[f'H{dim}_count'] = len(valid_lifetimes)
                        features[f'H{dim}_total_persistence'] = float(np.sum(valid_lifetimes))
                        features[f'H{dim}_max_persistence'] = float(np.max(valid_lifetimes))
                        features[f'H{dim}_mean_persistence'] = float(np.mean(valid_lifetimes))
                        features[f'H{dim}_median_persistence'] = float(np.median(valid_lifetimes))
                        features[f'H{dim}_std_persistence'] = float(np.std(valid_lifetimes))
                        
                        # Barcode entropy (measure of complexity)
                        total = np.sum(valid_lifetimes)
                        if total > 0:
                            probs = valid_lifetimes / total
                            entropy = -np.sum(probs * np.log(probs + 1e-10))
                            features[f'H{dim}_entropy'] = float(entropy)
                        else:
                            features[f'H{dim}_entropy'] = 0.0
                    else:
                        # All lifetimes were invalid - use NaN to signal missing data
                        features[f'H{dim}_count'] = 0
                        features[f'H{dim}_total_persistence'] = np.nan
                        features[f'H{dim}_max_persistence'] = np.nan
                        features[f'H{dim}_mean_persistence'] = np.nan
                        features[f'H{dim}_median_persistence'] = np.nan
                        features[f'H{dim}_std_persistence'] = np.nan
                        features[f'H{dim}_entropy'] = np.nan
                else:
                    # No features found
                    features[f'H{dim}_count'] = 0
                    features[f'H{dim}_total_persistence'] = 0.0
                    features[f'H{dim}_max_persistence'] = 0.0
                    features[f'H{dim}_mean_persistence'] = 0.0
                    features[f'H{dim}_median_persistence'] = 0.0
                    features[f'H{dim}_std_persistence'] = 0.0
                    features[f'H{dim}_entropy'] = 0.0
                    features[f'H{dim}_invalid_count'] = 0
                
                # Include count of infinite features (components that never die)
                infinite_count = len(dgm) - len(finite_dgm)
                features[f'H{dim}_infinite_count'] = infinite_count
            
            return features
            
        except Exception as e:
            print(f"  Warning: TDA computation failed - {str(e)}")
            return self._get_empty_tda_features(max_dim=max_dim)
    
    def _get_empty_tda_features(self, max_dim=2):
        """Return empty feature dict when TDA computation fails"""
        features = {}
        for dim in range(max_dim + 1):
            features[f'H{dim}_count'] = 0
            features[f'H{dim}_total_persistence'] = np.nan
            features[f'H{dim}_max_persistence'] = np.nan
            features[f'H{dim}_mean_persistence'] = np.nan
            features[f'H{dim}_median_persistence'] = np.nan
            features[f'H{dim}_std_persistence'] = np.nan
            features[f'H{dim}_entropy'] = np.nan
            features[f'H{dim}_infinite_count'] = 0
            features[f'H{dim}_invalid_count'] = 0
        return features
    
    def compute_network_features(self, node):
        """
        Compute traditional graph-based features for a node
        
        Parameters:
        -----------
        node : int
            Target node
            
        Returns:
        --------
        features : dict
            Dictionary of network features
        """
        features = {}
        
        # Basic features
        features['degree'] = self.lcc.degree(node)
        features['clustering_coefficient'] = nx.clustering(self.lcc, node)
        
        # Ego graph features
        ego = nx.ego_graph(self.lcc, node, radius=1)
        features['ego_size_1hop'] = len(ego.nodes())
        features['ego_edges_1hop'] = len(ego.edges())
        features['ego_density_1hop'] = nx.density(ego) if len(ego.nodes()) > 1 else 0.0
        
        # 2-hop ego features
        ego2 = nx.ego_graph(self.lcc, node, radius=2)
        features['ego_size_2hop'] = len(ego2.nodes())
        features['ego_edges_2hop'] = len(ego2.edges())
        features['ego_density_2hop'] = nx.density(ego2) if len(ego2.nodes()) > 1 else 0.0
        
        return features
    
    def extract_all_features(self, radius=2, max_dim=2, max_ego_size=300):
        """
        Extract TDA and network features for all target nodes
        
        Parameters:
        -----------
        radius : int
            Ego graph radius for TDA computation
        max_dim : int
            Maximum homology dimension
        max_ego_size : int
            Maximum ego graph size for TDA (larger graphs are subsampled)
        """
        print("\n" + "="*80)
        print("EXTRACTING FEATURES")
        print("="*80)
        print(f"Parameters:")
        print(f"  • Ego graph radius: {radius}")
        print(f"  • Max homology dimension: H{max_dim}")
        print(f"  • Max ego graph size for TDA: {max_ego_size}")
        print(f"  • Target nodes: {len(self.target_nodes)}")
        
        all_features = []
        ego_sizes = []  # Track ego graph sizes for diagnostics
        import time
        start_time = time.time()
        
        for i, node in enumerate(self.target_nodes):
            # Progress with ETA
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(self.target_nodes) - i - 1) / rate if rate > 0 else 0
            print(f"  Processing {i+1}/{len(self.target_nodes)} "
                    f"({(i+1)/len(self.target_nodes)*100:.1f}%) "
                    f"- ETA: {remaining/60:.1f} min    ", end='\r')
            
            node_features = {'node_id': node, 'is_ad': int(node in self.ad_genes)}
            
            try:
                # Extract ego graph
                ego_graph = self.extract_ego_graph(node, radius=radius)
                original_ego_size = len(ego_graph.nodes())
                ego_sizes.append(original_ego_size)
                
                # Skip if ego graph too small
                if original_ego_size < 2:
                    # Add placeholder features
                    node_features.update(self._get_empty_tda_features(max_dim=max_dim))
                    node_features.update(self.compute_network_features(node))
                    node_features['ego_original_size'] = original_ego_size
                    node_features['ego_was_subsampled'] = False
                    all_features.append(node_features)
                    continue
                
                # Subsample large ego graphs for speed AND numerical stability
                was_subsampled = False
                if original_ego_size > max_ego_size:
                    was_subsampled = True
                    # Use BFS-based sampling to ensure connected subgraph
                    # This is crucial for valid H0 computation!
                    
                    # Start BFS from central node
                    visited = set([node])
                    queue = [node]
                    
                    while len(visited) < max_ego_size and queue:
                        current = queue.pop(0)
                        for neighbor in ego_graph.neighbors(current):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                                if len(visited) >= max_ego_size:
                                    break
                    
                    # Create connected subgraph
                    ego_graph = ego_graph.subgraph(list(visited)).copy()
                
                # Store diagnostic info
                node_features['ego_original_size'] = original_ego_size
                node_features['ego_was_subsampled'] = was_subsampled
                
                # Use Ripser with SPARSE distance matrix
                # This is the correct "growing balls" Vietoris-Rips approach
                # Sparse mode is much faster than dense!
                USE_SPARSE = True  # Set to False to compare with dense mode
                
                if USE_SPARSE:
                    sparse_edges, n_nodes = self.compute_distance_matrix(
                        ego_graph, sparse=True, max_distance=3
                    )
                    tda_features = self.compute_tda_features_sparse(sparse_edges, n_nodes, max_dim=max_dim)
                else:
                    # Dense mode (slower but exact)
                    distance_matrix, node_list = self.compute_distance_matrix(ego_graph, sparse=False)
                    tda_features = self.compute_tda_features(distance_matrix, max_dim=max_dim)
                
                node_features.update(tda_features)
                
                # Compute network features
                network_features = self.compute_network_features(node)
                node_features.update(network_features)
                
                all_features.append(node_features)
                
            except Exception as e:
                print(f"\n  Warning: Failed to process node {node} - {str(e)}")
                # Add placeholder features
                node_features.update(self._get_empty_tda_features(max_dim=max_dim))
                node_features.update(self.compute_network_features(node))
                node_features['ego_original_size'] = 0
                node_features['ego_was_subsampled'] = False
                all_features.append(node_features)
        
        print(f"\n✓ Extracted features for {len(all_features)} nodes")
        
        # Print ego graph size statistics
        if ego_sizes:
            ego_sizes = np.array(ego_sizes)
            subsampled_count = sum(1 for s in ego_sizes if s > max_ego_size)
            print(f"\n📐 Ego Graph Size Statistics:")
            print(f"  • Min size: {np.min(ego_sizes)}")
            print(f"  • Max size: {np.max(ego_sizes)}")
            print(f"  • Mean size: {np.mean(ego_sizes):.1f}")
            print(f"  • Median size: {np.median(ego_sizes):.1f}")
            print(f"  • Subsampled (>{max_ego_size}): {subsampled_count} ({subsampled_count/len(ego_sizes)*100:.1f}%)")
        
        # Convert to DataFrame
        self.features_df = pd.DataFrame(all_features)
        
        # Print feature summary
        print(f"\n📊 Feature Summary:")
        print(f"  • Total features: {len(self.features_df.columns) - 2}")  # Exclude node_id and is_ad
        tda_cols = [col for col in self.features_df.columns if col.startswith('H')]
        network_cols = [col for col in self.features_df.columns if not col.startswith('H') and col not in ['node_id', 'is_ad']]
        print(f"  • TDA features: {len(tda_cols)}")
        print(f"  • Network features: {len(network_cols)}")
        
        return self.features_df
    
    def save_features(self, output_file='computed_data/ad_network_features.csv'):
        """Save features to CSV"""
        if self.features_df is not None:
            self.features_df.to_csv(output_file, index=False)
            print(f"\n✓ Features saved to: {output_file}")
        else:
            print("⚠️  No features to save. Run extract_all_features() first.")
    
    def print_feature_statistics(self):
        """Print summary statistics of extracted features"""
        if self.features_df is None:
            print("⚠️  No features extracted yet.")
            return
        
        print("\n" + "="*80)
        print("FEATURE STATISTICS")
        print("="*80)
        
        # Class balance
        ad_count = self.features_df['is_ad'].sum()
        non_ad_count = len(self.features_df) - ad_count
        print(f"\n📊 Class Distribution:")
        print(f"  • AD genes: {ad_count} ({ad_count/len(self.features_df)*100:.1f}%)")
        print(f"  • Non-AD genes: {non_ad_count} ({non_ad_count/len(self.features_df)*100:.1f}%)")
        
        # TDA feature statistics
        print(f"\n🔍 TDA Feature Statistics (Mean ± Std):")
        tda_cols = [col for col in self.features_df.columns if col.startswith('H')]
        
        for col in sorted(tda_cols):
            mean = self.features_df[col].mean()
            std = self.features_df[col].std()
            print(f"  • {col:30s}: {mean:8.3f} ± {std:7.3f}")
        
        # Network feature statistics
        print(f"\n🕸️ Network Feature Statistics (Mean ± Std):")
        network_cols = [col for col in self.features_df.columns 
                       if not col.startswith('H') and col not in ['node_id', 'is_ad']]
        
        for col in sorted(network_cols):
            mean = self.features_df[col].mean()
            std = self.features_df[col].std()
            print(f"  • {col:30s}: {mean:8.3f} ± {std:7.3f}")
        
        # Compare AD vs non-AD
        print(f"\n🎯 Feature Comparison (AD vs Non-AD):")
        print(f"{'Feature':<30} {'AD Mean':>10} {'Non-AD Mean':>12} {'Ratio':>8}")
        print("-" * 65)
        
        important_features = ['degree', 'H1_count', 'H1_max_persistence', 'clustering_coefficient']
        for col in important_features:
            if col in self.features_df.columns:
                ad_mean = self.features_df[self.features_df['is_ad'] == 1][col].mean()
                non_ad_mean = self.features_df[self.features_df['is_ad'] == 0][col].mean()
                ratio = ad_mean / (non_ad_mean + 1e-10)
                print(f"{col:<30} {ad_mean:10.3f} {non_ad_mean:12.3f} {ratio:8.2f}x")
    
    def run_complete_pipeline(self, radius=2, neg_pos_ratio=2, max_ego_size=300, output_file='computed_data/ad_network_features.csv'):
        """
        Run complete feature extraction pipeline
        
        Parameters:
        -----------
        radius : int
            Ego graph radius
        neg_pos_ratio : int
            Negative to positive class ratio
        max_ego_size : int
            Maximum ego graph size for TDA (larger graphs are subsampled for speed)
        output_file : str
            Output CSV file path
        """
        print("\n" + "="*80)
        print("LOCAL TDA FEATURE EXTRACTION PIPELINE")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Build network
        self.build_network()
        
        # Sample target nodes
        self.sample_target_nodes(neg_pos_ratio=neg_pos_ratio)
        
        # Extract features
        self.extract_all_features(radius=radius, max_ego_size=max_ego_size)
        
        # Print statistics
        self.print_feature_statistics()
        
        # Save features
        self.save_features(output_file=output_file)
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE!")
        print("="*80)
        
        return self.features_df


def main(quick_test=False, n_test_nodes=50):
    """Main execution function
    
    Parameters:
    -----------
    quick_test : bool
        If True, only run on n_test_nodes nodes for testing
    n_test_nodes : int
        Number of nodes to test with if quick_test=True
    """
    extractor = LocalTDAFeatureExtractor(data_dir="data")
    
    if quick_test:
        # Quick test mode - just a few nodes
        print(f"\n🧪 QUICK TEST MODE: {n_test_nodes} nodes only\n")
        extractor.load_data()
        extractor.build_network()
        extractor.sample_target_nodes(neg_pos_ratio=2)
        extractor.target_nodes = extractor.target_nodes[:n_test_nodes]
        features_df = extractor.extract_all_features(radius=2, max_dim=2, max_ego_size=300)
        extractor.print_feature_statistics()
        print("\n✅ Quick test complete! (features not saved)")
    else:
        # Full pipeline
        features_df = extractor.run_complete_pipeline(
            radius=2,                    # 2-hop neighborhood
            neg_pos_ratio=2,            # 2:1 negative to positive ratio
            output_file='computed_data/ad_network_features.csv'
        )
    
    return extractor, features_df


if __name__ == "__main__":
    import sys
    
    # Check for quick test flag
    quick_test = '--quick' in sys.argv or '-q' in sys.argv
    
    extractor, features_df = main(quick_test=quick_test, n_test_nodes=50)

