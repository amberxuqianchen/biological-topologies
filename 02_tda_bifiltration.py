#!/usr/bin/env python3
"""
Efficient TDA Analysis for Alzheimer's Disease Networks
Fast implementation using PTM vs Drug Interaction parameters
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import TDA packages
try:
    import ripser
    from ripser import ripser as ripser_compute
    RIPSER_AVAILABLE = True
    print("✓ Ripser available - using for persistence computation")
except ImportError:
    RIPSER_AVAILABLE = False
    print("⚠️  Ripser not available")

class EfficientTDAAnalyzer:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.genes_df = None
        self.interactions_df = None
        self.chemicals_df = None
        self.network = None
        self.lcc = None
        self.bifiltration_values = {}
        self.persistence_diagrams = None
        
    def load_data(self):
        """Load all BIOGRID AD project data"""
        print("Loading BIOGRID Alzheimer's Disease Project data...")
        
        # Load genes
        genes_file = f"{self.data_dir}/BIOGRID-PROJECT-alzheimers_disease_project-GENES-5.0.250.projectindex.txt"
        self.genes_df = pd.read_csv(genes_file, sep='\t', header=0)
        print(f"✓ Loaded {len(self.genes_df)} genes")
        
        # Load interactions
        interactions_file = f"{self.data_dir}/BIOGRID-PROJECT-alzheimers_disease_project-INTERACTIONS-5.0.250.tab3.txt"
        self.interactions_df = pd.read_csv(interactions_file, sep='\t', header=0)
        print(f"✓ Loaded {len(self.interactions_df)} interactions")
        
        # Load chemical interactions
        chemicals_file = f"{self.data_dir}/BIOGRID-PROJECT-alzheimers_disease_project-CHEMICALS-5.0.250.chemtab.txt"
        self.chemicals_df = pd.read_csv(chemicals_file, sep='\t', header=0)
        print(f"✓ Loaded {len(self.chemicals_df)} chemical interactions")
        
    def build_network(self):
        """Build PPI network and extract LCC"""
        print("Building PPI network...")
        
        # Create network from interactions
        self.network = nx.Graph()
        
        # Add edges (ensure integer node IDs)
        for _, interaction in self.interactions_df.iterrows():
            try:
                gene_a = int(interaction['Entrez Gene Interactor A'])
                gene_b = int(interaction['Entrez Gene Interactor B']) 
                self.network.add_edge(gene_a, gene_b)
            except (ValueError, TypeError):
                # Skip non-integer gene IDs
                continue
        
        print(f"✓ Network: {self.network.number_of_nodes()} nodes, {self.network.number_of_edges()} edges")
        
        # Extract largest connected component
        connected_components = list(nx.connected_components(self.network))
        largest_cc = max(connected_components, key=len)
        self.lcc = self.network.subgraph(largest_cc).copy()
        
        print(f"✓ LCC: {self.lcc.number_of_nodes()} nodes, {self.lcc.number_of_edges()} edges")
        
    def compute_bifiltration_parameters(self):
        """Compute PTM vs Chemical interaction parameters"""
        print("Computing bifiltration parameters...")
        print("  Parameter 1: PTM (Post-Translational Modifications)")  
        print("  Parameter 2: Chemical Interaction Count")
        
        # Gene information mapping
        gene_info = {}
        for _, gene in self.genes_df.iterrows():
            entrez_id = gene['ENTREZ GENE ID']
            gene_info[entrez_id] = {
                'ptm_count': gene['PTM COUNT'],
                'chemical_count': gene['CHEMICAL INTERACTION COUNT']
            }
        
        # Get max values from curated genes only
        curated_genes = set(self.genes_df['ENTREZ GENE ID'])
        curated_ptm = [gene_info[g]['ptm_count'] for g in curated_genes if g in gene_info]
        curated_chemical = [gene_info[g]['chemical_count'] for g in curated_genes if g in gene_info]
        
        max_ptm = max(curated_ptm) if curated_ptm else 1
        max_chemical = max(curated_chemical) if curated_chemical else 1
        
        print(f"  Max PTM count (curated): {max_ptm}, Max chemical count (curated): {max_chemical}")
        print(f"  Curated genes: {len(curated_genes)}, Network nodes: {len(self.lcc.nodes())}")
        
        # Assign parameters to all network nodes
        for node in self.lcc.nodes():
            if node in curated_genes:
                # Use actual values for curated genes
                info = gene_info.get(node, {})
                ptm_param = info.get('ptm_count', 0) / max_ptm
                chemical_param = info.get('chemical_count', 0) / max_chemical
            else:
                # Assign minimal values to non-curated genes
                ptm_param = 0.0
                chemical_param = 0.0
            
            self.bifiltration_values[node] = (ptm_param, chemical_param)
        
        print(f"✓ Computed parameters for {len(self.bifiltration_values)} nodes")
        
    def _compute_drug_scores(self):
        """Simple drug interaction scoring"""
        drug_scores = {}
        
        # Count drug interactions per gene
        if len(self.chemicals_df) > 0:
            drug_counts = self.chemicals_df.groupby('Entrez Gene ID').size().to_dict()
        else:
            drug_counts = {}
        
        # Use chemical interaction count as fallback
        for _, gene in self.genes_df.iterrows():
            entrez_id = gene['ENTREZ GENE ID']
            drug_scores[entrez_id] = drug_counts.get(entrez_id, gene['CHEMICAL INTERACTION COUNT'])
        
        return drug_scores
    
    def run_tda_analysis(self, sample_size=1000):
        """Run efficient TDA analysis using Ripser"""
        print(f"Running TDA analysis...")
        
        if not RIPSER_AVAILABLE:
            print("⚠️  Ripser not available - skipping TDA")
            return None
        
        # Sample nodes for efficiency (only from nodes with parameters)
        valid_nodes = [n for n in self.lcc.nodes() if n in self.bifiltration_values]
        if len(valid_nodes) > sample_size:
            print(f"  Sampling {sample_size} nodes from {len(valid_nodes)} valid nodes")
            sampled_nodes = np.random.choice(valid_nodes, sample_size, replace=False)
        else:
            sampled_nodes = valid_nodes
            print(f"  Using all {len(valid_nodes)} valid nodes")
        
        # Create point cloud in parameter space
        points = np.array([self.bifiltration_values[node] for node in sampled_nodes])
        print(f"  Point cloud: {len(points)} points in 2D parameter space")
        
        # Compute persistence with Ripser
        print("  Computing persistent homology...")
        result = ripser_compute(points, maxdim=1, thresh=1.0)
        
        # Organize results
        persistence_diagrams = {}
        for dim in range(len(result['dgms'])):
            key = f'H{dim}'
            persistence_diagrams[key] = [tuple(interval) for interval in result['dgms'][dim]]
        
        # Summary statistics
        summary = self._compute_summary(persistence_diagrams)
        
        self.persistence_diagrams = {
            'persistence_intervals': persistence_diagrams,
            'summary': summary,
            'method': 'Ripser',
            'sample_size': len(sampled_nodes)
        }
        
        print(f"  ✓ Found {len(persistence_diagrams.get('H0', []))} H0, {len(persistence_diagrams.get('H1', []))} H1 features")
        return self.persistence_diagrams
    
    def _compute_summary(self, diagrams):
        """Compute summary statistics"""
        summary = {}
        for dim, intervals in diagrams.items():
            finite_intervals = [(b, d) for b, d in intervals if d != float('inf')]
            if finite_intervals:
                lifetimes = [d - b for b, d in finite_intervals]
                summary[dim] = {
                    'count': len(intervals),
                    'finite_count': len(finite_intervals),
                    'max_lifetime': max(lifetimes) if lifetimes else 0,
                    'mean_lifetime': np.mean(lifetimes) if lifetimes else 0
                }
            else:
                summary[dim] = {'count': 0, 'finite_count': 0, 'max_lifetime': 0, 'mean_lifetime': 0}
        return summary
    
    def visualize_results(self):
        """Create visualizations"""
        print("Creating visualizations...")
        
        # Parameter space plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract parameters
        nodes = list(self.lcc.nodes())
        ptm_vals = [self.bifiltration_values[n][0] for n in nodes]
        chemical_vals = [self.bifiltration_values[n][1] for n in nodes]
        
        # Scatter plot
        axes[0].scatter(ptm_vals, chemical_vals, alpha=0.6, s=20)
        axes[0].set_xlabel('PTM Density')
        axes[0].set_ylabel('Chemical Interaction Count')
        axes[0].set_title('TDA Parameter Space: PTM vs Chemical Interactions')
        axes[0].grid(True, alpha=0.3)
        
        # Persistence diagram
        if self.persistence_diagrams and 'persistence_intervals' in self.persistence_diagrams:
            intervals = self.persistence_diagrams['persistence_intervals']
            if 'H1' in intervals and len(intervals['H1']) > 0:
                births = [p[0] for p in intervals['H1'] if p[1] != float('inf')]
                deaths = [p[1] for p in intervals['H1'] if p[1] != float('inf')]
                
                if births and deaths:
                    axes[1].scatter(births, deaths, alpha=0.6, c='blue')
                    max_val = max(max(births), max(deaths))
                    axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
                    axes[1].set_xlabel('Birth')
                    axes[1].set_ylabel('Death')
                    axes[1].set_title('H1 Persistence Diagram')
                    axes[1].grid(True, alpha=0.3)
                else:
                    axes[1].text(0.5, 0.5, 'No finite H1 features', ha='center', va='center', transform=axes[1].transAxes)
            else:
                axes[1].text(0.5, 0.5, 'No H1 features found', ha='center', va='center', transform=axes[1].transAxes)
        else:
            axes[1].text(0.5, 0.5, 'No persistence data', ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def print_results(self):
        """Print analysis results"""
        print("\n" + "="*60)
        print("TDA ANALYSIS RESULTS")
        print("="*60)
        
        print(f"📊 Network Statistics:")
        print(f"  • Total genes: {len(self.genes_df)}")
        print(f"  • Network nodes: {self.network.number_of_nodes()}")
        print(f"  • LCC nodes: {self.lcc.number_of_nodes()}")
        print(f"  • Network edges: {self.network.number_of_edges()}")
        
        if self.persistence_diagrams:
            print(f"\n🔍 TDA Results ({self.persistence_diagrams['method']}):")
            summary = self.persistence_diagrams['summary']
            for dim, stats in summary.items():
                print(f"  • {dim}: {stats['count']} features ({stats['finite_count']} finite)")
                if stats['finite_count'] > 0:
                    print(f"    Max lifetime: {stats['max_lifetime']:.3f}")
                    print(f"    Mean lifetime: {stats['mean_lifetime']:.3f}")
        
        # Parameter distribution
        ptm_vals = [v[0] for v in self.bifiltration_values.values()]
        chemical_vals = [v[1] for v in self.bifiltration_values.values()]
        
        print(f"\n📈 Parameter Distribution:")
        print(f"  • PTM: mean={np.mean(ptm_vals):.3f}, std={np.std(ptm_vals):.3f}")
        print(f"  • Chemical: mean={np.mean(chemical_vals):.3f}, std={np.std(chemical_vals):.3f}")
        
        # High-value genes
        high_ptm = [n for n, (p, c) in self.bifiltration_values.items() if p >= 0.8]
        high_chemical = [n for n, (p, c) in self.bifiltration_values.items() if c >= 0.8]
        high_both = [n for n, (p, c) in self.bifiltration_values.items() if p >= 0.8 and c >= 0.8]
        
        print(f"\n🎯 Notable Genes:")
        print(f"  • High PTM (≥0.8): {len(high_ptm)} genes")
        print(f"  • High Chemical (≥0.8): {len(high_chemical)} genes")  
        print(f"  • High Both (≥0.8): {len(high_both)} genes")
        
        print("="*60)
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "="*60)
        print("EFFICIENT TDA ANALYSIS FOR ALZHEIMER'S DISEASE")
        print("="*60)
        
        # Run pipeline
        self.load_data()
        self.build_network()
        self.compute_bifiltration_parameters()
        self.run_tda_analysis()
        self.print_results()
        self.visualize_results()
        
        return self.persistence_diagrams

def main():
    """Main execution function"""
    analyzer = EfficientTDAAnalyzer()
    results = analyzer.run_complete_analysis()
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main()