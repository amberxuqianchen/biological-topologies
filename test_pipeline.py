#!/usr/bin/env python3
"""
Quick Test Script - Run with Small Sample
Use this to test the pipeline before running the full analysis
"""

import sys
import os

def test_imports():
    """Test if all required packages are available"""
    print("="*80)
    print("TESTING IMPORTS")
    print("="*80)
    
    required_packages = [
        'pandas', 'numpy', 'networkx', 'matplotlib', 
        'seaborn', 'sklearn', 'ripser'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT FOUND")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All packages available!")
    return True


def test_data_files():
    """Check if data files exist"""
    print("\n" + "="*80)
    print("CHECKING DATA FILES")
    print("="*80)
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found")
        return False
    
    required_files = [
        "BIOGRID-PROJECT-alzheimers_disease_project-GENES-5.0.250.projectindex.txt",
        "BIOGRID-PROJECT-alzheimers_disease_project-INTERACTIONS-5.0.250.tab3.txt",
    ]
    
    missing = []
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} - NOT FOUND")
            missing.append(filename)
    
    if missing:
        print(f"\n❌ Missing data files")
        return False
    
    print("\n✅ Data files found!")
    return True


def test_local_tda_small():
    """Test local TDA feature extraction with small sample"""
    print("\n" + "="*80)
    print("TESTING LOCAL TDA (SMALL SAMPLE)")
    print("="*80)
    
    try:
        # Import the module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "local_tda_features", 
            "03_local_tda_features.py"
        )
        local_tda = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(local_tda)
        
        # Create extractor
        extractor = local_tda.LocalTDAFeatureExtractor(data_dir="data")
        
        # Load data
        print("\nLoading data...")
        extractor.load_data()
        print(f"✓ Loaded {len(extractor.genes_df)} genes")
        
        # Build network
        print("\nBuilding network...")
        extractor.build_network()
        print(f"✓ Network: {extractor.lcc.number_of_nodes()} nodes")
        
        # Sample just a few nodes for testing
        print("\nSampling small test set (20 nodes)...")
        extractor.sample_target_nodes(neg_pos_ratio=2)
        
        # Override with smaller sample for testing
        import numpy as np
        np.random.seed(42)
        extractor.target_nodes = np.random.choice(
            extractor.target_nodes, 
            size=min(20, len(extractor.target_nodes)), 
            replace=False
        ).tolist()
        extractor.labels = np.array([
            1 if node in extractor.ad_genes else 0 
            for node in extractor.target_nodes
        ])
        
        print(f"✓ Test sample: {len(extractor.target_nodes)} nodes")
        
        # Extract features
        print("\nExtracting TDA features...")
        features_df = extractor.extract_all_features(radius=2)
        
        print(f"\n✅ SUCCESS! Extracted {len(features_df)} feature vectors")
        print(f"   Features per node: {len(features_df.columns) - 2}")
        print(f"\nSample features:")
        print(features_df.head(3))
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_network_features():
    """Test basic network feature computation"""
    print("\n" + "="*80)
    print("TESTING NETWORK FEATURES")
    print("="*80)
    
    try:
        import networkx as nx
        import numpy as np
        
        # Create simple test graph
        G = nx.karate_club_graph()
        print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Test ego graph extraction
        node = 0
        ego = nx.ego_graph(G, node, radius=2)
        print(f"✓ Ego graph (r=2) for node {node}: {len(ego.nodes())} nodes")
        
        # Test distance computation
        distances = dict(nx.all_pairs_shortest_path_length(ego))
        print(f"✓ Computed shortest path distances")
        
        # Test TDA computation
        from ripser import ripser as ripser_compute
        
        # Build distance matrix
        nodes = list(ego.nodes())
        n = len(nodes)
        dist_matrix = np.zeros((n, n))
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i != j:
                    dist_matrix[i, j] = distances[node_i].get(node_j, np.inf)
        
        # Compute persistence
        result = ripser_compute(dist_matrix, maxdim=1, distance_matrix=True, thresh=5.0)
        print(f"✓ Computed persistence: {len(result['dgms'][0])} H0, {len(result['dgms'][1])} H1 features")
        
        print("\n✅ Network feature computation works!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("QUICK PIPELINE TEST")
    print("="*80)
    print("\nThis script tests the pipeline with a small sample.")
    print("It should complete in < 1 minute.\n")
    
    # Run tests
    tests = [
        ("Import Test", test_imports),
        ("Data File Test", test_data_files),
        ("Network Features Test", test_network_features),
        ("Local TDA Test", test_local_tda_small),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {str(e)}")
            results[test_name] = False
        
        if not results[test_name]:
            print(f"\n⚠️  Stopping tests - {test_name} failed")
            break
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to run full pipeline.")
        print("\nNext steps:")
        print("  1. Run: python run_complete_analysis.py")
        print("  2. Or run individual scripts (03_*, 04_*)")
    else:
        print("\n⚠️  Some tests failed. Fix issues before running full pipeline.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

