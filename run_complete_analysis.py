#!/usr/bin/env python3
"""
Master Script for Complete AD Network TDA Analysis
Runs the full pipeline from data loading to classification
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    print("="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)
    
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'networkx': 'networkx',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'ripser': 'ripser'
    }
    
    optional = {
        'xgboost': 'xgboost',
        'shap': 'shap'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (REQUIRED)")
            missing.append(package)
    
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"✓ {package} (optional)")
        except ImportError:
            print(f"  {package} (optional - not installed)")
    
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All required dependencies satisfied!")
    return True


def check_data_files():
    """Check if data files exist"""
    print("\n" + "="*80)
    print("CHECKING DATA FILES")
    print("="*80)
    
    data_dir = Path("data")
    
    required_files = [
        "BIOGRID-PROJECT-alzheimers_disease_project-GENES-5.0.250.projectindex.txt",
        "BIOGRID-PROJECT-alzheimers_disease_project-INTERACTIONS-5.0.250.tab3.txt",
        "BIOGRID-PROJECT-alzheimers_disease_project-CHEMICALS-5.0.250.chemtab.txt"
    ]
    
    missing = []
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file}")
            missing.append(file)
    
    if missing:
        print(f"\n❌ Missing data files. Please download BioGRID dataset.")
        return False
    
    print("\n✅ All data files found!")
    return True


def run_step_1_local_tda():
    """Step 1: Extract local TDA features"""
    print("\n" + "="*80)
    print("STEP 1: LOCAL TDA FEATURE EXTRACTION")
    print("="*80)
    
    try:
        # Import from local file
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "local_tda_features", 
            "03_local_tda_features.py"
        )
        local_tda = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(local_tda)
        
        extractor = local_tda.LocalTDAFeatureExtractor(data_dir="data")
        features_df = extractor.run_complete_pipeline(
            radius=2,
            neg_pos_ratio=2,
            output_file='computed_data/ad_network_features.csv'
        )
        
        print("\n✅ Step 1 complete!")
        return True, extractor
        
    except Exception as e:
        print(f"\n❌ Step 1 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def run_step_2_classification(force_rerun=False):
    """Step 2: ML classification"""
    print("\n" + "="*80)
    print("STEP 2: MACHINE LEARNING CLASSIFICATION")
    print("="*80)
    
    # Check if features file exists
    features_file = Path('computed_data/ad_network_features.csv')
    if not features_file.exists() and not force_rerun:
        print("❌ Features file not found. Run Step 1 first.")
        return False, None
    
    try:
        # Import from local file
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ml_classification", 
            "04_ml_classification.py"
        )
        ml_class = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ml_class)
        
        classifier = ml_class.ADGeneClassifier(features_file='computed_data/ad_network_features.csv')
        classifier.run_complete_pipeline(cv_folds=5)
        
        print("\n✅ Step 2 complete!")
        return True, classifier
        
    except Exception as e:
        print(f"\n❌ Step 2 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def run_step_3_bifiltration():
    """Step 3: Bi-filtration analysis (parameter space TDA)"""
    print("\n" + "="*80)
    print("STEP 3: BI-FILTRATION ANALYSIS (PARAMETER SPACE)")
    print("="*80)
    
    try:
        # Import from local file
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "tda_bifiltration", 
            "02_tda_bifiltration.py"
        )
        tda_bifilt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tda_bifilt)
        
        analyzer = tda_bifilt.EfficientTDAAnalyzer(data_dir="data")
        results = analyzer.run_complete_analysis()
        
        print("\n✅ Step 3 complete!")
        return True, analyzer
        
    except Exception as e:
        print(f"\n❌ Step 3 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def print_final_summary():
    """Print final summary and next steps"""
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*80)
    
    print("""
📁 Output Files Generated:
  • computed_data/ad_network_features.csv - Feature matrix for classification
  • data/classification_results_summary.csv - Model performance metrics
  • figures/feature_importance.png - Feature importance visualization
  • figures/performance_comparison.png - Model comparison plots

📊 Key Results:
  • Local network TDA features extracted for ~1400 genes
  • Multiple classifiers trained with cross-validation
  • Feature importance analysis completed
  • Performance metrics computed (AUROC, AUPRC, F1)

📝 Next Steps for Your Paper:
  1. Review classification results in data/classification_results_summary.csv
  2. Analyze feature importance to understand which TDA features matter
  3. Identify top predicted AD candidate genes (high confidence, not in training set)
  4. Validate predictions against literature or pathway databases
  5. Create publication-quality figures from generated plots

🔬 For Methods Section:
  • Local TDA: 2-hop ego graphs, Vietoris-Rips filtration
  • Features: H0/H1 persistence statistics + network metrics
  • Models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
  • Evaluation: 5-fold stratified cross-validation
  • Metrics: AUROC, AUPRC, F1-score, precision, recall

💡 Additional Analyses You Can Run:
  • Vary ego graph radius (1, 2, 3 hops) to see impact
  • Test different neg:pos ratios (1:1, 2:1, 3:1)
  • Hyperparameter tuning for best model
  • Generate ROC/PR curves for publication
  • SHAP analysis for model interpretability (if SHAP installed)
    """)


def main():
    """Main execution function"""
    print("="*80)
    print("ALZHEIMER'S DISEASE NETWORK TDA ANALYSIS")
    print("Complete Pipeline Execution")
    print("="*80)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        sys.exit(1)
    
    # Create output directories
    Path("data").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    
    # Run pipeline steps
    print("\n" + "="*80)
    print("STARTING PIPELINE")
    print("="*80)
    
    # Step 1: Local TDA feature extraction
    success_1, extractor = run_step_1_local_tda()
    if not success_1:
        print("\n❌ Pipeline failed at Step 1")
        sys.exit(1)
    
    # Step 2: ML classification
    success_2, classifier = run_step_2_classification()
    if not success_2:
        print("\n❌ Pipeline failed at Step 2")
        sys.exit(1)
    
    # Step 3: Bi-filtration analysis (optional, for comparison)
    print("\n" + "="*80)
    print("RUNNING OPTIONAL STEP 3 (BI-FILTRATION)")
    print("="*80)
    print("This step runs parameter space TDA for comparison.")
    
    try:
        success_3, analyzer = run_step_3_bifiltration()
    except:
        print("⚠️  Step 3 skipped or failed (optional)")
    
    # Print final summary
    print_final_summary()
    
    print("\n" + "="*80)
    print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()

