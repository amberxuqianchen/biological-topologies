#!/usr/bin/env python3
"""
Machine Learning Pipeline for AD Gene Classification
Trains classifiers using TDA + network features to predict AD genes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✓ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available - install with: pip install xgboost")

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
    print("✓ SHAP available")
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available - install with: pip install shap")


class ADGeneClassifier:
    """ML pipeline for AD gene classification"""
    
    def __init__(self, features_file='computed_data/ad_network_features.csv'):
        self.features_file = features_file
        self.features_df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_features(self):
        """Load feature matrix"""
        print("="*80)
        print("LOADING FEATURES")
        print("="*80)
        
        self.features_df = pd.read_csv(self.features_file)
        print(f"✓ Loaded features: {self.features_df.shape}")
        
        # Separate features and labels
        X_cols = [col for col in self.features_df.columns 
                  if col not in ['node_id', 'is_ad']]
        
        self.X = self.features_df[X_cols].values
        self.y = self.features_df['is_ad'].values
        self.feature_names = X_cols
        
        print(f"✓ Features: {self.X.shape[1]}")
        print(f"✓ Samples: {self.X.shape[0]}")
        print(f"✓ Positive class: {sum(self.y)} ({sum(self.y)/len(self.y)*100:.1f}%)")
        
        # Check for NaN or inf and handle properly
        nan_count = np.sum(np.isnan(self.X))
        inf_count = np.sum(np.isinf(self.X))
        
        if nan_count > 0 or inf_count > 0:
            print(f"⚠️  Found {nan_count} NaN and {inf_count} inf values")
            
            # Replace inf with NaN first
            self.X = np.where(np.isinf(self.X), np.nan, self.X)
            
            # Count NaN per feature
            nan_per_feature = np.sum(np.isnan(self.X), axis=0)
            features_with_nan = [(self.feature_names[i], nan_per_feature[i]) 
                                 for i in range(len(nan_per_feature)) if nan_per_feature[i] > 0]
            if features_with_nan:
                print(f"   Features with missing values:")
                for fname, count in sorted(features_with_nan, key=lambda x: -x[1])[:5]:
                    print(f"     • {fname}: {count} ({count/self.X.shape[0]*100:.1f}%)")
            
            # Use median imputation (more robust than mean for skewed data)
            print("   Imputing missing values with median...")
            self.imputer = SimpleImputer(strategy='median')
            self.X = self.imputer.fit_transform(self.X)
            print("   ✓ Imputation complete")
        
        return self.X, self.y
    
    def prepare_feature_subsets(self):
        """Create feature subsets for comparison"""
        subsets = {}
        
        # All features
        all_features = list(range(len(self.feature_names)))
        subsets['all'] = all_features
        
        # TDA features only
        tda_idx = [i for i, name in enumerate(self.feature_names) if name.startswith('H')]
        if tda_idx:
            subsets['tda_only'] = tda_idx
        
        # Network features only
        network_idx = [i for i, name in enumerate(self.feature_names) if not name.startswith('H')]
        if network_idx:
            subsets['network_only'] = network_idx
        
        # H0 features
        h0_idx = [i for i, name in enumerate(self.feature_names) if name.startswith('H0')]
        if h0_idx:
            subsets['H0_features'] = h0_idx
        
        # H1 features
        h1_idx = [i for i, name in enumerate(self.feature_names) if name.startswith('H1')]
        if h1_idx:
            subsets['H1_features'] = h1_idx
        
        print(f"\n📦 Feature Subsets:")
        for name, idx in subsets.items():
            print(f"  • {name}: {len(idx)} features")
        
        return subsets
    
    def initialize_models(self):
        """Initialize classifier models"""
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        if XGBOOST_AVAILABLE:
            # Calculate scale_pos_weight for class imbalance
            scale_pos_weight = (len(self.y) - sum(self.y)) / sum(self.y)
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        return models
    
    def evaluate_model(self, model, X, y, cv_folds=5):
        """
        Evaluate model using cross-validation
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to evaluate
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        cv_folds : int
            Number of CV folds
            
        Returns:
        --------
        results : dict
            Dictionary of evaluation metrics
        """
        # Define scoring metrics
        scoring = {
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall'
        }
        
        # Stratified K-Fold
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Aggregate results
        results = {
            'auroc_mean': np.mean(cv_results['test_roc_auc']),
            'auroc_std': np.std(cv_results['test_roc_auc']),
            'auprc_mean': np.mean(cv_results['test_average_precision']),
            'auprc_std': np.std(cv_results['test_average_precision']),
            'f1_mean': np.mean(cv_results['test_f1']),
            'f1_std': np.std(cv_results['test_f1']),
            'precision_mean': np.mean(cv_results['test_precision']),
            'precision_std': np.std(cv_results['test_precision']),
            'recall_mean': np.mean(cv_results['test_recall']),
            'recall_std': np.std(cv_results['test_recall']),
            'cv_results': cv_results
        }
        
        return results
    
    def train_and_evaluate_all(self, feature_subset='all', cv_folds=5):
        """
        Train and evaluate all models
        
        Parameters:
        -----------
        feature_subset : str or list
            Feature subset to use ('all', 'tda_only', 'network_only', or list of indices)
        cv_folds : int
            Number of CV folds
        """
        print("\n" + "="*80)
        print(f"TRAINING AND EVALUATION")
        print("="*80)
        
        # Select features
        if isinstance(feature_subset, str):
            subsets = self.prepare_feature_subsets()
            feature_idx = subsets.get(feature_subset, subsets['all'])
        else:
            feature_idx = feature_subset
        
        X_subset = self.X[:, feature_idx]
        
        print(f"Using feature subset: {feature_subset} ({len(feature_idx)} features)")
        print(f"Cross-validation: {cv_folds}-fold stratified")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_subset)
        
        # Initialize models
        models = self.initialize_models()
        
        # Train and evaluate each model
        results = {}
        for model_name, model in models.items():
            print(f"\n⏳ Training {model_name}...")
            
            model_results = self.evaluate_model(model, X_scaled, self.y, cv_folds=cv_folds)
            results[model_name] = model_results
            
            print(f"  ✓ AUROC: {model_results['auroc_mean']:.4f} ± {model_results['auroc_std']:.4f}")
            print(f"  ✓ AUPRC: {model_results['auprc_mean']:.4f} ± {model_results['auprc_std']:.4f}")
            print(f"  ✓ F1:    {model_results['f1_mean']:.4f} ± {model_results['f1_std']:.4f}")
        
        self.results[feature_subset] = results
        return results
    
    def compare_feature_subsets(self, cv_folds=5):
        """Compare performance across different feature subsets"""
        print("\n" + "="*80)
        print("COMPARING FEATURE SUBSETS")
        print("="*80)
        
        subsets = self.prepare_feature_subsets()
        
        # Train on each subset
        for subset_name in ['all', 'tda_only', 'network_only']:
            if subset_name in subsets:
                print(f"\n{'='*80}")
                print(f"Feature Subset: {subset_name.upper()}")
                print('='*80)
                self.train_and_evaluate_all(feature_subset=subset_name, cv_folds=cv_folds)
    
    def compute_feature_importance(self, model_name='Random Forest', feature_subset='all', n_top=20):
        """
        Compute and visualize feature importance
        
        Parameters:
        -----------
        model_name : str
            Model to use for feature importance
        feature_subset : str
            Feature subset to use
        n_top : int
            Number of top features to display
        """
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Select features
        subsets = self.prepare_feature_subsets()
        feature_idx = subsets.get(feature_subset, subsets['all'])
        X_subset = self.X[:, feature_idx]
        X_scaled = self.scaler.fit_transform(X_subset)
        feature_names_subset = [self.feature_names[i] for i in feature_idx]
        
        # Train model
        models = self.initialize_models()
        model = models[model_name]
        model.fit(X_scaled, self.y)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            print(f"⚠️  {model_name} does not support feature importance")
            return None
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names_subset,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {n_top} Most Important Features ({model_name}):")
        print(importance_df.head(n_top).to_string(index=False))
        
        # Visualize
        self._plot_feature_importance(importance_df.head(n_top), model_name)
        
        return importance_df
    
    def _plot_feature_importance(self, importance_df, model_name):
        """Plot feature importance"""
        plt.figure(figsize=(10, 8))
        
        # Color by feature type
        colors = ['#e74c3c' if name.startswith('H') else '#3498db' 
                  for name in importance_df['feature']]
        
        plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', label='TDA Features'),
            Patch(facecolor='#3498db', label='Network Features')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Feature importance plot saved to: figures/feature_importance.png")
        plt.show()
    
    def plot_performance_comparison(self):
        """Plot performance comparison across models and feature subsets"""
        if not self.results:
            print("⚠️  No results to plot. Run training first.")
            return
        
        print("\n" + "="*80)
        print("PLOTTING PERFORMANCE COMPARISON")
        print("="*80)
        
        # Prepare data for plotting
        plot_data = []
        for subset_name, subset_results in self.results.items():
            for model_name, metrics in subset_results.items():
                plot_data.append({
                    'Feature Set': subset_name,
                    'Model': model_name,
                    'AUROC': metrics['auroc_mean'],
                    'AUROC_std': metrics['auroc_std'],
                    'AUPRC': metrics['auprc_mean'],
                    'AUPRC_std': metrics['auprc_std'],
                    'F1': metrics['f1_mean'],
                    'F1_std': metrics['f1_std']
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics = ['AUROC', 'AUPRC', 'F1']
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Group by model and feature set
            pivot_data = plot_df.pivot(index='Model', columns='Feature Set', values=metric)
            
            pivot_data.plot(kind='bar', ax=ax, width=0.8)
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.legend(title='Feature Set', loc='lower right')
            ax.set_ylim([0, 1.0])
            ax.grid(True, alpha=0.3)
            
            # Rotate x-labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('figures/performance_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Performance comparison plot saved to: figures/performance_comparison.png")
        plt.show()
    
    def print_summary_table(self):
        """Print summary table of all results"""
        if not self.results:
            print("⚠️  No results to display. Run training first.")
            return
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY TABLE")
        print("="*80)
        
        # Prepare data
        summary_data = []
        for subset_name, subset_results in self.results.items():
            for model_name, metrics in subset_results.items():
                summary_data.append({
                    'Feature Set': subset_name,
                    'Model': model_name,
                    'AUROC': f"{metrics['auroc_mean']:.3f} ± {metrics['auroc_std']:.3f}",
                    'AUPRC': f"{metrics['auprc_mean']:.3f} ± {metrics['auprc_std']:.3f}",
                    'F1': f"{metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f}",
                    'Precision': f"{metrics['precision_mean']:.3f} ± {metrics['precision_std']:.3f}",
                    'Recall': f"{metrics['recall_mean']:.3f} ± {metrics['recall_std']:.3f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        
        # Save to CSV
        summary_df.to_csv('data/classification_results_summary.csv', index=False)
        print("\n✓ Summary table saved to: data/classification_results_summary.csv")
    
    def run_complete_pipeline(self, cv_folds=5):
        """Run complete ML pipeline"""
        print("\n" + "="*80)
        print("MACHINE LEARNING CLASSIFICATION PIPELINE")
        print("="*80)
        
        # Create figures directory
        import os
        os.makedirs('figures', exist_ok=True)
        
        # Load features
        self.load_features()
        
        # Compare feature subsets
        self.compare_feature_subsets(cv_folds=cv_folds)
        
        # Feature importance
        self.compute_feature_importance(
            model_name='Random Forest',
            feature_subset='all',
            n_top=20
        )
        
        # Plot comparisons
        self.plot_performance_comparison()
        
        # Print summary
        self.print_summary_table()
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE!")
        print("="*80)


def main():
    """Main execution function"""
    
    # Initialize classifier
    classifier = ADGeneClassifier(features_file='computed_data/ad_network_features.csv')
    
    # Run complete pipeline
    classifier.run_complete_pipeline(cv_folds=5)
    
    return classifier


if __name__ == "__main__":
    classifier = main()

