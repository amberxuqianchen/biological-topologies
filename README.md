# Alzheimer's Disease Network Analysis using Topological Data Analysis

This repository contains code for applying topological data analysis (TDA) to protein-protein interaction networks to identify and predict genes associated with Alzheimer's disease.

## Project Structure

```
biological-topologies/
├── README.md                           # This file
├── UPDATED_METHODS.md                  # Updated methods section for your paper
├── data/                               # Data directory (create this)
│   └── BIOGRID-PROJECT-*.txt          # BioGRID data files
├── figures/                            # Output figures (auto-created)
├── 00_descriptive_stats.py            # Dataset statistics
├── 01_exploratory_analysis.ipynb      # Initial exploration (notebook)
├── 02_tda_bifiltration.py             # Parameter space TDA (global)
├── 03_local_tda_features.py           # Local network TDA (per-node)
├── 04_ml_classification.py            # Machine learning pipeline
└── run_complete_analysis.py           # Master script (runs everything)
```

## Installation

### 1. Required Dependencies

```bash
pip install pandas numpy networkx matplotlib seaborn scikit-learn ripser
```

### 2. Optional Dependencies (Recommended)

```bash
pip install xgboost shap
```

- `xgboost`: For improved gradient boosting classifier
- `shap`: For model interpretability analysis

### 3. Data Setup

Download the BioGRID Alzheimer's Disease Project dataset (v5.0.250):

1. Visit: https://downloads.thebiogrid.org/BioGRID/Release-Archive/BIOGRID-5.0.250/
2. Download the AD project files
3. Place in `data/` directory with these filenames:
   - `BIOGRID-PROJECT-alzheimers_disease_project-GENES-5.0.250.projectindex.txt`
   - `BIOGRID-PROJECT-alzheimers_disease_project-INTERACTIONS-5.0.250.tab3.txt`
   - `BIOGRID-PROJECT-alzheimers_disease_project-CHEMICALS-5.0.250.chemtab.txt`

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
python run_complete_analysis.py
```

This runs all analysis steps in sequence:
1. Local TDA feature extraction (~10-20 minutes)
2. ML classification with cross-validation (~5-10 minutes)
3. Parameter space TDA analysis (~2-3 minutes)

**Expected output:**
- `computed_data/ad_network_features.csv` - Feature matrix
- `data/classification_results_summary.csv` - Model performance
- `figures/feature_importance.png` - Feature importance plot
- `figures/performance_comparison.png` - Model comparison

---

### Option 2: Run Steps Individually

#### Step 1: Extract Local TDA Features

```python
from biological_topologies.local_tda_features import LocalTDAFeatureExtractor

extractor = LocalTDAFeatureExtractor(data_dir="data")
features_df = extractor.run_complete_pipeline(
    radius=2,              # 2-hop ego graphs
    neg_pos_ratio=2,       # 2:1 negative to positive
    output_file='computed_data/ad_network_features.csv'
)
```

**What this does:**
- Loads BioGRID data and builds PPI network
- Samples 466 AD genes (positive) + 932 non-AD genes (negative)
- For each gene, extracts 2-hop ego graph
- Computes persistent homology using Vietoris-Rips filtration
- Extracts 16 TDA features + 9 network features per node
- Saves feature matrix to CSV

**Parameters you can adjust:**
- `radius`: Ego graph size (1, 2, or 3 hops)
- `neg_pos_ratio`: Class balance (1, 2, or 3)

---

#### Step 2: Train ML Classifiers

```python
from biological_topologies.ml_classification import ADGeneClassifier

classifier = ADGeneClassifier(features_file='computed_data/ad_network_features.csv')
classifier.run_complete_pipeline(cv_folds=5)
```

**What this does:**
- Loads feature matrix from Step 1
- Trains 4 models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- Evaluates using 5-fold stratified cross-validation
- Computes AUROC, AUPRC, F1, precision, recall
- Compares performance across feature subsets (all, TDA-only, network-only)
- Generates feature importance plots

---

#### Step 3: Parameter Space TDA (Optional)

```python
from biological_topologies.tda_bifiltration import EfficientTDAAnalyzer

analyzer = EfficientTDAAnalyzer(data_dir="data")
results = analyzer.run_complete_analysis()
```

**What this does:**
- Projects proteins into 2D space (PTM count × Chemical interactions)
- Computes global persistent homology on point cloud
- Visualizes parameter space and persistence diagrams
- Provides complementary global perspective vs. local TDA in Steps 1-2

---

### Option 3: Get Dataset Statistics Only

```bash
python 00_descriptive_stats.py
```

Prints comprehensive statistics about the BioGRID dataset (useful for Methods/Dataset section).

---

## Understanding the Output

### Feature Matrix (`ad_network_features.csv`)

Each row is a protein with these columns:

**Metadata:**
- `node_id`: Entrez Gene ID
- `is_ad`: Label (1 = AD gene, 0 = non-AD gene)

**TDA Features (H0 - Connected Components):**
- `H0_count`: Number of components
- `H0_total_persistence`: Sum of component lifetimes
- `H0_max_persistence`: Longest-lived component
- `H0_mean_persistence`: Average lifetime
- `H0_median_persistence`: Median lifetime
- `H0_std_persistence`: Lifetime variability
- `H0_entropy`: Barcode complexity
- `H0_infinite_count`: Infinite components

**TDA Features (H1 - Loops/Cycles):**
- `H1_count`: Number of loops
- `H1_total_persistence`: Sum of loop lifetimes
- `H1_max_persistence`: Longest-lived loop
- `H1_mean_persistence`: Average lifetime
- `H1_median_persistence`: Median lifetime
- `H1_std_persistence`: Lifetime variability
- `H1_entropy`: Barcode complexity
- `H1_infinite_count`: Infinite loops

**Network Features:**
- `degree`: Number of direct neighbors
- `clustering_coefficient`: Local triangle density
- `ego_size_1hop`: Size of 1-hop neighborhood
- `ego_edges_1hop`: Edges in 1-hop neighborhood
- `ego_density_1hop`: Density of 1-hop neighborhood
- `ego_size_2hop`: Size of 2-hop neighborhood
- `ego_edges_2hop`: Edges in 2-hop neighborhood
- `ego_density_2hop`: Density of 2-hop neighborhood

---

### Classification Results (`classification_results_summary.csv`)

Performance metrics for each model and feature subset combination:

| Feature Set | Model | AUROC | AUPRC | F1 | Precision | Recall |
|------------|-------|-------|-------|----|-----------| -------|
| all | Random Forest | 0.XXX | 0.XXX | ... | ... | ... |
| tda_only | Random Forest | 0.XXX | 0.XXX | ... | ... | ... |
| network_only | Random Forest | 0.XXX | 0.XXX | ... | ... | ... |

**How to interpret:**
- **AUROC** (higher is better): Overall classification ability
- **AUPRC** (higher is better): Performance on minority class (AD genes)
- **F1** (higher is better): Balance of precision and recall
- Compare "all" vs "tda_only" vs "network_only" to see TDA's added value

---

## Expected Results

Based on similar studies (Ramos et al. 2025, Long et al. 2025), you should expect:

1. **Classification Performance:**
   - AUROC: 0.70-0.85 (depending on model)
   - AUPRC: 0.60-0.80
   - TDA features should improve performance by ~5-15% over network features alone

2. **Feature Importance:**
   - Top features typically include: degree, H1_count, H1_max_persistence, clustering_coefficient
   - TDA features capture complementary signal not present in traditional metrics

3. **AD vs Non-AD Differences:**
   - AD genes should have higher average degree (~2-3x)
   - AD genes may have distinct H1 signatures (more loops or longer persistence)

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'ripser'"

```bash
pip install ripser
```

### "FileNotFoundError: data/BIOGRID-PROJECT-..."

Make sure you've downloaded the BioGRID data files and placed them in the `data/` directory.

### "MemoryError" or very slow execution

Try reducing the ego graph radius or sample size:

```python
extractor.run_complete_pipeline(
    radius=1,           # Smaller ego graphs
    neg_pos_ratio=1     # Fewer negative samples
)
```

### Ripser warnings about infinite values

This is normal - some homology features persist forever (e.g., the main connected component). The code filters these appropriately.

---

## Customization

### Adjust Ego Graph Size

```python
# Smaller ego graphs (faster, fewer features)
extractor.run_complete_pipeline(radius=1)

# Larger ego graphs (slower, richer features)
extractor.run_complete_pipeline(radius=3)
```

### Change Class Balance

```python
# Balanced classes (1:1)
extractor.run_complete_pipeline(neg_pos_ratio=1)

# More negative samples (1:3)
extractor.run_complete_pipeline(neg_pos_ratio=3)
```

### Add More Models

Edit `04_ml_classification.py` and add to `initialize_models()`:

```python
from sklearn.svm import SVC

models['SVM'] = SVC(kernel='rbf', probability=True, class_weight='balanced')
```

### Hyperparameter Tuning

Use GridSearchCV or RandomizedSearchCV for optimal parameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='roc_auc'
)
grid_search.fit(X_scaled, y)
```

---

## Citation

If you use this code, please cite:

1. **BioGRID Database:**
   Oughtred et al. (2021) The BioGRID database: A comprehensive biomedical resource of curated protein, genetic, and chemical interactions. Protein Science, 30(1):187-200.

2. **Ripser:**
   Tralie et al. (2018) Ripser.py: A Lean Persistent Homology Library for Python. Journal of Open Source Software, 3(29):925.

3. **Related TDA in Biology:**
   - Ramos et al. (2025) Identifying Key Genes in Cancer Networks Using Persistent Homology. Scientific Reports.
   - Long et al. (2025) PATH: Persistent Homology for Drug-Target Binding Affinity Prediction.

---

## Contact & Support

For questions about the code:
- Check `UPDATED_METHODS.md` for detailed methodology
- Review the inline code documentation
- Open an issue on GitHub (if available)

For questions about the biology:
- Refer to BioGRID documentation
- Review cited papers (Ramos et al. 2025, Long et al. 2025)

---

## Timeline Checklist for Dec 1/8 Presentation

- [ ] **Day 1-2**: Run complete pipeline, verify outputs
- [ ] **Day 3-4**: Analyze results, identify top features and predictions
- [ ] **Day 5-6**: Create figures for presentation
- [ ] **Day 7**: Validate top predictions against literature
- [ ] **Day 8**: Prepare slides and practice

**Key figures to prepare:**
1. Network diagram with AD genes highlighted
2. Example persistence diagram with interpretation
3. ROC/PR curves comparing feature subsets
4. Feature importance bar chart
5. Table of classification metrics
6. Heatmap or scatter of top predicted genes

**Key results to highlight:**
1. How much does TDA improve prediction? (% increase in AUROC)
2. Which TDA features matter most?
3. What new AD candidates did we discover?
4. Do predictions make biological sense?

Good luck! 🎉
