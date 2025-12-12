# Alzheimer's Disease Network Analysis using Topological Data Analysis

This repository applies topological data analysis (TDA) to protein-protein interaction networks to identify and predict genes associated with Alzheimer's disease and other diseases.

## Overview

The project implements two TDA approaches:
1. **Perturbation TDA**: Computes topological features by removing nodes and measuring changes in homology
2. **Bifiltration TDA**: Multi-scale analysis filtering by post-translational modification (PTM) levels

**Key Result**: Bifiltration TDA improves AD gene classification by 17.2% (p < 10⁻³⁸) over graph features alone.

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies: `pandas`, `numpy`, `networkx`, `scikit-learn`, `gudhi`, `ripser`, `xgboost`

## Data Setup

Download BioGRID data files and place in `data/` directory:
- `BIOGRID-PROJECT-alzheimers_disease_project-GENES-*.projectindex.txt`
- `BIOGRID-PROJECT-alzheimers_disease_project-INTERACTIONS-*.tab3.txt`
- `BIOGRID-PROJECT-alzheimers_disease_project-PTM-*.ptmtab.txt`
- `BIOGRID-HUMAN-*.tab3.txt` (full human interactome)

See `data_setup_instructions.md` for details.

## Getting Started

### 1. Check Existing Results

Most analysis has already been run. Check for existing files:

```bash
ls computed_data/tda_bifiltration_features.csv
ls computed_data/tda_perturbation_*.csv
ls figures/tda_method_comparison.png
```

### 2. Review Results

```bash
# Compare methods and see performance metrics
python 22_compare_tda_methods.py

# Generate visualizations
python 23_visualize_comparison.py
```

### 3. Run New Analysis (if needed)

If you need to recompute features or run analysis for new diseases:

```bash
# Perturbation TDA for specific disease
python 07_perturbation_tda.py

# Bifiltration TDA
python 17_bifiltration_tda.py

# Batch processing for multiple diseases (overnight)
python 18_overnight_bifiltration.py
```

## What to Run: Scripts and Outputs

### Core TDA Feature Extraction

| Script | What It Does | Output Files |
|--------|--------------|--------------|
| `07_perturbation_tda.py` | Extracts perturbation TDA features (removes nodes, measures topology change) | `computed_data/tda_perturbation_*.csv` (by disease) |
| `17_bifiltration_tda.py` | Extracts bifiltration TDA features (PTM-based multi-scale filtering) | `computed_data/tda_bifiltration_*.csv` (by disease) |
| `09_overnight_batch_tda.py` | Batch perturbation TDA for multiple diseases | `computed_data/tda_perturbation_*.csv` (multiple diseases) |
| `18_overnight_bifiltration.py` | Batch bifiltration TDA for multiple diseases | `computed_data/tda_bifiltration_*.csv` (multiple diseases) |

### Analysis & Validation

| Script | What It Does | Output Files |
|--------|--------------|--------------|
| `10_explore_classification_tasks.py` | Finds best classification scenario | Console output with performance metrics |
| `11_statistical_significance.py` | Validates statistical significance | Console output with p-values and confidence intervals |
| `13_check_study_bias.py` | Checks for data collection bias | Console output with bias analysis |
| `14_test_without_simplices.py` | Tests with biased features removed | Console output with performance comparison |
| `22_compare_tda_methods.py` | Compares perturbation vs bifiltration TDA | Console output with performance comparison |
| `23_visualize_comparison.py` | Generates comparison figures | `figures/tda_method_comparison.png`, `figures/bifilt_feature_importance.png` |

### Quick Commands

```bash
# Review existing results
python 22_compare_tda_methods.py

# Generate all visualizations
python 23_visualize_comparison.py

# Run full validation pipeline (scripts 10-16)
python 10_explore_classification_tasks.py
python 11_statistical_significance.py
python 13_check_study_bias.py
python 14_test_without_simplices.py
```

## Output Files

### Feature Matrices

- **`computed_data/tda_perturbation_*.csv`** - Perturbation TDA features for each disease
  - Columns: `node_id`, `delta_H0`, `delta_H1`, `delta_H2`, `delta_H3`, plus network features
  - Files: `tda_perturbation_alzheimers.csv`, `tda_perturbation_autism.csv`, etc.

- **`computed_data/tda_bifiltration_features.csv`** - Combined bifiltration features (AD + matched background)
  - Columns: `node_id`, `is_ad`, `delta_H*_ptm_slope`, `delta_H*_ptm_range`, plus network features
  - Used for final classification analysis

- **`computed_data/tda_bifiltration_*.csv`** - Bifiltration TDA features for each disease
  - Files: `tda_bifiltration_alzheimers.csv`, `tda_bifiltration_autism.csv`, etc.

### Results

- **`computed_data/top_ad_candidates.csv`** - Predicted AD candidate genes with scores

### Figures

- **`figures/tda_method_comparison.png`** - Performance comparison (Graph vs Perturbation vs Bifiltration)
- **`figures/bifilt_feature_importance.png`** - Feature importance for bifiltration TDA
- **`figures/bifiltration_analysis.png`** - Bifiltration concept visualization

## Key Results

| Method | AUROC | Improvement over Graph |
|--------|-------|------------------------|
| Graph only | 0.514 | baseline |
| + Perturbation TDA | 0.602 | +8.8% |
| + Bifiltration TDA | **0.686** | **+17.2%** |

**Top features:** `delta_H2_ptm_slope`, `delta_H1_ptm_slope`, `delta_H1_ptm_range`

### TDA Methods Explained

**Perturbation TDA:**
- Removes each node and measures how topology changes
- Features: `delta_H0`, `delta_H1`, `delta_H2`, `delta_H3` (change in Betti numbers)
- Captures structural importance of nodes

**Bifiltration TDA:**
- Filters ego graphs by PTM levels (ptm10, ptm25, ptm50, ptm75, ptm90, ptm100)
- Computes topology at each threshold
- Features: `delta_H*_ptm_slope` (how topology changes with PTM), `delta_H*_ptm_range` (range across thresholds)
- Captures multi-scale topological response to PTM filtering

## Project Structure

```
biological-topologies/
├── computed_data/          # Feature matrices and results
├── figures/                # Generated visualizations
├── notes/                  # Additional documentation
├── scripts/                # Utility scripts
├── 00-24_*.py             # Numbered analysis scripts
├── ANALYSIS_PIPELINE.md   # Detailed methodology and results
└── requirements.txt       # Python dependencies
```

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: gudhi" or "ripser"**
```bash
pip install -r requirements.txt
```

**"FileNotFoundError: data/BIOGRID-..."**
- Download BioGRID data files from https://downloads.thebiogrid.org/
- Place in `data/` directory
- See `data_setup_instructions.md` for details

**Missing computed_data files**
- Most analysis has been run and files should exist
- Check `computed_data/` for existing files
- Run batch scripts if needed (overnight processing for large datasets)

**Memory errors or slow execution**
- Batch scripts are designed for overnight processing
- Individual scripts may take 10-30 minutes depending on dataset size

## Documentation

- **`notes/ANALYSIS_PIPELINE.md`** - Complete methodology, results, and interpretation
- **`notes/DATASET_DOCUMENTATION.md`** - Detailed dataset information
