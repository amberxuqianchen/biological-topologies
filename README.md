# biological-topologies
Biological networks (protein–protein interactions, gene–gene networks) are complex and noisy. Traditional graph features often fail to capture mesoscale structure. Topological data analysis (TDA) can identify higher-order features (holes, loops, communities) that may correlate with biological function.



# Topological Data Analysis of Alzheimer's Disease Protein Networks

A comprehensive analysis applying Topological Data Analysis (TDA) and persistent homology to protein-protein interaction (PPI) networks to identify and prioritize genes associated with Alzheimer's disease.

## Project Overview

This project applies mathematical tools from topological data analysis to understand the higher-order structure of biological networks in Alzheimer's disease. By analyzing protein-protein interactions from BioGRID, we aim to:

1. **Construct disease-specific PPI networks** from curated interaction data
2. **Apply single-parameter persistent homology** to quantify topological signatures
3. **Develop multi-parameter (bifiltration) approaches** incorporating network connectivity and gene expression
4. **Extract topological features** (barcode entropy, lifetimes, persistence images)
5. **Build machine learning models** to predict novel AD candidate genes

## Project Structure

```
291A/
├── data/                           # BioGRID Alzheimer's disease data
│   ├── BIOGRID-*-GENES-*.txt      # AD-associated genes (468 genes)
│   ├── BIOGRID-*-INTERACTIONS-*.txt # Protein interactions (184K interactions)
│   ├── BIOGRID-*-CHEMICALS-*.txt   # Chemical-gene interactions
│   └── BIOGRID-*-PTM-*.txt        # Post-translational modifications
├── 01_exploratory_analysis.ipynb   # Initial EDA and network construction
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

### 1. Create a Python environment

```bash
# Using conda (recommended)
conda create -n ad_tda python=3.10
conda activate ad_tda

# Or using venv
python -m venv ad_tda_env
source ad_tda_env/bin/activate  # On Windows: ad_tda_env\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

## Data Description

### Gene Annotations
- **468 AD-associated genes** categorized into:
  - **Amyloid gene set**: Genes related to amyloid-beta processing and plaques
  - **Tau modifier (NFT) gene set**: Genes involved in tau pathology and neurofibrillary tangles
  - Many genes belong to both categories

### Interaction Data
- **184,744 protein-protein interactions** from BioGRID
- Both physical and genetic interactions
- Experimental validation from multiple methodologies
- Includes interaction metadata (experimental system, throughput, publications)

## Analysis Workflow

### Phase 1: Exploratory Analysis (Current)
- ✅ Load and parse BioGRID data
- ✅ Analyze gene categories and annotations
- ✅ Construct PPI network using NetworkX
- ✅ Compute basic network properties (degree, clustering, betweenness)
- ✅ Identify hub genes and network structure
- ✅ Compare AD genes vs non-AD genes

### Phase 2: Topological Analysis (Next Steps)
- [ ] Apply Vietoris-Rips or Clique complex construction
- [ ] Compute persistent homology (H0, H1, H2)
- [ ] Generate persistence diagrams and barcodes
- [ ] Calculate topological features:
  - Betti numbers
  - Persistence entropy
  - Barcode statistics

### Phase 3: Multi-parameter Persistent Homology
- [ ] Integrate gene expression data (GTEx brain tissue)
- [ ] Implement bifiltration (degree + expression)
- [ ] Compute 2D persistence diagrams
- [ ] Generate persistence surfaces/images

### Phase 4: Machine Learning
- [ ] Feature engineering (graph + topological features)
- [ ] Train classifiers (Logistic Regression, Random Forest, XGBoost)
- [ ] Evaluate model performance
- [ ] Predict novel AD candidate genes
- [ ] Biological validation and interpretation

## Key Findings (Preliminary)

The exploratory analysis reveals:
- Large, densely connected PPI network with ~20K nodes
- Power-law degree distribution (scale-free network)
- High clustering coefficient indicating modular structure
- AD genes show distinct network properties compared to background

## Technologies

- **Python 3.10+**
- **NetworkX**: Graph construction and analysis
- **Gudhi/Ripser**: Persistent homology computation
- **scikit-learn**: Machine learning models
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

## References

### Foundational Papers
1. **Ramos et al. (2025)**: Persistent homology for cancer gene discovery
2. **Long et al. (2025)**: TDA for drug-target interaction networks

### Theoretical Background
- Persistent homology and simplicial complexes
- Multi-parameter persistent homology (bifiltrations)
- Topological signatures in biological networks

## Next Steps

To continue with the analysis:

1. **Run the exploratory notebook**: `01_exploratory_analysis.ipynb`
2. **Review network statistics** and identify key patterns
3. **Proceed to topological analysis** (upcoming notebooks)

## Contact

For questions about this analysis, please refer to the course materials or reach out during office hours.

---

**Last Updated**: October 22, 2025

