# Getting Started - Quick Guide

## What We've Built

You now have a **complete pipeline** for applying Topological Data Analysis (TDA) to predict Alzheimer's disease genes from protein interaction networks. Here's what's implemented:

### ✅ Core Components

1. **Local TDA Feature Extraction** (`03_local_tda_features.py`)
   - Extracts ego graphs (local neighborhoods) for each protein
   - Computes persistent homology using Vietoris-Rips filtration
   - Generates 16 TDA features + 9 network features per node
   - Creates training dataset (466 AD genes + 932 non-AD genes)

2. **Machine Learning Pipeline** (`04_ml_classification.py`)
   - Trains 4 classifiers: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
   - 5-fold cross-validation with stratification
   - Compares TDA vs network features
   - Feature importance analysis
   - Performance metrics (AUROC, AUPRC, F1)

3. **Parameter Space TDA** (`02_tda_bifiltration.py`)
   - Global topology analysis in PTM × Chemical space
   - Complementary to local approach
   - Good for visualization and interpretation

4. **Master Script** (`run_complete_analysis.py`)
   - Runs entire pipeline end-to-end
   - Dependency checking
   - Progress tracking

5. **Documentation**
   - `README.md` - Full documentation
   - `UPDATED_METHODS.md` - Methods section for your paper
   - This file - Quick start guide

---

## First Steps (Do These Now!)

### 1. Install Dependencies

```bash
# Required
pip install pandas numpy networkx matplotlib seaborn scikit-learn ripser

# Optional but recommended
pip install xgboost shap
```

### 2. Test Your Setup

```bash
python test_pipeline.py
```

This runs quick tests (~1 minute) to verify:
- All packages installed correctly
- Data files are present
- TDA computation works
- Network features compute properly

**If all tests pass** → You're ready to run the full pipeline!

**If tests fail** → Check error messages and fix dependencies/data files

### 3. Run Quick Test with Full Pipeline (Optional)

Before running on all 1,400 genes, you can test with a smaller sample by modifying `03_local_tda_features.py`:

In the `main()` function, change:
```python
neg_pos_ratio=2,       # Change to 0.5 for tiny test
```

This will use ~155 genes instead of 1,400 and finish in ~5 minutes.

### 4. Run Full Pipeline

```bash
python run_complete_analysis.py
```

**Expected runtime:**
- Step 1 (TDA features): ~10-20 minutes
- Step 2 (ML training): ~5-10 minutes  
- Step 3 (Parameter space): ~2-3 minutes
- **Total: ~20-35 minutes**

**Output files:**
- `computed_data/ad_network_features.csv` - Your feature matrix
- `data/classification_results_summary.csv` - Performance metrics
- `figures/feature_importance.png` - Which features matter most
- `figures/performance_comparison.png` - Model comparison

---

## Understanding Your Results

### What to Look For

After running the pipeline, check:

1. **Classification Performance** (`data/classification_results_summary.csv`)
   ```
   Feature Set: all
   Model: Random Forest
   AUROC: 0.XXX ± 0.XXX   ← Should be > 0.70 (good), > 0.80 (excellent)
   AUPRC: 0.XXX ± 0.XXX   ← Should be > 0.60
   ```

2. **TDA Contribution**
   - Compare "all" vs "tda_only" vs "network_only"
   - If "all" > "network_only" by ~0.05-0.15 → TDA is helping!

3. **Important Features** (`figures/feature_importance.png`)
   - Top features are usually: degree, H1_count, H1_max_persistence
   - Mix of red (TDA) and blue (network) = both matter

### Expected Performance

Based on similar studies:
- **AUROC: 0.70-0.85** (realistic range)
- **AUPRC: 0.60-0.80** (harder metric)
- **TDA improvement: 5-15%** over network features alone

If you get these numbers → **Your project is successful!** ✅

---

## For Your Paper

### Use the Updated Methods Section

Open `UPDATED_METHODS.md` and copy sections to your paper:
- Dataset description (already matches your intro)
- Network Construction (adds LCC details)
- **Topological Data Analysis** (completely rewritten to match code)
- **Machine Learning Classifiers** (new section)

### Key Changes from Original:

| Original Methods | Updated Methods |
|-----------------|-----------------|
| Vague about "local networks" | Explicit: 2-hop ego graphs |
| Mentioned RIVET (not implemented) | Removed RIVET, honest about using Ripser |
| Didn't specify TDA features | Lists all 16 features with definitions |
| Unclear about bi-filtration | Clarified as "parameter space topology" |
| No ML details | Full ML section with CV, metrics, etc. |

---

## For Your Results Section

Structure your results like this:

### 1. Dataset Overview (1 paragraph)
- Network properties from `00_descriptive_stats.py`
- AD gene coverage in LCC
- Class balance in training set

### 2. Feature Distribution (1 paragraph + figure)
- Compare TDA features between AD vs non-AD genes
- Show that AD genes have distinct topological signatures
- **Figure**: Violin plots or box plots of key features

### 3. Classification Performance (1-2 paragraphs + table/figure)
- Table of AUROC/AUPRC for all models
- Highlight best performer
- Show TDA improves over network-only baseline
- **Figure**: ROC curves or performance comparison bar chart

### 4. Feature Importance (1 paragraph + figure)
- Which features matter most?
- Mix of TDA and network features
- Interpret biological meaning
- **Figure**: Feature importance bar chart (already generated!)

### 5. Novel Predictions (1 paragraph + table)
- Load `computed_data/ad_network_features.csv`
- Find non-AD genes with high predicted probabilities
- List top 10 candidates with their features
- Brief literature check for validation

### 6. Parameter Space Analysis (optional, 1 paragraph + figure)
- Results from `02_tda_bifiltration.py`
- How proteins cluster in PTM/chemical space
- Complements local TDA results

---

## For Your Presentation (Dec 1 or 8)

### Slide Structure (10-12 slides)

1. **Title** - "Using TDA to Predict Alzheimer's Genes"
2. **Motivation** - Why TDA? (captures higher-order structure)
3. **Dataset** - BioGRID, 26K proteins, 466 AD genes
4. **Network Visualization** - PPI network with AD genes highlighted
5. **TDA Method** - Ego graph → Vietoris-Rips → Persistence features
6. **Example** - One persistence diagram with interpretation
7. **Machine Learning** - 4 models, 5-fold CV, feature comparison
8. **Results: Performance** - AUROC comparison table/chart
9. **Results: Features** - Feature importance chart
10. **Results: Predictions** - Top candidate genes
11. **Conclusions** - TDA improves prediction, generalizable method
12. **Future Work** - Validate predictions, other diseases

### Key Messages

1. **Problem**: AD is complex, need network-level features
2. **Solution**: TDA captures topology traditional metrics miss
3. **Result**: TDA improves prediction by X% (your AUROC difference)
4. **Impact**: Found Y novel candidate genes for validation

### Visuals to Create

Run these scripts to generate figures:

```python
# From 04_ml_classification.py
classifier = ADGeneClassifier('computed_data/ad_network_features.csv')
classifier.load_features()

# Generate ROC curves
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# ... (code to create ROC curves)

# Network visualization
import networkx as nx
# Highlight AD genes in red, non-AD in blue
# Use spring layout or force-directed layout
```

---

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: ripser"**
```bash
pip install ripser
```

**"FileNotFoundError: data/BIOGRID-PROJECT-..."**
- Download data from BioGRID website
- Place files in `data/` directory

**"MemoryError" or very slow**
- Reduce ego graph radius: `radius=1` instead of `radius=2`
- Reduce sample size: `neg_pos_ratio=1` instead of `neg_pos_ratio=2`

**Poor classification performance (AUROC < 0.60)**
- Check class balance
- Try different radius values
- Inspect features for NaN/inf values
- Check if AD genes are actually in network

---

## Next Steps Checklist

### This Week
- [ ] Run `python test_pipeline.py` to verify setup
- [ ] Run `python run_complete_analysis.py` for full results
- [ ] Examine output files and figures
- [ ] Copy updated methods section to your paper

### Next Week  
- [ ] Write Results section using generated outputs
- [ ] Create additional figures for paper/presentation
- [ ] Validate top predictions against literature
- [ ] Prepare presentation slides

### Week 10 (Presentation)
- [ ] Practice presentation (aim for 10-12 minutes)
- [ ] Prepare to answer questions about:
  - Why TDA? What does it capture?
  - How does Vietoris-Rips work?
  - Why ego graphs? Why radius=2?
  - How do you validate predictions?

---

## Getting Help

### Code Questions
- Check inline comments in the Python files
- Review `README.md` for detailed documentation
- Look at examples in `test_pipeline.py`

### Methods Questions
- See `UPDATED_METHODS.md` for detailed explanations
- Review cited papers (Ramos et al., Long et al.)

### TDA Questions
- Ripser documentation: https://ripser.scikit-tda.org/
- Persistence tutorials: https://www.math.upenn.edu/~ghrist/notes.html

---

## Quick Command Reference

```bash
# Test setup (fast)
python test_pipeline.py

# Run full pipeline
python run_complete_analysis.py

# Run individual steps
python 03_local_tda_features.py
python 04_ml_classification.py
python 02_tda_bifiltration.py

# Get dataset statistics
python 00_descriptive_stats.py
```

---

## Success Criteria

Your project is **successful** if:

✅ Pipeline runs without errors
✅ AUROC > 0.70 (preferably > 0.75)
✅ TDA features improve performance over network-only baseline
✅ Feature importance shows mix of TDA and network features
✅ Top predictions have biological plausibility

You're already 90% done! Just need to **run the code** and **write up results**. 🎉

---

**Good luck with your presentation! You've got this! 🚀**

