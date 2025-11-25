# Quick Reference Card

## 🚀 Getting Started in 3 Steps

```bash
# 1. Test setup (1 minute)
python test_pipeline.py

# 2. Run full analysis (20-30 minutes)
python run_complete_analysis.py

# 3. Check results
ls -lh computed_data/ad_network_features.csv
ls -lh figures/*.png
cat data/classification_results_summary.csv
```

---

## 📁 Key Files

| File | Purpose | Runtime |
|------|---------|---------|
| `test_pipeline.py` | Verify setup works | ~1 min |
| `run_complete_analysis.py` | **Run everything** | ~20-30 min |
| `03_local_tda_features.py` | Extract TDA features | ~10-20 min |
| `04_ml_classification.py` | Train classifiers | ~5-10 min |
| `02_tda_bifiltration.py` | Parameter space TDA | ~2-3 min |

---

## 📊 Output Files

| File | What It Contains |
|------|------------------|
| `computed_data/ad_network_features.csv` | Feature matrix (1398 genes × 25 features) |
| `data/classification_results_summary.csv` | Model performance (AUROC, AUPRC, F1) |
| `figures/feature_importance.png` | Which features matter most |
| `figures/performance_comparison.png` | TDA vs network features |

---

## 🔧 Adjustable Parameters

### In `03_local_tda_features.py`:
```python
radius=2              # Ego graph size (try 1, 2, or 3)
neg_pos_ratio=2       # Class balance (try 1, 2, or 3)
```

### In `04_ml_classification.py`:
```python
cv_folds=5           # Cross-validation folds (try 5 or 10)
```

---

## 📈 Expected Performance

| Metric | Good | Excellent |
|--------|------|-----------|
| AUROC | 0.70-0.75 | 0.75-0.85 |
| AUPRC | 0.60-0.70 | 0.70-0.80 |
| F1 Score | 0.60-0.70 | 0.70-0.80 |

**Key comparison:** AUROC(all features) should be ~0.05-0.15 higher than AUROC(network only)

---

## 🔍 Features Extracted

### TDA Features (16)
- **H0** (8 features): Connected components
  - count, total_persistence, max_persistence, mean, median, std, entropy, infinite_count
- **H1** (8 features): Loops/cycles
  - count, total_persistence, max_persistence, mean, median, std, entropy, infinite_count

### Network Features (9)
- degree
- clustering_coefficient
- ego_size_1hop, ego_edges_1hop, ego_density_1hop
- ego_size_2hop, ego_edges_2hop, ego_density_2hop

---

## 📝 For Your Paper

### Methods Section
Copy from: `UPDATED_METHODS.md`

### Results Section Structure
1. Dataset overview (network stats)
2. Feature distributions (AD vs non-AD)
3. Classification performance (table + ROC curves)
4. Feature importance (bar chart)
5. Novel predictions (top candidates)

### Key Numbers to Report
- Network: 26,687 nodes, 137,659 edges
- Training set: 1,398 genes (466 AD, 932 non-AD)
- Features: 25 total (16 TDA, 9 network)
- Models: 4 (LogReg, RF, GBM, XGBoost)
- CV: 5-fold stratified
- Performance: AUROC = X.XX ± X.XX (report from your results)

---

## 🎤 For Your Presentation

### Slide Outline (10-12 slides)
1. Title
2. Motivation (why TDA?)
3. Dataset (BioGRID)
4. Method - Overview
5. Method - Local TDA
6. Method - Features
7. Results - Performance
8. Results - Features
9. Results - Predictions
10. Conclusions
11. Future Work
12. Questions?

### Key Message
"TDA captures higher-order network structure that improves AD gene prediction by X% over traditional network features."

---

## 🐛 Troubleshooting

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: ripser` | `pip install ripser` |
| `FileNotFoundError: data/...` | Check BioGRID data files in `data/` |
| `MemoryError` | Reduce `radius=1` or `neg_pos_ratio=1` |
| Poor performance (AUROC<0.6) | Check class balance, try different radius |

---

## 📚 Documentation

- **Start here:** `GETTING_STARTED.md` (your checklist)
- **Full docs:** `README.md` (complete guide)
- **Implementation:** `IMPLEMENTATION_SUMMARY.md` (what we built)
- **Methods:** `UPDATED_METHODS.md` (for your paper)

---

## ✅ Success Checklist

### Today
- [ ] Run `python test_pipeline.py`
- [ ] Run `python run_complete_analysis.py`
- [ ] Check outputs exist

### This Week
- [ ] Examine results (AUROC, feature importance)
- [ ] Copy methods section to paper
- [ ] Write results section
- [ ] Create presentation figures

### Before Presentation
- [ ] Make slides (10-12)
- [ ] Practice (10 minutes)
- [ ] Prepare for questions

---

## 💡 Quick Tips

1. **Start small**: Run `test_pipeline.py` first
2. **Use defaults**: Parameters are already optimized
3. **Check outputs**: Look at CSV files before writing
4. **Visualize**: Figures tell the story better than tables
5. **Interpret**: Don't just report numbers, explain what they mean

---

## 🎯 What Makes This Project Strong

✅ Novel application of TDA to AD networks
✅ Proper local TDA (per-node features)
✅ Rigorous ML evaluation (CV, multiple metrics)
✅ Feature importance analysis
✅ Clear biological interpretation

---

## 📞 Need Help?

1. Check inline code comments
2. Read documentation files (GETTING_STARTED.md, README.md)
3. Review test_pipeline.py for examples
4. Look at similar studies (Ramos et al. 2025, Long et al. 2025)

---

## ⏱️ Timeline

| When | What | Time |
|------|------|------|
| **Now** | Run pipeline | ~30 min |
| **Day 1-2** | Analyze results | 2-3 hours |
| **Day 3-4** | Write results section | 3-4 hours |
| **Day 5-6** | Make presentation | 3-4 hours |
| **Day 7+** | Practice & revise | 2-3 hours |

**You have time!** 📅

---

## 🎉 You're Ready!

Everything is implemented, documented, and tested. Just run it and write up your results!

**Good luck! 🚀**

---

*For detailed information, see:*
- *`GETTING_STARTED.md` - Action checklist*
- *`IMPLEMENTATION_SUMMARY.md` - What we built*
- *`README.md` - Complete documentation*

