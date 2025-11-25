# Updated Methods Section

## Methods

### Dataset

This study utilized the BioGRID Alzheimer's Disease Project dataset (version 5.0.250), a curated collection of experimentally validated human protein interactions and annotations specifically related to Alzheimer's disease. The dataset comprises five complementary data sources:

**Gene Annotations**: 466 AD-associated genes were categorized into two primary pathological pathways: an amyloid gene set (219 genes linked to amyloid-β production, aggregation, and clearance, including regulators of APP processing and plaque formation) and a tau-modifier gene set (300 genes that influence tau phosphorylation, stability, and aggregation into neurofibrillary tangles). Notably, 53 genes belonged to both categories, reflecting shared pathways such as proteostasis, vesicular trafficking, and neuroinflammation that contribute to the interplay between amyloid and tau pathology. These 466 AD genes served as the positive class in our classification framework.

**Protein-Protein Interactions**: 184,742 interactions from Tab3 data, including both physical (98.8%) and genetic (1.2%) interactions. Interactions were filtered to include only high-quality, experimentally validated human proteins (organism ID 9606), with preference for low-throughput experimental systems to ensure reliability. For the negative class, we sampled 932 non-AD genes (1:2 ratio) from proteins in the network, excluding those with known AD associations.

**Chemical Interactions**: 5,134 chemical-protein interactions documenting drug-target relationships, including inhibitor/activator classifications and ATC therapeutic codes for druggability assessment.

**Post-Translational Modifications**: 57,095 PTM records with precise amino acid positions and modification types, capturing the wide range of chemical changes that occur to proteins after translation—such as phosphorylation, acetylation, and ubiquitination—that regulate protein activity, stability, and cellular function.

**PTM Relationships**: 41,260 records linking PTMs to protein interactions, enabling analysis of modification-dependent network dynamics.

---

### Network Construction

To map how Alzheimer's-associated proteins function within shared molecular pathways, we constructed a protein-protein interaction graph using NetworkX (v3.1) from BioGRID interaction data. Self-interactions (0.42% of total) were excluded to focus on inter-protein relationships. The resulting network contained 26,687 nodes and 137,659 edges with a density of 3.87×10⁻⁴. The largest connected component (LCC) comprised 26,684 nodes (>99.9%), indicating a highly interconnected protein interaction landscape that reflects the cooperative, network-level organization of cellular biology. This level of connectivity suggests that AD-associated genes are embedded within broad functional neighborhoods rather than isolated pathways, enabling the downstream identification of key hubs, bottleneck proteins, and modules that may disproportionately influence disease progression.

---

### Topological Data Analysis

**Persistent Homology Framework**

Persistent homology is a central technique in topological data analysis (TDA) that quantifies how topological features of a dataset emerge and persist across different spatial scales. Topological features of importance include connected components (H₀), loops or cycles (H₁), and higher-dimensional cavities. In persistent homology, a filtration consists of a sequence of simplicial complexes constructed by gradually adding edges or simplices according to a chosen parameter. At each step of the filtration, the topology of the complex is recorded using homology groups. The birth (appearance) and death (disappearance or merging) of each feature is tracked as the scale parameter changes. Features that persist across a wide range of scales are considered topologically meaningful and robust to noise, whereas short-lived features are typically interpreted as artifacts. Persistence diagrams and barcodes provide summary representations that encode the lifespan of each topological feature.

**Local Network TDA (Single-Parameter Filtration)**

To generate per-node topological features for classification, we applied persistent homology to the local network neighborhood of each protein. For each target node, we extracted its ego graph—the induced subgraph containing the node and all neighbors within a specified radius *r* (we used *r* = 2, corresponding to 2-hop neighborhoods). We then computed a Vietoris-Rips filtration using shortest-path distances as the metric. Specifically, the distance between any two nodes in the ego graph was defined as their shortest-path distance in the graph topology.

Using the Ripser package, we computed persistent homology up to dimension H₁ (loops) for each ego graph. From the resulting persistence diagrams, we extracted the following TDA-derived features:

- **H₀ and H₁ feature counts**: Number of connected components and loops
- **Total persistence**: Sum of all feature lifetimes
- **Maximum persistence**: Longest-lived feature
- **Lifetime statistics**: Mean, median, and standard deviation of feature lifetimes
- **Barcode entropy**: Information-theoretic measure of topological complexity, computed as the Shannon entropy of normalized lifetimes
- **Infinite feature count**: Components or loops that persist indefinitely

These topological features encode local connectivity patterns, presence of cycles, and mesoscale structural organization around each protein. Proteins with rich topological structure (e.g., high H₁ counts, long persistence) may occupy positions in functional modules or regulatory circuits that are characteristic of disease pathways.

**Complementary Network Features**

In addition to TDA features, we computed traditional graph-theoretic metrics for each node:

- **Degree**: Number of direct neighbors
- **Clustering coefficient**: Local triangle density
- **Ego graph size and density**: Number of nodes and edge density in 1-hop and 2-hop neighborhoods

These features provide complementary information about local connectivity and were combined with TDA features to form comprehensive feature vectors.

---

### Feature Matrix Construction

We constructed a feature matrix where each row corresponds to a protein (either AD-associated or sampled non-AD) and each column represents a computed feature (TDA or network-based). In total, we extracted features for 1,398 proteins: 466 AD genes (positive class) and 932 non-AD genes (negative class), yielding a balanced dataset with a 1:2 positive-to-negative ratio. Each protein was represented by 25 features: 16 TDA-derived features (8 each for H₀ and H₁) and 9 network-based features.

All features were standardized (zero mean, unit variance) prior to model training to ensure fair comparison and prevent features with larger magnitudes from dominating the learning process.

---

### Machine Learning Classifiers

**Model Selection**

We trained and evaluated four supervised learning models to predict whether a gene belongs to the AD-associated annotation set:

1. **Logistic Regression**: A linear model with L2 regularization, providing interpretable coefficients and serving as a baseline classifier.
2. **Random Forest**: An ensemble of decision trees with bagging, capturing nonlinear interactions between features.
3. **Gradient Boosting**: Sequential ensemble method that iteratively corrects misclassifications, offering strong predictive performance.
4. **XGBoost** (if available): Optimized gradient boosting implementation with regularization and efficient handling of class imbalance.

All models were configured to account for class imbalance using class weighting or sample weighting strategies (e.g., `class_weight='balanced'` for scikit-learn models, `scale_pos_weight` for XGBoost).

**Cross-Validation and Evaluation**

We employed stratified 5-fold cross-validation to assess model performance while preserving class proportions in each fold. For each model, we computed the following evaluation metrics:

- **AUROC (Area Under ROC Curve)**: Measures discriminative ability across all classification thresholds
- **AUPRC (Area Under Precision-Recall Curve)**: Emphasizes performance on the minority (positive) class
- **F1-score**: Harmonic mean of precision and recall
- **Precision**: Proportion of true positives among predicted positives
- **Recall**: Proportion of true positives identified

We report mean and standard deviation of each metric across the five folds.

**Feature Importance Analysis**

To identify which topological and network features contribute most to AD gene prediction, we computed feature importance scores using:

- **Tree-based feature importance**: For Random Forest and Gradient Boosting, we extracted mean decrease in impurity (Gini importance)
- **Coefficient magnitude**: For Logistic Regression, we examined absolute values of learned coefficients

We visualized the top 20 most important features and assessed the relative contribution of TDA-derived versus traditional network features.

**Comparative Analysis: TDA vs. Network Features**

To evaluate the added value of topological features, we compared model performance across three feature subsets:

1. **All features**: TDA + network features (full model)
2. **TDA features only**: Only H₀ and H₁ persistence statistics
3. **Network features only**: Traditional graph metrics without TDA

This comparison quantifies whether topological features provide complementary signal beyond what is captured by conventional network analysis.

---

### Parameter Space Topology (Bi-Filtration Analysis)

As a complementary analysis, we explored the global topology of the AD protein network in a two-dimensional parameter space defined by PTM density and chemical interaction count. Each node in the network was assigned coordinates (*x*, *y*) where *x* represents normalized PTM count and *y* represents normalized chemical interaction count. We computed persistent homology on this 2D point cloud using Ripser, revealing how proteins cluster and form cycles in biochemical parameter space.

This global approach captures network-level topology shaped by regulatory and pharmacological properties, whereas the local TDA approach (described above) captures topological features specific to each protein's neighborhood. Together, these analyses provide multi-scale insights into the organizational principles of the AD interactome.

---

### Implementation Details

All analyses were implemented in Python 3.9+ using the following packages:

- **Network analysis**: NetworkX 3.1
- **TDA**: Ripser 0.6.1
- **Machine learning**: scikit-learn 1.3, XGBoost 2.0 (optional)
- **Data manipulation**: pandas 2.0, NumPy 1.24
- **Visualization**: matplotlib 3.7, seaborn 0.12

Code and documentation are available at [repository link]. All random processes (negative class sampling, cross-validation splits) used fixed random seeds for reproducibility.

---

## What Changed from Original Methods?

### Key Updates:

1. **Clarified local vs. global TDA**: Original methods were ambiguous about whether TDA was computed locally (per-node) or globally (whole network). We now explicitly describe both approaches and their distinct purposes.

2. **Specified filtration construction**: Original methods said "shortest-path distances" but didn't clarify how this was used in the Vietoris-Rips complex. We now explicitly state that ego graphs are extracted and distance matrices computed from shortest paths.

3. **Removed RIVET reference**: RIVET is complex and not implemented. We moved bi-filtration to a "parameter space topology" analysis using standard Ripser, which is more honest about what's actually implemented.

4. **Added feature details**: Original methods didn't specify what TDA features were extracted. We now list 16 TDA features (8 per homology dimension) with clear definitions.

5. **Removed vague statements**: Phrases like "lifetimes, Betti curves" were replaced with concrete feature names ("mean persistence", "barcode entropy", etc.).

6. **Computational scope**: Original methods implied computing TDA for all 26K nodes, which is infeasible. We now state we use 1,398 target nodes (466 AD + 932 non-AD).

7. **Feature comparison framework**: Added explicit comparison of TDA-only vs. network-only vs. combined features to demonstrate added value.

8. **Implementation transparency**: Added section listing exact software versions and reproducibility details.

---

## Suggestions for Your Results Section

Your Results section should include:

1. **Dataset Summary**: 
   - Network properties (nodes, edges, density, LCC coverage)
   - AD gene coverage in LCC (should be ~99%)

2. **Feature Distribution Analysis**:
   - Compare TDA feature distributions between AD and non-AD genes
   - Show that AD genes have distinct topological signatures (e.g., higher H1 counts, longer persistence)

3. **Classification Performance**:
   - Table of AUROC, AUPRC, F1 for all models and feature subsets
   - Highlight best-performing model and feature combination
   - Show that TDA features improve performance over network features alone

4. **Feature Importance**:
   - Bar plot of top features
   - Discuss which TDA features are most predictive
   - Interpret biological meaning (e.g., "H1 max persistence captures regulatory modules")

5. **Predicted Candidate Genes**:
   - List top non-AD genes with high AD prediction scores
   - These are potential novel AD-associated genes for follow-up validation

6. **Parameter Space Analysis** (optional):
   - Show global persistence diagram
   - Discuss how AD genes cluster in PTM/chemical space

---

## Tips for Your Presentation

**For December 1/8 presentation, emphasize:**

1. **Motivation**: Why TDA? → Captures higher-order structure missed by degree, centrality, etc.
2. **Method**: Local ego graph → Vietoris-Rips → Persistence features → Classification
3. **Key Result**: TDA features improve AD gene prediction by X% over network features alone
4. **Validation**: Top predicted genes have literature support or pathway enrichment
5. **Impact**: Framework generalizes to other complex diseases

**Visual suggestions:**
- Network diagram with AD genes highlighted
- Example persistence diagram with annotation
- ROC curves comparing feature subsets
- Feature importance bar plot
- Heatmap of predicted probabilities

Good luck! 🚀

