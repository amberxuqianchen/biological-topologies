# BIOGRID Alzheimer's Disease Project Dataset Documentation

This document provides comprehensive documentation for the BIOGRID Alzheimer's Disease Project datasets used in our enhanced TDA-ML pipeline for Alzheimer's gene discovery.

## 📋 Enhanced Pipeline Dataset Overview

Our enhanced TDA pipeline utilizes the BIOGRID Alzheimer's Disease Project's expertly curated datasets, providing:

- **466 curated AD genes** with pathway classifications
- **184,743 AD-specific interactions** with rich experimental metadata
- **Biochemical annotations** including PTMs and chemical interactions
- **Multi-parameter features** for advanced topological analysis

The pipeline integrates 5 complementary datasets:

1. **GENES** - Pathway-classified AD gene annotations
2. **INTERACTIONS** - High-quality protein-protein interactions
3. **CHEMICALS** - Drug-target interaction networks
4. **PTM** - Post-translational modification profiles
5. **PTM-RELATIONSHIPS** - PTM-mediated interaction dynamics

---

## 🧬 1. GENES Dataset
**File:** `BIOGRID-PROJECT-alzheimers_disease_project-GENES-5.0.250.projectindex.txt`

### Purpose
Curated list of 466 Alzheimer's disease-related genes with pathway classifications and interaction statistics.

### Column Structure

| Column | Description | Computational Use |
|--------|-------------|-------------------|
| **#BIOGRID ID** | Unique BioGRID gene identifier | Primary key for cross-dataset joins |
| **ENTREZ GENE ID** | NCBI Entrez gene identifier | External database integration |
| **SYSTEMATIC NAME** | Systematic gene nomenclature | Alternative gene identification |
| **OFFICIAL SYMBOL** | Official gene symbol | Node labels in network analysis |
| **SYNONYMS** | Alternative gene names (pipe-separated) | Gene name normalization |
| **ORGANISM ID** | NCBI taxonomy ID (9606 = human) | Species filtering |
| **ORGANISM NAME** | Species name | Data validation |
| **INTERACTION COUNT** | Total protein interactions | **Node importance/centrality weighting** |
| **PTM COUNT** | Post-translational modifications | **Biochemical activity features** |
| **CHEMICAL INTERACTION COUNT** | Drug/chemical interactions | **Druggability features** |
| **SOURCE** | Data source (BIOGRID) | Data provenance |
| **CATEGORY VALUES** | Pathway classification | **Primary pathway labels for TDA** |
| **CATEGORY IDS** | Category identifiers | Pathway ID mapping |
| **CATEGORY TAGS** | Additional tags | Supplementary annotations |
| **CATEGORY EVIDENCE VALUES** | Evidence descriptions | Quality assessment |
| **CATEGORY EVIDENCE IDS** | Evidence identifiers | Evidence tracking |
| **CATEGORY EVIDENCE CLASSES** | Evidence classification | Evidence weighting |
| **CATEGORY EVIDENCE METHODS** | Evidence methods | Methodology tracking |

### Enhanced Pipeline Applications

#### **Pathway-Specific TDA Analysis**
```python
# Extract pathway gene sets for specialized analysis
amyloid_genes = genes_df[genes_df['CATEGORY VALUES'].str.contains('Amyloid gene set', na=False)]['OFFICIAL SYMBOL'].tolist()
tau_genes = genes_df[genes_df['CATEGORY VALUES'].str.contains('Tau modifier', na=False)]['OFFICIAL SYMBOL'].tolist()
both_pathways = genes_df[genes_df['CATEGORY VALUES'].str.contains('|', na=False)]['OFFICIAL SYMBOL'].tolist()

# Pathway-specific network construction
amyloid_network = build_pathway_network(interactions_df, amyloid_genes)
tau_network = build_pathway_network(interactions_df, tau_genes)
```

#### **Multi-Parameter Persistent Homology**
```python
# Enhanced filtration combining multiple biological parameters
def compute_enhanced_filtration(node):
    degree_param = network.degree(node)
    pathway_param = get_pathway_weight(node)  # From CATEGORY VALUES
    activity_param = genes_df.loc[node, 'INTERACTION COUNT']
    ptm_param = genes_df.loc[node, 'PTM COUNT']
    drug_param = genes_df.loc[node, 'CHEMICAL INTERACTION COUNT']
    
    return (degree_param, pathway_param, activity_param, ptm_param, drug_param)
```

#### **Enhanced Feature Engineering**
- **Pathway stratification:** Use `CATEGORY VALUES` for pathway-specific analysis
- **Biological importance:** Combine `INTERACTION COUNT` + `PTM COUNT` for hub scoring
- **Therapeutic relevance:** `CHEMICAL INTERACTION COUNT` for druggability assessment
- **Cross-pathway analysis:** Identify genes in `both_pathways` as critical connectors

---

## 🔗 2. INTERACTIONS Dataset  
**File:** `BIOGRID-PROJECT-alzheimers_disease_project-INTERACTIONS-5.0.250.tab3.txt`

### Purpose
Comprehensive protein-protein interactions in Tab3 format with rich experimental metadata.

### Column Structure

| Column | Description | Computational Use |
|--------|-------------|-------------------|
| **#BioGRID Interaction ID** | Unique interaction identifier | Edge IDs for network construction |
| **Entrez Gene Interactor A/B** | Gene IDs for both proteins | **Network edge creation** |
| **BioGRID ID Interactor A/B** | BioGRID IDs for both proteins | Cross-reference with GENES dataset |
| **Systematic Name Interactor A/B** | Systematic names | Alternative identification |
| **Official Symbol Interactor A/B** | Gene symbols | **Primary network node labels** |
| **Synonyms Interactor A/B** | Alternative names | Gene name resolution |
| **Experimental System** | Interaction detection method | **Interaction confidence weighting** |
| **Experimental System Type** | Physical vs genetic | **Interaction type classification** |
| **Author** | Publication author | Literature provenance |
| **Publication Source** | PubMed ID | **Publication-based edge weighting** |
| **Organism ID Interactor A/B** | Species taxonomy ID | **Human filtering (9606)** |
| **Throughput** | High/Low throughput | **Data quality assessment** |
| **Score** | Quantitative interaction score | **Edge weight assignment** |
| **Modification** | Interaction modifications | Biochemical context |
| **Qualifications** | Additional interaction info | Context annotation |
| **Tags** | Interaction tags | Classification |
| **Source Database** | Data source | Provenance tracking |
| **SWISS-PROT Accessions A/B** | UniProt IDs | Protein database integration |
| **TREMBL Accessions A/B** | UniProt TrEMBL IDs | Extended protein mapping |
| **REFSEQ Accessions A/B** | RefSeq IDs | Sequence database links |
| **Ontology Term IDs/Names** | GO term annotations | **Functional classification** |
| **Ontology Term Categories** | GO categories | Functional grouping |
| **Organism Name Interactor A/B** | Species names | Data validation |

### Enhanced Pipeline Applications

#### **Multi-Layer Network Construction**
```python
# Enhanced network with rich metadata
def build_enhanced_network(interactions_df):
    G = nx.Graph()
    for _, row in interactions_df.iterrows():
        if pd.notna(row['Official Symbol Interactor A']) and pd.notna(row['Official Symbol Interactor B']):
            G.add_edge(
                row['Official Symbol Interactor A'], 
                row['Official Symbol Interactor B'],
                interaction_id=row['#BioGRID Interaction ID'],
                experimental_system=row['Experimental System'],
                system_type=row['Experimental System Type'],
                confidence=get_confidence_score(row),
                throughput=row['Throughput'],
                pubmed_id=row['Publication Source']
            )
    return G
```

#### **Enhanced Quality Filtering**
```python
# Multi-criteria quality assessment
def apply_quality_filters(interactions_df):
    return interactions_df[
        (interactions_df['Organism ID Interactor A'] == 9606) &  # Human only
        (interactions_df['Organism ID Interactor B'] == 9606) &
        (interactions_df['Experimental System Type'] == 'physical') &  # Physical interactions
        (interactions_df['Throughput'].isin(['Low Throughput', 'Both']))  # High-quality experiments
    ]
```

#### **Pathway-Specific TDA Networks**
```python
# Build pathway-specific networks for comparative TDA
def build_pathway_networks(interactions_df, gene_categories):
    networks = {}
    
    # Amyloid-specific network
    amyloid_interactions = filter_pathway_interactions(interactions_df, gene_categories['amyloid_genes'])
    networks['amyloid'] = build_network_from_interactions(amyloid_interactions)
    
    # Tau-specific network  
    tau_interactions = filter_pathway_interactions(interactions_df, gene_categories['tau_genes'])
    networks['tau'] = build_network_from_interactions(tau_interactions)
    
    # Cross-pathway network (both pathways)
    cross_interactions = filter_cross_pathway_interactions(interactions_df, gene_categories)
    networks['cross_pathway'] = build_network_from_interactions(cross_interactions)
    
    return networks
```

#### **Enhanced Edge Weighting for TDA**
```python
# Sophisticated edge weighting for persistent homology
def compute_edge_weights(row):
    base_weight = 1.0
    
    # Experimental confidence weighting
    confidence_multiplier = get_experimental_confidence(row['Experimental System'])
    
    # Publication support weighting
    literature_weight = get_literature_support_weight(row['Publication Source'])
    
    # Throughput quality adjustment
    quality_multiplier = 1.2 if row['Throughput'] == 'Low Throughput' else 1.0
    
    return base_weight * confidence_multiplier * literature_weight * quality_multiplier
```

---

## 💊 3. CHEMICALS Dataset
**File:** `BIOGRID-PROJECT-alzheimers_disease_project-CHEMICALS-5.0.250.chemtab.txt`

### Purpose
Chemical-protein interactions including drugs, metabolites, and small molecules.

### Column Structure

| Column | Description | Computational Use |
|--------|-------------|-------------------|
| **#BioGRID Chemical Interaction ID** | Unique chemical interaction ID | Edge identification |
| **BioGRID Gene ID** | Target gene BioGRID ID | **Drug target identification** |
| **Entrez Gene ID** | Target gene Entrez ID | Gene mapping |
| **Official Symbol** | Target gene symbol | **Drug-gene network nodes** |
| **Action** | Drug action (inhibitor/activator) | **Edge direction/type** |
| **Interaction Type** | Interaction mechanism | Mechanistic classification |
| **Author** | Publication author | Literature source |
| **Pubmed ID** | Publication reference | Evidence tracking |
| **BioGRID Chemical ID** | Chemical identifier | **Drug node identification** |
| **Chemical Name** | Drug/chemical name | **Chemical node labels** |
| **Chemical Synonyms** | Alternative names | Name normalization |
| **Chemical Brands** | Commercial names | Drug identification |
| **Chemical Source** | Database source | Data provenance |
| **Chemical Source ID** | External database ID | Cross-referencing |
| **Molecular Formula** | Chemical formula | **Molecular property features** |
| **Chemical Type** | Chemical classification | **Drug type features** |
| **ATC Codes** | Anatomical Therapeutic Chemical codes | **Drug classification** |
| **CAS Number** | Chemical registry number | Unique identification |
| **Method** | Experimental method | Confidence assessment |
| **InChIKey** | Chemical structure key | **Structure-based features** |

### Enhanced Pipeline Applications

#### **Multiplex Network Construction**
```python
# Build multiplex network combining protein-protein and drug-target layers
def build_multiplex_network(interactions_df, chemicals_df):
    # Layer 1: Protein-protein interactions
    ppi_network = build_enhanced_network(interactions_df)
    
    # Layer 2: Drug-target interactions
    drug_network = nx.Graph()
    for _, row in chemicals_df.iterrows():
        drug_network.add_edge(
            f"DRUG_{row['Chemical Name']}", 
            row['Official Symbol'],
            action=row['Action'],
            interaction_type=row['Interaction Type'],
            atc_code=row['ATC Codes'],
            evidence=row['Pubmed ID']
        )
    
    # Combine into multiplex structure
    multiplex = nx.compose(ppi_network, drug_network)
    return multiplex, ppi_network, drug_network
```

#### **Enhanced Druggability Analysis**
```python
# Advanced druggability scoring for gene prioritization
def compute_druggability_features(chemicals_df, genes_df):
    druggability_features = {}
    
    for gene in genes_df['OFFICIAL SYMBOL']:
        gene_drugs = chemicals_df[chemicals_df['Official Symbol'] == gene]
        
        druggability_features[gene] = {
            'drug_count': len(gene_drugs),
            'inhibitor_count': len(gene_drugs[gene_drugs['Action'] == 'inhibitor']),
            'activator_count': len(gene_drugs[gene_drugs['Action'] == 'activator']),
            'atc_diversity': gene_drugs['ATC Codes'].nunique(),
            'drug_evidence': gene_drugs['Pubmed ID'].nunique(),
            'is_druggable': len(gene_drugs) > 0
        }
    
    return druggability_features
```

#### **Pathway-Drug Integration**
```python
# Analyze drug targeting patterns across AD pathways
def analyze_pathway_drug_targeting(chemicals_df, gene_categories):
    pathway_drug_analysis = {}
    
    # Amyloid pathway drug targets
    amyloid_drugs = chemicals_df[chemicals_df['Official Symbol'].isin(gene_categories['amyloid_genes'])]
    pathway_drug_analysis['amyloid'] = {
        'target_count': amyloid_drugs['Official Symbol'].nunique(),
        'drug_count': amyloid_drugs['Chemical Name'].nunique(),
        'inhibitor_ratio': len(amyloid_drugs[amyloid_drugs['Action'] == 'inhibitor']) / len(amyloid_drugs)
    }
    
    # Tau pathway drug targets
    tau_drugs = chemicals_df[chemicals_df['Official Symbol'].isin(gene_categories['tau_genes'])]
    pathway_drug_analysis['tau'] = {
        'target_count': tau_drugs['Official Symbol'].nunique(),
        'drug_count': tau_drugs['Chemical Name'].nunique(), 
        'inhibitor_ratio': len(tau_drugs[tau_drugs['Action'] == 'inhibitor']) / len(tau_drugs)
    }
    
    return pathway_drug_analysis
```

#### **TDA-Enhanced Drug Discovery**
```python
# Use topological features to identify promising drug targets
def identify_topological_drug_targets(tda_features, druggability_features):
    candidate_targets = []
    
    for gene in tda_features.keys():
        if gene in druggability_features:
            # Combine topological importance with druggability
            topo_score = tda_features[gene]['centrality_persistence'] 
            drug_score = druggability_features[gene]['drug_count']
            
            combined_score = topo_score * (1 + drug_score)
            
            candidate_targets.append({
                'gene': gene,
                'topological_score': topo_score,
                'druggability_score': drug_score,
                'combined_score': combined_score
            })
    
    return sorted(candidate_targets, key=lambda x: x['combined_score'], reverse=True)
```

---

## 🔬 4. PTM Dataset
**File:** `BIOGRID-PROJECT-alzheimers_disease_project-PTM-5.0.250.ptmtab.txt`

### Purpose
Post-translational modifications with precise location and modification type information.

### Column Structure

| Column | Description | Computational Use |
|--------|-------------|-------------------|
| **#PTM ID** | Unique PTM identifier | PTM tracking |
| **Entrez Gene ID** | Modified gene ID | Gene mapping |
| **BioGRID ID** | BioGRID gene identifier | Cross-dataset linking |
| **Official Symbol** | Gene symbol | **PTM target identification** |
| **Sequence** | Protein sequence context | Sequence analysis |
| **Refseq ID** | Reference sequence ID | Sequence validation |
| **Position** | Modification position | **Spatial PTM features** |
| **Post Translational Modification** | PTM type | **Biochemical activity classification** |
| **Residue** | Modified amino acid | **Residue-specific features** |
| **Author** | Publication author | Literature source |
| **Pubmed ID** | Publication ID | Evidence quality |
| **Organism ID** | Species taxonomy | Species filtering |
| **Organism Name** | Species name | Data validation |
| **Has Relationships** | Related PTMs exist | **PTM network connectivity** |
| **Notes** | Additional information | Context annotation |
| **Source Database** | Data source | Provenance |

### Key Computational Applications

#### **PTM-Based Gene Features**
```python
# PTM density features
ptm_counts = ptm_df.groupby('Official Symbol').size()
ptm_types = ptm_df.groupby('Official Symbol')['Post Translational Modification'].nunique()

# Biochemical activity scoring
highly_modified_genes = ptm_counts[ptm_counts > ptm_counts.quantile(0.8)].index
```

#### **PTM Type Classification**
```python
# Major PTM types for AD
ptm_type_features = {
    'phosphorylation': ptm_df[ptm_df['Post Translational Modification'] == 'Phosphorylation'],
    'ubiquitination': ptm_df[ptm_df['Post Translational Modification'] == 'Ubiquitination'],
    'acetylation': ptm_df[ptm_df['Post Translational Modification'] == 'Acetylation']
}
```

#### **Spatial PTM Analysis**
- **Position clustering:** Group PTMs by sequence position
- **Domain analysis:** Map PTMs to protein domains
- **Functional regions:** Identify highly modified regions

#### **PTM Network Features**
- **PTM co-occurrence:** Proteins with multiple PTM types
- **PTM propagation:** PTMs affecting protein interactions
- **Regulatory networks:** PTM-mediated signaling cascades

---

## 🔄 5. PTM-RELATIONSHIPS Dataset
**File:** `BIOGRID-PROJECT-alzheimers_disease_project-PTM-RELATIONSHIPS-5.0.250.ptmrel.txt`

### Purpose
Relationships between post-translational modifications and protein interactions.

### Key Computational Applications

#### **PTM-Interaction Integration**
- **PTM-dependent interactions:** Interactions requiring specific PTMs
- **PTM regulatory networks:** How PTMs control interaction dynamics
- **Temporal analysis:** PTM-interaction cascades

---

## 🧮 Computational Workflow Integration

### 1. **Multi-Parameter TDA Pipeline**

```python
# Primary filtration: Network topology
degree_filtration = [network.degree(node) for node in nodes]

# Secondary filtration: Pathway membership  
pathway_filtration = [
    2 if node in both_pathways else
    1.5 if node in amyloid_genes else  
    1 if node in tau_genes else 0
    for node in nodes
]

# Tertiary filtration: Biochemical activity
ptm_filtration = [ptm_counts.get(node, 0) for node in nodes]

# Quaternary filtration: Druggability
drug_filtration = [druggable_genes.get(node, 0) for node in nodes]
```

### 2. **Feature Engineering Matrix**

| Feature Category | Data Source | Computation |
|------------------|-------------|-------------|
| **Graph Features** | INTERACTIONS | Centrality, clustering, paths |
| **Pathway Features** | GENES | Binary/categorical pathway membership |
| **PTM Features** | PTM | Count, type diversity, position |
| **Drug Features** | CHEMICALS | Target count, action types |
| **Literature Features** | All datasets | Publication count, evidence quality |
| **TDA Features** | Computed | Persistence, entropy, lifetimes |

### 3. **Quality Control Filters**

```python
# High-quality interactions only
quality_filter = (
    (interactions_df['Organism ID Interactor A'] == 9606) &
    (interactions_df['Organism ID Interactor B'] == 9606) &
    (interactions_df['Experimental System Type'] == 'physical') &
    (interactions_df['Throughput'] == 'Low Throughput')
)
```

### 4. **Cross-Dataset Integration**

```python
# Join all datasets
integrated_features = genes_df.merge(
    ptm_summary, on='Official Symbol', how='left'
).merge(
    drug_summary, on='Official Symbol', how='left'  
).merge(
    interaction_summary, on='Official Symbol', how='left'
)
```

---

## 📊 Summary: Dataset Utility for TDA

| **Analysis Type** | **Primary Dataset** | **Key Columns** |
|-------------------|--------------------|--------------| 
| **Network Construction** | INTERACTIONS | Official Symbol A/B, Experimental System |
| **Pathway Analysis** | GENES | Category Values, Interaction Count |
| **Drug Discovery** | CHEMICALS | Action, Chemical Name, ATC Codes |
| **Biochemical Features** | PTM | PTM Type, Position, Residue |
| **Multi-Parameter TDA** | ALL | Combined features from all datasets |
| **Quality Assessment** | INTERACTIONS | Throughput, Experimental System Type |
| **Literature Validation** | ALL | Pubmed ID, Author, Evidence |

This comprehensive dataset collection provides rich, multi-dimensional data for sophisticated topological data analysis of Alzheimer's disease networks, enabling novel insights into disease mechanisms and therapeutic targets.