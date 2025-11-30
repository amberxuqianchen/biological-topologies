#!/usr/bin/env python3
"""
Overnight Bifiltration for Disease Projects + Candidates

Similar to 18_overnight_bifiltration.py but for:
1. All BioGRID disease projects (not just AD)
2. Top degree-matched candidates for discovery

Features:
- Incremental saving (resume if interrupted)
- Progress tracking with ETA
- Separate output files per project
"""

import numpy as np
import pandas as pd
import time
import argparse
import os
import csv
import sys
from datetime import datetime, timedelta

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

def log(msg):
    """Print with immediate flush."""
    print(msg, flush=True)

# Import from bifiltration module (same as 18_)
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader

_spec = spec_from_loader("bifiltration_tda", SourceFileLoader("bifiltration_tda", "17_bifiltration_tda.py"))
bifiltration_tda = module_from_spec(_spec)
_spec.loader.exec_module(bifiltration_tda)

load_human_interactome = bifiltration_tda.load_human_interactome
load_ptm_counts = bifiltration_tda.load_ptm_counts
compute_bifiltration_features = bifiltration_tda.compute_bifiltration_features


# ============================================================================
# CONFIGURATION
# ============================================================================

BIOGRID_PROJECTS = {
    'alzheimers': {
        'genes_file': 'data/BIOGRID-PROJECT-alzheimers_disease_project-GENES-5.0.250.projectindex.txt',
        'output_file': 'computed_data/tda_bifiltration_alzheimers.csv',
        'name': "Alzheimer's Disease"
    },
    'autism': {
        'genes_file': 'data/BIOGRID-PROJECT-autism_spectrum_disorder_asd_project-LATEST/BIOGRID-PROJECT-autism_spectrum_disorder_asd_project-GENES-5.0.251.projectindex.txt',
        'output_file': 'computed_data/tda_bifiltration_autism.csv',
        'name': "Autism"
    },
    'autophagy': {
        'genes_file': 'data/BIOGRID-PROJECT-autophagy_project-LATEST/BIOGRID-PROJECT-autophagy_project-GENES-5.0.251.projectindex.txt',
        'output_file': 'computed_data/tda_bifiltration_autophagy.csv',
        'name': "Autophagy"
    },
    'fanconi': {
        'genes_file': 'data/BIOGRID-PROJECT-fanconi_anemia_project-LATEST/BIOGRID-PROJECT-fanconi_anemia_project-GENES-5.0.251.projectindex.txt',
        'output_file': 'computed_data/tda_bifiltration_fanconi.csv',
        'name': "Fanconi Anemia"
    },
    'glioblastoma': {
        'genes_file': 'data/BIOGRID-PROJECT-glioblastoma_project-LATEST/BIOGRID-PROJECT-glioblastoma_project-GENES-5.0.251.projectindex.txt',
        'output_file': 'computed_data/tda_bifiltration_glioblastoma.csv',
        'name': "Glioblastoma"
    },
}

CANDIDATES_OUTPUT = 'computed_data/tda_bifiltration_candidates.csv'
PERTURBATION_CANDIDATES = 'computed_data/tda_perturbation_top_candidates.csv'
PERTURBATION_AD = 'computed_data/tda_perturbation_alzheimers.csv'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def get_already_computed(output_file):
    """Get set of gene IDs already computed."""
    if not os.path.exists(output_file):
        return set()
    try:
        df = pd.read_csv(output_file)
        return set(df['node_id'].values)
    except:
        return set()


def load_project_genes(project_key):
    """Load gene IDs for a BioGRID project."""
    project = BIOGRID_PROJECTS[project_key]
    df = pd.read_csv(project['genes_file'], sep='\t')
    return df['ENTREZ GENE ID'].dropna().astype(int).tolist()


def get_degree_matched_candidates(G, n_candidates=5000):
    """Get candidates with similar degree to AD genes (that have perturbation computed)."""
    
    # Load perturbation candidates
    if not os.path.exists(PERTURBATION_CANDIDATES):
        log(f"ERROR: {PERTURBATION_CANDIDATES} not found!")
        return []
    
    perturb_df = pd.read_csv(PERTURBATION_CANDIDATES)
    candidate_ids = set(perturb_df['node_id'].values)
    log(f"  Candidates with perturbation data: {len(candidate_ids)}")
    
    # Load AD genes to exclude and get their degrees
    ad_df = pd.read_csv(PERTURBATION_AD)
    ad_genes = set(ad_df['node_id'].values)
    ad_degrees = ad_df['degree'].values
    min_deg, max_deg = min(ad_degrees), max(ad_degrees)
    mean_deg = np.mean(ad_degrees)
    
    log(f"  AD gene degrees: min={min_deg}, mean={mean_deg:.0f}, max={max_deg}")
    
    # Filter candidates by degree (within AD range with 20% tolerance)
    tolerance = 0.2
    lower = min_deg * (1 - tolerance)
    upper = max_deg * (1 + tolerance)
    
    matched = []
    for node in candidate_ids:
        if node in G and node not in ad_genes:
            deg = G.degree(node)
            if lower <= deg <= upper:
                matched.append((node, deg))
    
    log(f"  Degree-matched candidates: {len(matched)}")
    
    # Sort by degree (closest to mean first) and take top N
    matched.sort(key=lambda x: abs(x[1] - mean_deg))
    selected = [m[0] for m in matched[:n_candidates]]
    
    log(f"  Selected top {len(selected)} candidates")
    return selected


def process_genes_incremental(G, genes, ptm_counts, output_file, project_name):
    """
    Process genes with incremental saving (same pattern as 18_).
    """
    already_computed = get_already_computed(output_file)
    genes_to_process = [g for g in genes if g not in already_computed and g in G]
    
    total_genes = len(genes)
    already_done = len(already_computed)
    remaining = len(genes_to_process)
    
    log(f"\n  Total genes: {total_genes}")
    log(f"  Already computed: {already_done}")
    log(f"  Remaining: {remaining}")
    log(f"  Output: {output_file}")
    
    if remaining == 0:
        log("  ✓ All genes already computed!")
        return already_done
    
    log(f"\n  Starting at {datetime.now().strftime('%H:%M:%S')}")
    log("-" * 50)
    
    processed = 0
    errors = 0
    times = []
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    for i, gene in enumerate(genes_to_process):
        gene_start = time.time()
        current_num = already_done + i + 1
        
        try:
            # NOTE: Correct order is (G, gene, ptm_counts)
            features = compute_bifiltration_features(G, gene, ptm_counts)
            
            if features is None:
                continue
            
            features['project'] = project_name
            features['timestamp'] = datetime.now().isoformat()
            
            # Append to CSV
            file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0
            
            with open(output_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(features.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(features)
            
            processed += 1
            gene_time = time.time() - gene_start
            times.append(gene_time)
            
            # Progress
            avg_time = np.mean(times[-50:])
            remaining_genes = len(genes_to_process) - (i + 1)
            eta = avg_time * remaining_genes
            
            log(f"  [{current_num}/{total_genes}] Gene {gene}: {gene_time:.2f}s - ETA: {format_time(eta)}")
            
        except Exception as e:
            errors += 1
            log(f"  [{current_num}/{total_genes}] ERROR on {gene}: {e}")
    
    log("-" * 50)
    log(f"  Processed: {processed}, Errors: {errors}")
    
    return already_done + processed


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_project(project_key, G, ptm_counts):
    """Process a single BioGRID project."""
    project = BIOGRID_PROJECTS[project_key]
    output_file = project['output_file']
    
    log(f"\n{'='*60}")
    log(f"Processing: {project['name']}")
    log(f"{'='*60}")
    
    # Check if file exists
    if not os.path.exists(project['genes_file']):
        log(f"  ERROR: File not found: {project['genes_file']}")
        return
    
    # Load genes
    genes = load_project_genes(project_key)
    genes_in_network = [g for g in genes if g in G]
    log(f"  Genes: {len(genes)} total, {len(genes_in_network)} in network")
    
    # Process
    process_genes_incremental(G, genes_in_network, ptm_counts, output_file, project['name'])
    
    log(f"  Finished {project['name']}!")


def process_candidates(G, ptm_counts, n_candidates=5000):
    """Process top degree-matched candidates."""
    output_file = CANDIDATES_OUTPUT
    
    log(f"\n{'='*60}")
    log(f"Processing: Top {n_candidates} Degree-Matched Candidates")
    log(f"{'='*60}")
    
    # Get degree-matched candidates
    candidates = get_degree_matched_candidates(G, n_candidates)
    
    if not candidates:
        log("  No candidates found!")
        return
    
    # Process
    process_genes_incremental(G, candidates, ptm_counts, output_file, "Candidate")
    
    log(f"  Finished candidates!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Overnight bifiltration computation')
    parser.add_argument('--projects', action='store_true', 
                        help='Process all BioGRID projects')
    parser.add_argument('--candidates', action='store_true',
                        help='Process top degree-matched candidates')
    parser.add_argument('--n-candidates', type=int, default=5000,
                        help='Number of candidates to process (default: 5000)')
    parser.add_argument('--project', type=str, choices=list(BIOGRID_PROJECTS.keys()),
                        help='Process a specific project only')
    
    args = parser.parse_args()
    
    # Default: do everything
    if not args.projects and not args.candidates and not args.project:
        args.projects = True
        args.candidates = True
    
    log("="*60)
    log("OVERNIGHT BIFILTRATION COMPUTATION")
    log(f"Started: {datetime.now()}")
    log("="*60)
    
    # Load network
    log("\nLoading human interactome... (may take 2-5 minutes)")
    G = load_human_interactome()
    log(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Load PTM data
    log("\nLoading PTM counts...")
    ptm_counts = load_ptm_counts()
    log(f"PTM data: {len(ptm_counts)} genes with PTM counts")
    
    # Process projects
    if args.project:
        process_project(args.project, G, ptm_counts)
    elif args.projects:
        for project_key in BIOGRID_PROJECTS:
            process_project(project_key, G, ptm_counts)
    
    # Process candidates
    if args.candidates:
        process_candidates(G, ptm_counts, args.n_candidates)
    
    log("\n" + "="*60)
    log(f"COMPLETED: {datetime.now()}")
    log("="*60)


if __name__ == "__main__":
    main()
