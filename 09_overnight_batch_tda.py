#!/usr/bin/env python3
"""
Overnight Batch TDA Computation

This script runs perturbation TDA analysis on large gene sets with:
- Incremental saving (writes after each gene - safe to kill)
- Resume capability (skips already-computed genes)
- Progress tracking with ETA
- Support for multiple BioGRID disease projects
- Top N candidate genes from the full network

Usage:
    # Run all BioGRID disease projects
    python 09_overnight_batch_tda.py --projects
    
    # Run top 10000 candidate genes from network
    python 09_overnight_batch_tda.py --top 10000
    
    # Run both
    python 09_overnight_batch_tda.py --projects --top 10000
    
    # Resume interrupted run (automatically detects existing files)
    python 09_overnight_batch_tda.py --top 10000  # Just run same command again

Author: Overnight batch processing script
"""

import numpy as np
import pandas as pd
import time
import argparse
import os
import csv
import sys
from datetime import datetime, timedelta

# Force unbuffered output (important for logging to file)
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

def log(msg):
    """Print with immediate flush for real-time logging."""
    print(msg, flush=True)

# Import core TDA functions from 07_perturbation_tda.py
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader

# Load 07_perturbation_tda.py as a module
_spec = spec_from_loader("perturbation_tda", SourceFileLoader("perturbation_tda", "07_perturbation_tda.py"))
perturbation_tda = module_from_spec(_spec)
_spec.loader.exec_module(perturbation_tda)

# Import the functions we need
load_human_interactome = perturbation_tda.load_human_interactome
compute_perturbation_features = perturbation_tda.compute_perturbation_features

# ============================================================================
# CONFIGURATION - All BioGRID projects
# ============================================================================

BIOGRID_PROJECTS = {
    'alzheimers': {
        'name': 'Alzheimer\'s Disease',
        'genes_file': 'data/BIOGRID-PROJECT-alzheimers_disease_project-GENES-5.0.250.projectindex.txt',
        'output_file': 'computed_data/tda_perturbation_alzheimers.csv',
    },
    'autism': {
        'name': 'Autism Spectrum Disorder',
        'genes_file': 'data/BIOGRID-PROJECT-autism_spectrum_disorder_asd_project-LATEST/BIOGRID-PROJECT-autism_spectrum_disorder_asd_project-GENES-5.0.251.projectindex.txt',
        'output_file': 'computed_data/tda_perturbation_autism.csv',
    },
    'autophagy': {
        'name': 'Autophagy',
        'genes_file': 'data/BIOGRID-PROJECT-autophagy_project-LATEST/BIOGRID-PROJECT-autophagy_project-GENES-5.0.251.projectindex.txt',
        'output_file': 'computed_data/tda_perturbation_autophagy.csv',
    },
    'fanconi': {
        'name': 'Fanconi Anemia',
        'genes_file': 'data/BIOGRID-PROJECT-fanconi_anemia_project-LATEST/BIOGRID-PROJECT-fanconi_anemia_project-GENES-5.0.251.projectindex.txt',
        'output_file': 'computed_data/tda_perturbation_fanconi.csv',
    },
    'glioblastoma': {
        'name': 'Glioblastoma',
        'genes_file': 'data/BIOGRID-PROJECT-glioblastoma_project-LATEST/BIOGRID-PROJECT-glioblastoma_project-GENES-5.0.251.projectindex.txt',
        'output_file': 'computed_data/tda_perturbation_glioblastoma.csv',
    },
}

# Output file for top N candidates
TOP_CANDIDATES_OUTPUT = 'computed_data/tda_perturbation_top_candidates.csv'

# CSV column order (for consistent output)
CSV_COLUMNS = [
    'node_id', 'degree', 'ego_original_size', 'ego_actual_size', 'ego_edges',
    'was_subsampled', 'n_simplices', 'time_seconds',
    'H0_with', 'H1_with', 'H2_with', 'H3_with',
    'H0_without', 'H1_without', 'H2_without', 'H3_without',
    'delta_H0', 'delta_H1', 'delta_H2', 'delta_H3',
    'project', 'gene_symbol', 'timestamp'
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_project_genes(project_key):
    """Load genes for a specific BioGRID project."""
    project = BIOGRID_PROJECTS[project_key]
    filepath = project['genes_file']
    
    if not os.path.exists(filepath):
        log(f"  WARNING: File not found: {filepath}")
        return {}, {}
    
    df = pd.read_csv(filepath, sep='\t')
    
    # Filter to human genes only (organism ID 9606)
    df = df[df['ORGANISM ID'] == 9606]
    
    genes = set(df['ENTREZ GENE ID'].values)
    
    # Create gene ID to symbol mapping
    gene_symbols = {}
    for _, row in df.iterrows():
        gene_id = row['ENTREZ GENE ID']
        symbol = row.get('OFFICIAL SYMBOL', str(gene_id))
        gene_symbols[gene_id] = symbol
    
    return genes, gene_symbols


def get_already_computed(output_file):
    """Get set of gene IDs already computed (for resume capability)."""
    if not os.path.exists(output_file):
        return set()
    
    try:
        df = pd.read_csv(output_file)
        return set(df['node_id'].values)
    except:
        return set()


# ============================================================================
# INCREMENTAL BATCH PROCESSING
# ============================================================================

def append_result_to_csv(output_file, result, write_header=False):
    """Append a single result to CSV file (incremental saving)."""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Ensure all columns exist
    for col in CSV_COLUMNS:
        if col not in result:
            result[col] = ''
    
    file_exists = os.path.exists(output_file)
    
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header or not file_exists:
            writer.writeheader()
        writer.writerow({k: result.get(k, '') for k in CSV_COLUMNS})


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def process_genes_incremental(G, genes, gene_symbols, project_name, output_file, max_ego_size=300):
    """
    Process genes with incremental saving.
    
    Key features:
    - Saves after EACH gene (safe to kill anytime)
    - Skips already-computed genes (resume capability)
    - Shows detailed progress with ETA
    """
    # Get already computed genes
    already_computed = get_already_computed(output_file)
    genes_to_process = [g for g in genes if g not in already_computed]
    
    total_genes = len(genes)
    already_done = len(already_computed)
    remaining = len(genes_to_process)
    
    log(f"\n{'='*60}")
    log(f"Processing: {project_name}")
    log(f"{'='*60}")
    log(f"  Total genes: {total_genes}")
    log(f"  Already computed: {already_done}")
    log(f"  Remaining: {remaining}")
    log(f"  Output file: {output_file}")
    
    if remaining == 0:
        log("  ✓ All genes already computed!")
        return already_done
    
    # Filter to genes in network
    genes_in_network = [g for g in genes_to_process if g in G]
    not_in_network = remaining - len(genes_in_network)
    
    if not_in_network > 0:
        log(f"  Not in network: {not_in_network}")
    
    log(f"  Genes to process: {len(genes_in_network)}")
    log(f"\nStarting computation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("-" * 60)
    
    processed = 0
    skipped = 0
    errors = 0
    times = []
    
    start_time = time.time()
    
    for i, gene in enumerate(genes_in_network):
        gene_start = time.time()
        current_num = already_done + i + 1
        
        try:
            features = compute_perturbation_features(
                G, gene, radius=2, max_ego_size=max_ego_size, max_dim=3
            )
            
            if features is None:
                skipped += 1
                continue
            
            # Add metadata
            features['project'] = project_name
            features['gene_symbol'] = gene_symbols.get(gene, str(gene))
            features['timestamp'] = datetime.now().isoformat()
            
            # Save immediately (incremental!)
            write_header = (already_done == 0 and processed == 0)
            append_result_to_csv(output_file, features, write_header=write_header)
            
            processed += 1
            gene_time = time.time() - gene_start
            times.append(gene_time)
            
            # Progress update
            avg_time = np.mean(times[-100:])  # Last 100 for smoother ETA
            remaining_genes = len(genes_in_network) - (i + 1)
            eta_seconds = avg_time * remaining_genes
            
            # Print progress every gene
            symbol = gene_symbols.get(gene, str(gene))
            log(f"[{current_num}/{total_genes}] Gene {gene} ({symbol}) - "
                f"{gene_time:.2f}s - ETA: {format_time(eta_seconds)}")
            
        except Exception as e:
            errors += 1
            log(f"[{current_num}/{total_genes}] ERROR on gene {gene}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    log("-" * 60)
    log(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Processed: {processed}")
    log(f"  Skipped: {skipped}")
    log(f"  Errors: {errors}")
    log(f"  Total time: {format_time(total_time)}")
    log(f"  Avg time per gene: {total_time/max(1, processed):.2f}s")
    
    return already_done + processed


def get_top_candidate_genes(G, n=10000, exclude_genes=None):
    """
    Get top N candidate genes from network, ranked by degree.
    
    Higher degree genes are more likely to be biologically interesting.
    """
    if exclude_genes is None:
        exclude_genes = set()
    
    # Get all genes with their degrees
    gene_degrees = [(node, G.degree(node)) for node in G.nodes() if node not in exclude_genes]
    
    # Sort by degree (descending)
    gene_degrees.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N
    top_genes = [gene for gene, degree in gene_degrees[:n]]
    
    return top_genes


# ============================================================================
# MAIN ENTRY POINTS
# ============================================================================

def run_all_projects(G, max_ego_size=300):
    """Run TDA analysis on all BioGRID disease projects."""
    log("\n" + "=" * 70)
    log("RUNNING ALL BIOGRID DISEASE PROJECTS")
    log("=" * 70)
    
    total_processed = 0
    
    for project_key, project_info in BIOGRID_PROJECTS.items():
        genes, gene_symbols = load_project_genes(project_key)
        
        if not genes:
            log(f"\nSkipping {project_info['name']}: No genes found")
            continue
        
        genes_list = list(genes)
        processed = process_genes_incremental(
            G, genes_list, gene_symbols,
            project_name=project_info['name'],
            output_file=project_info['output_file'],
            max_ego_size=max_ego_size
        )
        total_processed += processed
    
    return total_processed


def run_top_candidates(G, n=10000, max_ego_size=300):
    """Run TDA analysis on top N candidate genes from network."""
    log("\n" + "=" * 70)
    log(f"RUNNING TOP {n:,} CANDIDATE GENES")
    log("=" * 70)
    
    # Collect all disease genes to potentially exclude or mark
    all_disease_genes = set()
    for project_key in BIOGRID_PROJECTS:
        genes, _ = load_project_genes(project_key)
        all_disease_genes.update(genes)
    
    log(f"Known disease genes across all projects: {len(all_disease_genes)}")
    
    # Get top candidates (by degree)
    top_genes = get_top_candidate_genes(G, n=n)
    log(f"Selected top {len(top_genes):,} genes by degree")
    
    # Create simple symbol mapping (use Entrez ID as symbol if not known)
    gene_symbols = {g: str(g) for g in top_genes}
    
    processed = process_genes_incremental(
        G, top_genes, gene_symbols,
        project_name='Top_Candidates',
        output_file=TOP_CANDIDATES_OUTPUT,
        max_ego_size=max_ego_size
    )
    
    return processed


def main():
    parser = argparse.ArgumentParser(
        description='Overnight Batch TDA Computation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 09_overnight_batch_tda.py --projects           # All disease projects
  python 09_overnight_batch_tda.py --top 10000          # Top 10k candidates
  python 09_overnight_batch_tda.py --projects --top 5000  # Both
  
The script saves after each gene, so it's safe to stop and resume later.
Just run the same command again to continue where you left off.
        """
    )
    parser.add_argument('--projects', action='store_true', 
                        help='Run on all BioGRID disease projects')
    parser.add_argument('--top', type=int, default=None,
                        help='Run on top N candidate genes (by degree)')
    parser.add_argument('--max-ego-size', type=int, default=300,
                        help='Maximum ego graph size (default: 300)')
    parser.add_argument('--list-projects', action='store_true',
                        help='List available projects and exit')
    
    args = parser.parse_args()
    
    # List projects mode
    if args.list_projects:
        log("\nAvailable BioGRID Projects:")
        log("-" * 50)
        for key, info in BIOGRID_PROJECTS.items():
            exists = "✓" if os.path.exists(info['genes_file']) else "✗"
            log(f"  [{exists}] {key}: {info['name']}")
            log(f"      File: {info['genes_file']}")
        return
    
    # Default to --projects if nothing specified
    if not args.projects and args.top is None:
        log("No mode specified. Use --projects or --top N")
        log("Run with --help for usage information.")
        return
    
    log("=" * 70)
    log("OVERNIGHT BATCH TDA COMPUTATION")
    log("=" * 70)
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Settings:")
    log(f"  Max ego graph size: {args.max_ego_size}")
    log(f"  Run projects: {args.projects}")
    log(f"  Top candidates: {args.top}")
    log("")
    log("Loading human interactome (this takes 2-5 minutes)...")
    
    # Load network once
    G = load_human_interactome()
    log(f"Network loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    total_processed = 0
    
    # Run projects
    if args.projects:
        total_processed += run_all_projects(G, max_ego_size=args.max_ego_size)
    
    # Run top candidates
    if args.top:
        total_processed += run_top_candidates(G, n=args.top, max_ego_size=args.max_ego_size)
    
    # Final summary
    log("\n" + "=" * 70)
    log("BATCH PROCESSING COMPLETE")
    log("=" * 70)
    log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Total genes processed: {total_processed:,}")
    log("\nOutput files:")
    
    for key, info in BIOGRID_PROJECTS.items():
        if os.path.exists(info['output_file']):
            df = pd.read_csv(info['output_file'])
            log(f"  {info['output_file']}: {len(df)} genes")
    
    if os.path.exists(TOP_CANDIDATES_OUTPUT):
        df = pd.read_csv(TOP_CANDIDATES_OUTPUT)
        log(f"  {TOP_CANDIDATES_OUTPUT}: {len(df)} genes")
    
    log("\n✓ Done!")


if __name__ == "__main__":
    main()

