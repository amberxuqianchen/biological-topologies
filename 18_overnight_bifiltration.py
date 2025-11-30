#!/usr/bin/env python3
"""
Overnight Bifiltration TDA Computation

Runs bifiltration TDA analysis on AD genes + degree-matched background.
Similar to 09_overnight_batch_tda.py but for bifiltration features.

Features:
- Incremental saving (writes after each gene - safe to kill)
- Resume capability (skips already-computed genes)
- Progress tracking with ETA

Usage:
    python 18_overnight_bifiltration.py              # Run on AD + background
    python 18_overnight_bifiltration.py --ad-only    # Run on AD genes only
    python 18_overnight_bifiltration.py --resume     # Resume interrupted run
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

# Import from bifiltration module
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

OUTPUT_FILE = 'computed_data/tda_bifiltration_features.csv'


# ============================================================================
# DATA LOADING
# ============================================================================

def get_genes_to_process():
    """Get AD genes and degree-matched background (same as original TDA)."""
    log("Loading gene lists...")
    
    # Load existing perturbation data to get the same genes
    ad_df = pd.read_csv('computed_data/tda_perturbation_alzheimers.csv')
    bg_df = pd.read_csv('computed_data/tda_perturbation_top_candidates.csv')
    
    # Get degree-matched background (same logic as before)
    ad_ids = set(ad_df['node_id'].values)
    bg_df = bg_df[~bg_df['node_id'].isin(ad_ids)]
    
    np.random.seed(42)  # Same seed as original
    matched_bg = []
    for degree in ad_df['degree'].values:
        tolerance = max(5, degree * 0.2)
        candidates = bg_df[(bg_df['degree'] >= degree - tolerance) & 
                          (bg_df['degree'] <= degree + tolerance)]
        if len(candidates) > 0:
            sampled = candidates.sample(n=1, random_state=len(matched_bg))
            matched_bg.append(sampled.iloc[0]['node_id'])
    
    matched_bg = list(set(matched_bg))
    ad_genes = list(ad_df['node_id'].values)
    
    log(f"  AD genes: {len(ad_genes)}")
    log(f"  Matched background: {len(matched_bg)}")
    
    return ad_genes, matched_bg


def get_already_computed(output_file):
    """Get set of gene IDs already computed."""
    if not os.path.exists(output_file):
        return set()
    try:
        df = pd.read_csv(output_file)
        return set(df['node_id'].values)
    except:
        return set()


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_genes_incremental(G, genes, is_ad_list, ptm_counts, output_file):
    """
    Process genes with incremental saving.
    """
    already_computed = get_already_computed(output_file)
    genes_to_process = [(g, is_ad) for g, is_ad in zip(genes, is_ad_list) 
                        if g not in already_computed and g in G]
    
    total_genes = len(genes)
    already_done = len(already_computed)
    remaining = len(genes_to_process)
    
    log(f"\n{'='*60}")
    log(f"BIFILTRATION TDA PROCESSING")
    log(f"{'='*60}")
    log(f"  Total genes: {total_genes}")
    log(f"  Already computed: {already_done}")
    log(f"  Remaining: {remaining}")
    log(f"  Output: {output_file}")
    
    if remaining == 0:
        log("  ✓ All genes already computed!")
        return already_done
    
    log(f"\nStarting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("-" * 60)
    
    processed = 0
    errors = 0
    times = []
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    for i, (gene, is_ad) in enumerate(genes_to_process):
        gene_start = time.time()
        current_num = already_done + i + 1
        
        try:
            features = compute_bifiltration_features(G, gene, ptm_counts)
            
            if features is None:
                continue
            
            features['is_ad'] = is_ad
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
            
            label = "AD" if is_ad else "BG"
            log(f"[{current_num}/{total_genes}] Gene {gene} ({label}) - "
                f"{gene_time:.2f}s - ETA: {format_time(eta)}")
            
        except Exception as e:
            errors += 1
            log(f"[{current_num}/{total_genes}] ERROR on {gene}: {e}")
    
    log("-" * 60)
    log(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Processed: {processed}")
    log(f"  Errors: {errors}")
    
    return already_done + processed


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Overnight Bifiltration TDA Computation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 18_overnight_bifiltration.py            # AD + background
  python 18_overnight_bifiltration.py --ad-only  # AD genes only
  
Safe to stop and resume - just run the same command again.
        """
    )
    parser.add_argument('--ad-only', action='store_true',
                        help='Run only on AD genes (skip background)')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE,
                        help=f'Output file (default: {OUTPUT_FILE})')
    
    args = parser.parse_args()
    
    log("="*60)
    log("OVERNIGHT BIFILTRATION TDA")
    log("="*60)
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Output: {args.output}")
    log(f"Mode: {'AD only' if args.ad_only else 'AD + Background'}")
    log("")
    
    # Load network
    log("Loading human interactome (this takes 2-5 minutes)...")
    G = load_human_interactome()
    log(f"  Network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Load PTM data
    ptm_counts = load_ptm_counts()
    
    # Get genes
    ad_genes, bg_genes = get_genes_to_process()
    
    if args.ad_only:
        all_genes = ad_genes
        is_ad_list = [True] * len(ad_genes)
    else:
        all_genes = ad_genes + bg_genes
        is_ad_list = [True] * len(ad_genes) + [False] * len(bg_genes)
    
    # Process
    total = process_genes_incremental(G, all_genes, is_ad_list, ptm_counts, args.output)
    
    # Summary
    log("\n" + "="*60)
    log("COMPLETE")
    log("="*60)
    log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Total genes in output: {total}")
    log(f"Output file: {args.output}")
    log("\n✓ Done!")


if __name__ == "__main__":
    main()

