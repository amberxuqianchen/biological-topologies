#!/usr/bin/env python3
"""
Extract Human Interactions from BioGRID

This script extracts human protein-protein interactions from the full
BioGRID organism file and saves them to a separate file.

Usage:
    python scripts/extract_human_interactions.py

The script will:
1. Look for BIOGRID-ORGANISM-*.tab3.txt in data/
2. Extract only Homo sapiens interactions
3. Save to data/BIOGRID-HUMAN-*.tab3.txt

Download the source file from:
https://downloads.thebiogrid.org/BioGRID/Release-Archive/
"""

import os
import glob
import pandas as pd
import sys


def find_biogrid_file():
    """Find the BioGRID organism file in the data directory."""
    # Look for the full organism file
    patterns = [
        'data/BIOGRID-ORGANISM-*.tab3.txt',
        'data/BIOGRID-ALL-*.tab3.txt',
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    
    return None


def extract_human_interactions(input_file, output_file=None):
    """
    Extract human interactions from the full BioGRID file.
    
    Parameters:
    -----------
    input_file : str
        Path to the full BioGRID organism file
    output_file : str or None
        Path to save extracted human interactions
        If None, will auto-generate based on input filename
    
    Returns:
    --------
    str : Path to the output file
    """
    print(f"Loading {input_file}...")
    print("(This may take a minute for large files)")
    
    # Read the file
    df = pd.read_csv(input_file, sep='\t', low_memory=False)
    
    print(f"Total interactions loaded: {len(df):,}")
    
    # Check what organism columns exist
    org_cols = [col for col in df.columns if 'Organism' in col]
    print(f"Organism columns: {org_cols}")
    
    # Filter for human interactions (Homo sapiens, taxonomy ID 9606)
    # Both interactors must be human
    if 'Organism ID Interactor A' in df.columns:
        human_df = df[
            (df['Organism ID Interactor A'] == 9606) & 
            (df['Organism ID Interactor B'] == 9606)
        ]
    elif 'Organism Interactor A' in df.columns:
        human_df = df[
            (df['Organism Interactor A'] == 'Homo sapiens') & 
            (df['Organism Interactor B'] == 'Homo sapiens')
        ]
    else:
        print("ERROR: Could not find organism column!")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    print(f"Human interactions: {len(human_df):,}")
    
    # Generate output filename if not provided
    if output_file is None:
        # Extract version from input filename
        basename = os.path.basename(input_file)
        if 'BIOGRID-ORGANISM-' in basename:
            version = basename.replace('BIOGRID-ORGANISM-', '').replace('.tab3.txt', '')
            # Version might be like "5.0.251" - extract just that
            parts = version.split('-')
            if len(parts) > 1:
                version = parts[-1]  # Take last part which should be version
        elif 'BIOGRID-ALL-' in basename:
            version = basename.replace('BIOGRID-ALL-', '').replace('.tab3.txt', '')
        else:
            version = 'extracted'
        
        output_file = f'data/BIOGRID-HUMAN-{version}.tab3.txt'
    
    # Save
    print(f"Saving to {output_file}...")
    human_df.to_csv(output_file, sep='\t', index=False)
    
    # Print stats
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"\nDone!")
    print(f"  Output file: {output_file}")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Interactions: {len(human_df):,}")
    
    # Count unique genes
    if 'Entrez Gene Interactor A' in human_df.columns:
        genes_a = set(human_df['Entrez Gene Interactor A'].dropna().astype(int))
        genes_b = set(human_df['Entrez Gene Interactor B'].dropna().astype(int))
        all_genes = genes_a | genes_b
        print(f"  Unique genes: {len(all_genes):,}")
    
    return output_file


def main():
    print("="*60)
    print("EXTRACT HUMAN INTERACTIONS FROM BIOGRID")
    print("="*60)
    
    # Check if human file already exists
    existing = glob.glob('data/BIOGRID-HUMAN-*.tab3.txt')
    if existing:
        print(f"\nHuman interactions file already exists: {existing[0]}")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Exiting.")
            return
    
    # Find input file
    input_file = find_biogrid_file()
    
    if input_file is None:
        print("\nERROR: No BioGRID file found!")
        print("\nPlease download the BioGRID organism file:")
        print("1. Go to: https://downloads.thebiogrid.org/BioGRID/Release-Archive/")
        print("2. Download: BIOGRID-ORGANISM-<version>.tab3.zip")
        print("3. Unzip and place in data/ folder")
        print("\nOr download the ALL file:")
        print("   BIOGRID-ALL-<version>.tab3.zip")
        sys.exit(1)
    
    print(f"\nFound: {input_file}")
    
    # Extract human interactions
    output_file = extract_human_interactions(input_file)
    
    if output_file:
        print("\n" + "="*60)
        print("SUCCESS! You can now run the TDA analysis scripts.")
        print("="*60)


if __name__ == "__main__":
    main()

