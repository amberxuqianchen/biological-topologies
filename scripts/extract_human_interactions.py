#!/usr/bin/env python3
"""
Extract Human Interactions from BioGRID

This script extracts human-only protein interactions from the full BioGRID
database and saves them to a separate file.

Usage:
    python scripts/extract_human_interactions.py
    python scripts/extract_human_interactions.py --input data/BIOGRID-ALL-5.0.251.tab3.txt

The script processes the file in chunks to handle the large file size (~20GB).
"""

import pandas as pd
import argparse
import os
import sys
from glob import glob


def find_biogrid_all_file(data_dir='data'):
    """Find the BIOGRID-ALL file in the data directory."""
    patterns = [
        os.path.join(data_dir, 'BIOGRID-ALL-*.tab3.txt'),
        os.path.join(data_dir, 'BIOGRID-ALL-*.tab3'),
    ]
    
    for pattern in patterns:
        matches = glob(pattern)
        if matches:
            return matches[0]
    
    return None


def extract_human_interactions(input_file, output_file, chunk_size=500000):
    """
    Extract human-only interactions from BioGRID.
    
    Parameters:
    -----------
    input_file : str
        Path to BIOGRID-ALL file
    output_file : str
        Path to output human-only file
    chunk_size : int
        Number of rows to process at a time
    """
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Processing in chunks of {chunk_size:,} rows...")
    
    # Human taxonomy ID
    HUMAN_TAX_ID = 9606
    
    # Process in chunks
    chunks_processed = 0
    total_rows = 0
    human_rows = 0
    
    # First chunk - write with header
    first_chunk = True
    
    try:
        for chunk in pd.read_csv(input_file, sep='\t', chunksize=chunk_size, 
                                  low_memory=False, on_bad_lines='skip'):
            chunks_processed += 1
            total_rows += len(chunk)
            
            # Filter to human-human interactions
            # Both interactors must be human
            human_chunk = chunk[
                (chunk['Organism ID Interactor A'] == HUMAN_TAX_ID) &
                (chunk['Organism ID Interactor B'] == HUMAN_TAX_ID)
            ]
            
            human_rows += len(human_chunk)
            
            # Write to output
            if len(human_chunk) > 0:
                if first_chunk:
                    human_chunk.to_csv(output_file, sep='\t', index=False, mode='w')
                    first_chunk = False
                else:
                    human_chunk.to_csv(output_file, sep='\t', index=False, mode='a', header=False)
            
            # Progress update
            print(f"  Chunk {chunks_processed}: {total_rows:,} total, {human_rows:,} human", end='\r')
            
    except Exception as e:
        print(f"\nError processing file: {e}")
        return False
    
    print(f"\n\nDone!")
    print(f"  Total rows processed: {total_rows:,}")
    print(f"  Human interactions: {human_rows:,}")
    print(f"  Output file: {output_file}")
    
    # Verify output
    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  Output size: {size_mb:.1f} MB")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Extract human interactions from BioGRID')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to BIOGRID-ALL file (auto-detected if not specified)')
    parser.add_argument('--output', type=str, default='data/BIOGRID-HUMAN-5.0.251.tab3.txt',
                        help='Path to output file')
    parser.add_argument('--chunk-size', type=int, default=500000,
                        help='Chunk size for processing')
    args = parser.parse_args()
    
    print("="*60)
    print("EXTRACT HUMAN INTERACTIONS FROM BIOGRID")
    print("="*60)
    
    # Find input file
    if args.input:
        input_file = args.input
    else:
        input_file = find_biogrid_all_file()
        if input_file is None:
            print("\nError: Could not find BIOGRID-ALL file in data/ directory")
            print("\nPlease either:")
            print("  1. Download from https://downloads.thebiogrid.org/BioGRID/")
            print("  2. Place BIOGRID-ALL-*.tab3.txt in the data/ folder")
            print("  3. Specify the path with --input")
            sys.exit(1)
    
    if not os.path.exists(input_file):
        print(f"\nError: Input file not found: {input_file}")
        sys.exit(1)
    
    # Run extraction
    success = extract_human_interactions(
        input_file, 
        args.output, 
        chunk_size=args.chunk_size
    )
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS! Human interactions extracted.")
        print("="*60)
    else:
        print("\nExtraction failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

