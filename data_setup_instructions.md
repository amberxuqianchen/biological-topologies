# Data Setup Instructions

This document explains how to set up the required data files for the TDA analysis.

## Required Data Files

1. **BioGRID Human Interactions** (`data/BIOGRID-HUMAN-5.0.251.tab3.txt`)
   - Full human protein-protein interaction network
   - ~900K interactions, ~20K proteins

2. **BioGRID AD Project Data** (already included in `data/`)
   - `BIOGRID-PROJECT-alzheimers_disease_project-GENES-5.0.250.projectindex.txt`
   - `BIOGRID-PROJECT-alzheimers_disease_project-INTERACTIONS-5.0.250.tab3.txt`
   - `BIOGRID-PROJECT-alzheimers_disease_project-PTM-5.0.250.ptmtab.txt`

## Step 1: Download BioGRID Data

1. Go to [BioGRID Downloads](https://downloads.thebiogrid.org/BioGRID/Release-Archive/)

2. Download the latest release (e.g., `BIOGRID-ALL-5.0.251.tab3.zip`)
   - Direct link (may need updating): https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-5.0.251/BIOGRID-ALL-5.0.251.tab3.zip

3. Extract the ZIP file to get `BIOGRID-ALL-5.0.251.tab3.txt`

4. Place the file in the `data/` folder

## Step 2: Extract Human Interactions

Run the extraction script:

```bash
python scripts/extract_human_interactions.py
```

This will:
- Read the full BioGRID file (~20GB)
- Filter to human-only interactions (Taxonomy ID 9606)
- Save to `data/BIOGRID-HUMAN-5.0.251.tab3.txt`

## Step 3: Verify the Data

Run a quick check:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/BIOGRID-HUMAN-5.0.251.tab3.txt', sep='\t', low_memory=False)
print(f'Rows: {len(df):,}')
print(f'Columns: {list(df.columns)[:5]}...')
physical = df[df['Experimental System Type'] == 'physical']
print(f'Physical interactions: {len(physical):,}')
"
```

Expected output:
- ~1.2M total rows
- ~900K physical interactions

## File Sizes (Approximate)

| File | Size |
|------|------|
| BIOGRID-ALL-*.tab3.txt | ~20 GB |
| BIOGRID-HUMAN-*.tab3.txt | ~600 MB |
| AD project files | ~50 MB total |

## Troubleshooting

### "File not found" errors
- Make sure the file is in the `data/` folder
- Check the exact filename matches what's expected

### Memory issues with full BioGRID
- The extraction script processes the file in chunks
- Should work on machines with 8GB+ RAM

### Different BioGRID version
- Update the version number in the filename
- The script should still work with minor version differences

