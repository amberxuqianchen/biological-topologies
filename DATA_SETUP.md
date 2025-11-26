# Data Setup Instructions

This guide explains how to download and prepare the required data files for the TDA analysis.

## Quick Start

```bash
# 1. Download the BioGRID file (see Step 1 below)
# 2. Place it in data/ folder
# 3. Run the extraction script:
python scripts/extract_human_interactions.py
```

---

## Required Data Files

| File | Description | How to Get |
|------|-------------|------------|
| `BIOGRID-HUMAN-*.tab3.txt` | Human protein interactions | Extract from BioGRID (see below) |
| `BIOGRID-PROJECT-alzheimers_*.projectindex.txt` | AD gene list | Download from BioGRID AD project |
| `BIOGRID-PROJECT-alzheimers_*.tab3.txt` | AD interactions | Download from BioGRID AD project |

All files go in the `data/` folder.

---

## Step 1: Download BioGRID Data

### Option A: Download Full Organism File (Recommended)

1. Go to: https://downloads.thebiogrid.org/BioGRID/Release-Archive/

2. Click on the latest version (e.g., `BIOGRID-5.0.251`)

3. Download: `BIOGRID-ORGANISM-5.0.251.tab3.zip` (~200 MB)

4. Unzip and place in `data/` folder:
   ```
    BIOGRID-ALL-5.0.251.tab3.zip
   ```

5. Run the extraction script:
   ```bash
   python scripts/extract_human_interactions.py
   ```

   This will create:
   ```
   data/BIOGRID-HUMAN-5.0.251.tab3.txt
   ```

### Option B: Using Command Line

```bash
cd biological-topologies

# Download
curl -O https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-5.0.251/BIOGRID-ORGANISM-5.0.251.tab3.zip

# Unzip
unzip BIOGRID-ORGANISM-5.0.251.tab3.zip -d data/

# Extract human interactions
python scripts/extract_human_interactions.py

# Clean up (optional - remove the large organism file)
rm data/BIOGRID-ORGANISM-5.0.251.tab3.txt
rm BIOGRID-ORGANISM-5.0.251.tab3.zip
```

---

## Step 2: Download Alzheimer's Disease Project Data

1. Go to: https://thebiogrid.org/project/2

2. Download:
   - **Gene List**: Click "Download Gene List"
   - **Interactions**: Click "Download Interactions"

3. Move both files to `data/` folder

---

## Step 3: Verify Setup

```bash
cd biological-topologies

# Check files exist
ls -la data/BIOGRID-HUMAN*.txt
ls -la data/BIOGRID-PROJECT-alzheimers*.txt

# Quick test
python -c "
import pandas as pd
df = pd.read_csv('data/BIOGRID-HUMAN-5.0.251.tab3.txt', sep='\t', nrows=5)
print('✓ Human interactome loaded successfully!')
print(f'  Columns: {list(df.columns)[:3]}...')
"
```

---

## Expected Data Statistics

After setup, you should have:

| File | Size | Contents |
|------|------|----------|
| BIOGRID-HUMAN-*.tab3.txt | ~600 MB | ~1.5M interactions, ~20K proteins |
| AD gene list | ~50 KB | ~466 AD genes |
| AD interactions | ~5 MB | ~120K AD-related interactions |

---

## File Format Reference

### Human Interactome (TAB3 format)

Key columns used by our scripts:
- `Entrez Gene Interactor A`: Gene ID of protein A
- `Entrez Gene Interactor B`: Gene ID of protein B  
- `Experimental System Type`: Filter for `physical` interactions

### AD Gene List (projectindex format)

Key columns:
- `ENTREZ GENE ID`: Gene identifier
- `CATEGORY VALUES`: Amyloid, Tau, or Both

---

## Troubleshooting

### "File not found" errors
- Make sure files are in `data/` folder
- Check the version number matches your download

### Memory issues
- The human interactome is ~250 MB
- You need at least 8 GB RAM for TDA analysis

### Version mismatch
If you download a different version, update the filename in:
- `perturbation_tda_module.py` (line 22)
- `07_perturbation_tda.py` (line 35)
- `06_data_bias_analysis.py` (lines 67, 308)

Or modify the scripts to auto-detect the version.

---

## Questions?

- BioGRID Documentation: https://wiki.thebiogrid.org/doku.php/downloads
- AD Project Info: https://thebiogrid.org/project/2
