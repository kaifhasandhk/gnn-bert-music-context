# Data Splits Directory

This directory contains the train, validation, and test split definitions for the MagnaTagATune dataset.

## Split Configuration
- **Total Subset Clips:** 3,184 audio tracks (Top-50 tag vocabulary)
- **Train Split (80%):** ~2,548 clips
- **Validation Split (10%):** ~318 clips
- **Test Split (10%):** ~318 clips

## Usage
When running `notebooks/demo_context.ipynb` or `src/train.py`, split metadata JSONs (`train.json`, `val.json`, `test.json`) are automatically generated here if not present, pairing audio clip IDs with multi-label target vectors.
