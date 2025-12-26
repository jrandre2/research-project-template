#!/usr/bin/env python3
"""
Stage 00: Data Ingestion

Purpose: Load and preprocess raw data files.

Input Files
-----------
- data_raw/<your_data_file>.csv

Output Files
------------
- data_work/data_raw.parquet

Usage
-----
    python src/pipeline.py ingest_data
"""
from __future__ import annotations

from pathlib import Path

# Define paths
RAW_DIR = Path('data_raw')
OUT_DIR = Path('data_work')
OUT_FILE = OUT_DIR / 'data_raw.parquet'


def main():
    """Execute data ingestion pipeline."""
    print("Stage 00: Data Ingestion")
    print("-" * 40)

    # Ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: Implement your data loading logic here
    # Example:
    # import pandas as pd
    # df = pd.read_csv(RAW_DIR / 'your_data.csv')
    # df = clean_data(df)
    # df.to_parquet(OUT_FILE, index=False)

    print(f"Output: {OUT_FILE}")
    print("Stage 00 complete.")


if __name__ == '__main__':
    main()
