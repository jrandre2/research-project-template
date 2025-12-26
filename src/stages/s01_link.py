#!/usr/bin/env python3
"""
Stage 01: Record Linkage

Purpose: Link records across multiple data sources.

Input Files
-----------
- data_work/data_raw.parquet
- data_work/<additional_source>.parquet

Output Files
------------
- data_work/data_linked.parquet

Usage
-----
    python src/pipeline.py link_records
"""
from __future__ import annotations

from pathlib import Path

# Define paths
DATA_DIR = Path('data_work')
INPUT_FILE = DATA_DIR / 'data_raw.parquet'
OUT_FILE = DATA_DIR / 'data_linked.parquet'


def main():
    """Execute record linkage pipeline."""
    print("Stage 01: Record Linkage")
    print("-" * 40)

    # TODO: Implement your linkage logic here
    # Example:
    # import pandas as pd
    # df = pd.read_parquet(INPUT_FILE)
    # df_other = pd.read_parquet(DATA_DIR / 'other_source.parquet')
    # df_linked = df.merge(df_other, on='key_column', how='left')
    # df_linked.to_parquet(OUT_FILE, index=False)

    print(f"Output: {OUT_FILE}")
    print("Stage 01 complete.")


if __name__ == '__main__':
    main()
