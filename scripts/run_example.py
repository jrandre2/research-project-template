#!/usr/bin/env python3
"""
Example Analysis Script: Descriptive Statistics

Purpose: Demonstrate the structure for extended analysis scripts.
Input:   data_work/panel.parquet
Output:  data_work/exploratory/descriptive_stats.csv

Usage:
    python scripts/run_example.py
    python scripts/run_example.py --output custom_output.csv

Notes:
    This is an extended analysis script, separate from the core pipeline.
    Results should be validated before incorporating into the main manuscript.

    Use this as a template for your own analysis scripts. Key features:
    - Imports from the src/ directory
    - Reads from standard data locations
    - Writes to exploratory output directory
    - Includes progress printing and error handling
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import from pipeline modules
try:
    from config import DATA_WORK_DIR, FIGURES_DIR
except ImportError:
    # Fallback if config not available
    DATA_WORK_DIR = Path(__file__).parent.parent / 'data_work'
    FIGURES_DIR = Path(__file__).parent.parent / 'figures'

from utils.helpers import load_data, ensure_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Example analysis script - compute descriptive statistics'
    )
    parser.add_argument(
        '--input', '-i',
        default=None,
        help='Input parquet file (default: data_work/panel.parquet)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output CSV file (default: data_work/exploratory/descriptive_stats.csv)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed output'
    )
    return parser.parse_args()


def compute_descriptive_stats(df):
    """
    Compute descriptive statistics for a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data

    Returns
    -------
    pandas.DataFrame
        Summary statistics
    """
    import pandas as pd

    # Numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns

    stats = []
    for col in numeric_cols:
        series = df[col].dropna()
        stats.append({
            'variable': col,
            'n': len(series),
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'p25': series.quantile(0.25),
            'median': series.median(),
            'p75': series.quantile(0.75),
            'max': series.max(),
            'missing': df[col].isna().sum(),
            'missing_pct': df[col].isna().mean() * 100,
        })

    return pd.DataFrame(stats)


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Example Analysis: Descriptive Statistics")
    print("=" * 60)

    # Determine input file
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = DATA_WORK_DIR / 'panel.parquet'

    # Check input exists
    if not input_path.exists():
        print(f"\nERROR: Input file not found: {input_path}")
        print("Run the pipeline first to generate panel data:")
        print("  python src/pipeline.py ingest_data")
        print("  python src/pipeline.py build_panel")
        sys.exit(1)

    # Load data
    print(f"\nLoading: {input_path}")
    df = load_data(input_path)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # Compute statistics
    print("\nComputing descriptive statistics...")
    stats_df = compute_descriptive_stats(df)
    print(f"  Variables analyzed: {len(stats_df)}")

    # Determine output file
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = DATA_WORK_DIR / 'exploratory'
        ensure_dir(output_dir)
        output_path = output_dir / 'descriptive_stats.csv'

    # Save results
    stats_df.to_csv(output_path, index=False)
    print(f"\nResults saved: {output_path}")

    # Print summary
    if args.verbose:
        print("\n" + "-" * 60)
        print("RESULTS")
        print("-" * 60)
        print(stats_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
