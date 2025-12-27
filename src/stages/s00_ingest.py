#!/usr/bin/env python3
"""
Stage 00: Data Ingestion

Purpose: Load and preprocess raw data files into a standardized format.

This stage handles:
- Loading data from various formats (CSV, parquet, Excel, etc.)
- Initial data cleaning and type conversion
- Basic validation checks
- Output to standardized parquet format

Input Files
-----------
- data_raw/*.csv, *.parquet, *.xlsx (configurable)
- OR synthetic data if no raw data exists

Output Files
------------
- data_work/data_raw.parquet

Usage
-----
    python src/pipeline.py ingest_data
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from utils.helpers import (
    get_project_root,
    get_data_dir,
    load_data,
    save_data,
    ensure_dir,
)
from utils.validation import (
    DataValidator,
    no_missing_values,
    unique_values,
    row_count,
)
from stages._qa_utils import qa_for_stage


# ============================================================
# CONFIGURATION
# ============================================================

# Default input patterns
INPUT_PATTERNS = ['*.csv', '*.parquet', '*.xlsx']

# Output configuration
OUTPUT_FILE = 'data_raw.parquet'

# Required columns (customize for your project)
REQUIRED_COLUMNS = ['id']

# Optional: Column type mapping
COLUMN_TYPES = {
    'id': 'int64',
    # Add your column types here
}


# ============================================================
# DATA LOADING
# ============================================================

def find_input_files(
    raw_dir: Path,
    patterns: list[str] = None
) -> list[Path]:
    """
    Find all input files matching specified patterns.

    Parameters
    ----------
    raw_dir : Path
        Directory to search
    patterns : list, optional
        Glob patterns to match

    Returns
    -------
    list[Path]
        List of matching files
    """
    patterns = patterns or INPUT_PATTERNS
    files = []
    for pattern in patterns:
        files.extend(raw_dir.glob(pattern))
    return sorted(files)


def load_all_sources(
    files: list[Path],
    concat: bool = True
) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Load data from multiple source files.

    Parameters
    ----------
    files : list[Path]
        Files to load
    concat : bool
        If True, concatenate all files into single DataFrame

    Returns
    -------
    DataFrame or dict
        Loaded data
    """
    dataframes = {}

    for path in files:
        print(f"  Loading: {path.name}")
        try:
            df = load_data(path)
            dataframes[path.stem] = df
            print(f"    -> {len(df):,} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"    ERROR: {e}")

    if concat and dataframes:
        return pd.concat(dataframes.values(), ignore_index=True)
    return dataframes


def generate_demo_data() -> pd.DataFrame:
    """
    Generate synthetic demo data when no raw data exists.

    Returns
    -------
    pd.DataFrame
        Synthetic demonstration dataset
    """
    print("  Generating synthetic demo data...")

    from utils.synthetic_data import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=42)
    df = gen.generate_panel(
        n_units=500,
        n_periods=24,
        treatment_period=12,
        treatment_share=0.5,
        treatment_effect=0.15,
        n_covariates=3
    )

    # Rename to standard columns
    df = df.rename(columns={'unit_id': 'id'})

    print(f"    -> Generated {len(df):,} rows")
    return df


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    reset_index: bool = True
) -> pd.DataFrame:
    """
    Apply basic data cleaning operations.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    drop_duplicates : bool
        Whether to drop duplicate rows
    reset_index : bool
        Whether to reset the index

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    """
    print("  Cleaning data...")
    n_original = len(df)

    # Drop completely empty rows
    df = df.dropna(how='all')
    n_after_empty = len(df)
    if n_after_empty < n_original:
        print(f"    Dropped {n_original - n_after_empty} empty rows")

    # Drop duplicates
    if drop_duplicates:
        df = df.drop_duplicates()
        n_after_dups = len(df)
        if n_after_dups < n_after_empty:
            print(f"    Dropped {n_after_empty - n_after_dups} duplicate rows")

    # Reset index
    if reset_index:
        df = df.reset_index(drop=True)

    return df


def convert_types(
    df: pd.DataFrame,
    type_map: Optional[dict] = None
) -> pd.DataFrame:
    """
    Convert column types according to specification.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    type_map : dict, optional
        Mapping of column names to types

    Returns
    -------
    pd.DataFrame
        DataFrame with converted types
    """
    type_map = type_map or COLUMN_TYPES

    for col, dtype in type_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError) as e:
                print(f"  Warning: Could not convert {col} to {dtype}: {e}")

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_input(df: pd.DataFrame) -> bool:
    """
    Validate input data meets requirements.

    Parameters
    ----------
    df : pd.DataFrame
        Data to validate

    Returns
    -------
    bool
        True if validation passes
    """
    print("  Validating data...")

    validator = DataValidator()

    # Add validation rules
    validator.add_rule(row_count(min_rows=1))

    if REQUIRED_COLUMNS:
        validator.add_rule(no_missing_values(REQUIRED_COLUMNS))

    # Run validation
    report = validator.validate(df)

    if report.has_errors:
        print(report.format())
        return False

    passed = sum(1 for r in report.results if r.passed)
    total = len(report.results)
    print(f"    All validations passed ({passed}/{total})")
    return True


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(
    use_demo: bool = False,
    validate: bool = True,
    verbose: bool = True
):
    """
    Execute data ingestion pipeline.

    Parameters
    ----------
    use_demo : bool
        Force use of synthetic demo data
    validate : bool
        Run validation checks
    verbose : bool
        Print detailed output
    """
    print("=" * 60)
    print("Stage 00: Data Ingestion")
    print("=" * 60)

    # Setup paths
    project_root = get_project_root()
    raw_dir = get_data_dir('raw')
    work_dir = get_data_dir('work')
    output_path = work_dir / OUTPUT_FILE

    ensure_dir(work_dir)

    # Find input files
    input_files = find_input_files(raw_dir)

    if use_demo or not input_files:
        if not input_files:
            print("\n  No raw data files found in data_raw/")
        df = generate_demo_data()
    else:
        print(f"\n  Found {len(input_files)} input file(s)")
        df = load_all_sources(input_files)

    # Clean data
    df = clean_data(df)
    df = convert_types(df)

    # Validate
    if validate:
        if not validate_input(df):
            print("\nERROR: Validation failed. Aborting.")
            sys.exit(1)

    # Save output
    print(f"\n  Saving to: {output_path}")
    save_data(df, output_path)

    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Output: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    if verbose:
        print("\n  Columns:")
        for col in df.columns:
            print(f"    - {col}: {df[col].dtype}")

    # Generate QA report
    qa_for_stage('s00_ingest', df, output_file=str(output_path))

    print("\n" + "=" * 60)
    print("Stage 00 complete.")
    print("=" * 60)

    return df


if __name__ == '__main__':
    main()
