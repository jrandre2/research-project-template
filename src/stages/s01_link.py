#!/usr/bin/env python3
"""
Stage 01: Record Linkage

Purpose: Link records across multiple data sources using various matching strategies.

This stage handles:
- Exact matching on key columns
- Fuzzy matching using string similarity
- Merge quality diagnostics
- Match rate tracking

Input Files
-----------
- data_work/data_raw.parquet (primary)
- data_work/<additional_sources>.parquet (optional)

Output Files
------------
- data_work/data_linked.parquet
- data_work/diagnostics/linkage_summary.csv

Usage
-----
    python src/pipeline.py link_records
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Literal
from dataclasses import dataclass, field
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
    save_diagnostic,
    ensure_dir,
    calculate_match_rate,
)
from stages._qa_utils import qa_for_stage


# ============================================================
# CONFIGURATION
# ============================================================

# Input/output configuration
INPUT_FILE = 'data_raw.parquet'
OUTPUT_FILE = 'data_linked.parquet'

# Default key columns for matching
DEFAULT_KEY_COLUMNS = ['id']


# ============================================================
# LINKAGE RESULT TRACKING
# ============================================================

@dataclass
class LinkageResult:
    """Track results of a linkage operation."""
    source_name: str
    n_source: int
    n_matched: int
    n_unmatched: int
    match_type: str  # 'exact', 'fuzzy', 'spatial'
    key_columns: list = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        """Calculate match rate."""
        total = self.n_matched + self.n_unmatched
        return self.n_matched / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'source': self.source_name,
            'n_source': self.n_source,
            'n_matched': self.n_matched,
            'n_unmatched': self.n_unmatched,
            'match_rate': self.match_rate,
            'match_type': self.match_type,
            'key_columns': ','.join(self.key_columns)
        }


# ============================================================
# EXACT MATCHING
# ============================================================

def exact_match(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    on: Union[str, list[str]],
    how: Literal['left', 'inner', 'outer'] = 'left',
    suffixes: tuple = ('', '_right'),
    validate_merge: bool = True
) -> tuple[pd.DataFrame, LinkageResult]:
    """
    Perform exact matching on key columns.

    Parameters
    ----------
    df_left : pd.DataFrame
        Primary DataFrame
    df_right : pd.DataFrame
        Secondary DataFrame to match
    on : str or list
        Column(s) to match on
    how : str
        Type of merge
    suffixes : tuple
        Suffixes for duplicate columns
    validate_merge : bool
        Validate merge relationship

    Returns
    -------
    tuple[pd.DataFrame, LinkageResult]
        Merged DataFrame and linkage result
    """
    if isinstance(on, str):
        on = [on]

    # Perform merge
    df_merged = df_left.merge(
        df_right,
        on=on,
        how=how,
        suffixes=suffixes,
        indicator=True if how == 'left' else False
    )

    # Track match results
    if '_merge' in df_merged.columns:
        n_matched = (df_merged['_merge'] == 'both').sum()
        n_unmatched = (df_merged['_merge'] == 'left_only').sum()
        df_merged = df_merged.drop('_merge', axis=1)
    else:
        # For inner/outer, count matched differently
        n_matched = len(df_merged)
        n_unmatched = len(df_left) - n_matched

    result = LinkageResult(
        source_name=getattr(df_right, 'name', 'secondary'),
        n_source=len(df_right),
        n_matched=int(n_matched),
        n_unmatched=int(n_unmatched),
        match_type='exact',
        key_columns=on
    )

    return df_merged, result


# ============================================================
# FUZZY MATCHING
# ============================================================

def fuzzy_match(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_on: str,
    right_on: str,
    threshold: float = 0.8,
    method: Literal['levenshtein', 'jaro_winkler', 'contains'] = 'levenshtein'
) -> tuple[pd.DataFrame, LinkageResult]:
    """
    Perform fuzzy string matching.

    Parameters
    ----------
    df_left : pd.DataFrame
        Primary DataFrame
    df_right : pd.DataFrame
        Secondary DataFrame
    left_on : str
        Column in left DataFrame
    right_on : str
        Column in right DataFrame
    threshold : float
        Minimum similarity score (0-1)
    method : str
        Matching method

    Returns
    -------
    tuple[pd.DataFrame, LinkageResult]
        Merged DataFrame and linkage result
    """
    matches = []

    # Get unique values from right side for matching
    right_values = df_right[right_on].unique()

    for idx, row in df_left.iterrows():
        left_val = str(row[left_on])
        best_match = None
        best_score = 0

        for right_val in right_values:
            right_val_str = str(right_val)

            if method == 'levenshtein':
                score = _levenshtein_similarity(left_val, right_val_str)
            elif method == 'jaro_winkler':
                score = _jaro_winkler_similarity(left_val, right_val_str)
            elif method == 'contains':
                score = 1.0 if left_val.lower() in right_val_str.lower() else 0.0
            else:
                score = 1.0 if left_val == right_val_str else 0.0

            if score > best_score and score >= threshold:
                best_score = score
                best_match = right_val

        matches.append({
            'left_idx': idx,
            'match_value': best_match,
            'match_score': best_score
        })

    # Create match DataFrame
    match_df = pd.DataFrame(matches)

    # Merge matches back
    df_left = df_left.copy()
    df_left['_fuzzy_match'] = match_df['match_value'].values
    df_left['_fuzzy_score'] = match_df['match_score'].values

    # Join with right DataFrame
    df_merged = df_left.merge(
        df_right,
        left_on='_fuzzy_match',
        right_on=right_on,
        how='left'
    )

    n_matched = df_merged['_fuzzy_match'].notna().sum()

    result = LinkageResult(
        source_name=getattr(df_right, 'name', 'secondary'),
        n_source=len(df_right),
        n_matched=int(n_matched),
        n_unmatched=len(df_left) - int(n_matched),
        match_type=f'fuzzy_{method}',
        key_columns=[left_on, right_on]
    )

    # Clean up temp columns
    df_merged = df_merged.drop(['_fuzzy_match', '_fuzzy_score'], axis=1, errors='ignore')

    return df_merged, result


def _levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculate Levenshtein similarity (0-1)."""
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    if len(s1) == 0 or len(s2) == 0:
        return 0.0

    # Simple edit distance calculation
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1].lower() == s2[j-1].lower() else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )

    distance = dp[m][n]
    max_len = max(m, n)
    return 1.0 - (distance / max_len)


def _jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Calculate Jaro-Winkler similarity (0-1)."""
    # Simplified implementation
    s1, s2 = s1.lower(), s2.lower()

    if s1 == s2:
        return 1.0
    if len(s1) == 0 or len(s2) == 0:
        return 0.0

    # Matching window
    match_distance = max(len(s1), len(s2)) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    matches = 0
    transpositions = 0

    for i in range(len(s1)):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(s2))

        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len(s1)):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len(s1) + matches / len(s2) +
            (matches - transpositions / 2) / matches) / 3

    # Winkler modification (common prefix bonus)
    prefix = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1 - jaro)


# ============================================================
# DIAGNOSTICS
# ============================================================

def generate_linkage_diagnostics(
    results: list[LinkageResult],
    output_dir: Path
) -> pd.DataFrame:
    """
    Generate linkage diagnostics report.

    Parameters
    ----------
    results : list[LinkageResult]
        List of linkage results
    output_dir : Path
        Directory for output

    Returns
    -------
    pd.DataFrame
        Summary DataFrame
    """
    summary_data = [r.to_dict() for r in results]
    summary_df = pd.DataFrame(summary_data)

    # Save to diagnostics
    ensure_dir(output_dir)
    save_diagnostic(summary_df, 'linkage_summary')

    return summary_df


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(
    additional_sources: Optional[list[str]] = None,
    key_columns: Optional[list[str]] = None,
    verbose: bool = True
):
    """
    Execute record linkage pipeline.

    Parameters
    ----------
    additional_sources : list, optional
        Additional parquet files to link
    key_columns : list, optional
        Columns to use for matching
    verbose : bool
        Print detailed output
    """
    print("=" * 60)
    print("Stage 01: Record Linkage")
    print("=" * 60)

    # Setup paths
    work_dir = get_data_dir('work')
    diag_dir = get_data_dir('diagnostics')
    input_path = work_dir / INPUT_FILE
    output_path = work_dir / OUTPUT_FILE

    key_columns = key_columns or DEFAULT_KEY_COLUMNS
    linkage_results = []

    # Load primary data
    print(f"\n  Loading primary data: {INPUT_FILE}")
    if not input_path.exists():
        print(f"  ERROR: Input file not found: {input_path}")
        print("  Run 'ingest_data' stage first.")
        sys.exit(1)

    df = load_data(input_path)
    print(f"    -> {len(df):,} rows, {len(df.columns)} columns")

    # Check for additional sources to link
    if additional_sources:
        for source in additional_sources:
            source_path = work_dir / source
            if source_path.exists():
                print(f"\n  Linking: {source}")
                df_source = load_data(source_path)
                df_source.name = source

                # Determine common columns for matching
                common_cols = [c for c in key_columns if c in df_source.columns]

                if common_cols:
                    df, result = exact_match(df, df_source, on=common_cols)
                    linkage_results.append(result)
                    print(f"    Match rate: {result.match_rate:.1%}")
                else:
                    print(f"    Warning: No common key columns found")
            else:
                print(f"  Warning: Source not found: {source_path}")

    # If no additional sources, demonstrate self-linkage or pass-through
    if not linkage_results:
        print("\n  No additional sources specified.")
        print("  Passing data through with synthetic linkage columns...")

        # Add placeholder linkage columns for demonstration
        df['source_file'] = 'primary'
        df['link_status'] = 'direct'

        # Create a synthetic linkage result
        result = LinkageResult(
            source_name='passthrough',
            n_source=len(df),
            n_matched=len(df),
            n_unmatched=0,
            match_type='passthrough',
            key_columns=key_columns
        )
        linkage_results.append(result)

    # Generate diagnostics
    print("\n  Generating linkage diagnostics...")
    summary_df = generate_linkage_diagnostics(linkage_results, diag_dir)

    # Save output
    print(f"\n  Saving to: {output_path}")
    save_data(df, output_path)

    # Summary
    print("\n" + "-" * 60)
    print("LINKAGE SUMMARY")
    print("-" * 60)

    for result in linkage_results:
        print(f"\n  {result.source_name}:")
        print(f"    Match type: {result.match_type}")
        print(f"    Matched: {result.n_matched:,}")
        print(f"    Unmatched: {result.n_unmatched:,}")
        print(f"    Match rate: {result.match_rate:.1%}")

    print(f"\n  Final dataset: {len(df):,} rows, {len(df.columns)} columns")

    if verbose:
        print("\n  Columns:")
        for col in df.columns:
            print(f"    - {col}")

    # Generate QA report
    qa_for_stage('s01_link', df, output_file=str(output_path))

    print("\n" + "=" * 60)
    print("Stage 01 complete.")
    print("=" * 60)

    return df


if __name__ == '__main__':
    main()
