#!/usr/bin/env python3
"""
Stage 02: Panel Construction

Purpose: Construct analysis panel from linked data.

This stage handles:
- Panel structure creation (balanced/unbalanced)
- Time period generation
- Treatment variable construction
- Fixed effects preparation
- Panel diagnostics

Input Files
-----------
- data_work/data_linked.parquet

Output Files
------------
- data_work/panel.parquet
- data_work/diagnostics/panel_summary.csv

Usage
-----
    python src/pipeline.py build_panel
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from utils.helpers import (
    get_data_dir,
    load_data,
    save_data,
    save_diagnostic,
    ensure_dir,
)
from utils.validation import (
    DataValidator,
    no_missing_values,
    unique_values,
    row_count,
    no_duplicate_rows,
)
from stages._qa_utils import qa_for_stage


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = 'data_linked.parquet'
OUTPUT_FILE = 'panel.parquet'

# Panel structure configuration
UNIT_ID_COL = 'id'
TIME_COL = 'period'
TREATMENT_COL = 'treatment'


# ============================================================
# PANEL DIAGNOSTICS
# ============================================================

@dataclass
class PanelDiagnostics:
    """Diagnostics for panel structure."""
    n_units: int
    n_periods: int
    n_observations: int
    is_balanced: bool
    balance_rate: float
    treatment_share: float
    n_treated_units: int
    n_control_units: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'n_units': self.n_units,
            'n_periods': self.n_periods,
            'n_observations': self.n_observations,
            'expected_if_balanced': self.n_units * self.n_periods,
            'is_balanced': self.is_balanced,
            'balance_rate': self.balance_rate,
            'treatment_share': self.treatment_share,
            'n_treated_units': self.n_treated_units,
            'n_control_units': self.n_control_units
        }

    def format(self) -> str:
        """Format as string."""
        lines = [
            "PANEL DIAGNOSTICS",
            "-" * 40,
            f"Units: {self.n_units:,}",
            f"Periods: {self.n_periods}",
            f"Observations: {self.n_observations:,}",
            f"Balanced: {'Yes' if self.is_balanced else 'No'} ({self.balance_rate:.1%})",
            f"Treatment share: {self.treatment_share:.1%}",
            f"Treated units: {self.n_treated_units:,}",
            f"Control units: {self.n_control_units:,}",
        ]
        return "\n".join(lines)


def calculate_panel_diagnostics(
    df: pd.DataFrame,
    unit_col: str = UNIT_ID_COL,
    time_col: str = TIME_COL,
    treatment_col: str = TREATMENT_COL
) -> PanelDiagnostics:
    """
    Calculate panel structure diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame
    unit_col : str
        Unit identifier column
    time_col : str
        Time period column
    treatment_col : str
        Treatment indicator column

    Returns
    -------
    PanelDiagnostics
        Diagnostic results
    """
    n_units = df[unit_col].nunique()
    n_periods = df[time_col].nunique()
    n_observations = len(df)
    expected_balanced = n_units * n_periods

    is_balanced = n_observations == expected_balanced
    balance_rate = n_observations / expected_balanced if expected_balanced > 0 else 0

    # Treatment statistics
    if treatment_col in df.columns:
        treatment_share = df[treatment_col].mean()
        treated_units = df.groupby(unit_col)[treatment_col].max()
        n_treated = (treated_units > 0).sum()
        n_control = n_units - n_treated
    else:
        treatment_share = 0.0
        n_treated = 0
        n_control = n_units

    return PanelDiagnostics(
        n_units=n_units,
        n_periods=n_periods,
        n_observations=n_observations,
        is_balanced=is_balanced,
        balance_rate=balance_rate,
        treatment_share=treatment_share,
        n_treated_units=n_treated,
        n_control_units=n_control
    )


# ============================================================
# PANEL CONSTRUCTION
# ============================================================

def balance_panel(
    df: pd.DataFrame,
    unit_col: str = UNIT_ID_COL,
    time_col: str = TIME_COL,
    fill_method: Literal['drop', 'ffill', 'zero'] = 'drop'
) -> pd.DataFrame:
    """
    Balance panel by handling missing unit-period combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    unit_col : str
        Unit identifier column
    time_col : str
        Time period column
    fill_method : str
        How to handle missing combinations:
        - 'drop': Keep only complete units
        - 'ffill': Forward fill missing values
        - 'zero': Fill with zeros

    Returns
    -------
    pd.DataFrame
        Balanced panel
    """
    # Get all unique units and periods
    all_units = df[unit_col].unique()
    all_periods = df[time_col].unique()

    # Create full index
    full_index = pd.MultiIndex.from_product(
        [all_units, all_periods],
        names=[unit_col, time_col]
    )

    # Set index and reindex
    df_indexed = df.set_index([unit_col, time_col])
    df_balanced = df_indexed.reindex(full_index)

    if fill_method == 'drop':
        # Keep only units present in all periods
        obs_per_unit = df.groupby(unit_col).size()
        complete_units = obs_per_unit[obs_per_unit == len(all_periods)].index
        df_balanced = df_balanced.loc[complete_units]
    elif fill_method == 'ffill':
        df_balanced = df_balanced.groupby(level=0).ffill()
    elif fill_method == 'zero':
        numeric_cols = df_balanced.select_dtypes(include=[np.number]).columns
        df_balanced[numeric_cols] = df_balanced[numeric_cols].fillna(0)

    return df_balanced.reset_index()


def create_fixed_effects(
    df: pd.DataFrame,
    unit_col: str = UNIT_ID_COL,
    time_col: str = TIME_COL
) -> pd.DataFrame:
    """
    Create fixed effect indicator columns.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame
    unit_col : str
        Unit identifier column
    time_col : str
        Time period column

    Returns
    -------
    pd.DataFrame
        DataFrame with FE columns added
    """
    df = df.copy()

    # Create unit FE code
    df['unit_fe'] = pd.Categorical(df[unit_col]).codes

    # Create time FE code
    df['time_fe'] = pd.Categorical(df[time_col]).codes

    return df


def create_event_time(
    df: pd.DataFrame,
    unit_col: str = UNIT_ID_COL,
    time_col: str = TIME_COL,
    treatment_col: str = TREATMENT_COL,
    treatment_period_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Create event time variable for event study analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame
    unit_col : str
        Unit identifier column
    time_col : str
        Time period column
    treatment_col : str
        Treatment indicator column
    treatment_period_col : str, optional
        Column with treatment timing (if pre-existing)

    Returns
    -------
    pd.DataFrame
        DataFrame with event_time column added
    """
    df = df.copy()

    if treatment_period_col and treatment_period_col in df.columns:
        # Use existing treatment period column
        df['event_time'] = df[time_col] - df[treatment_period_col]
    else:
        # Infer treatment period from treatment indicator
        # Find first period where treatment = 1 for each unit
        treatment_start = df[df[treatment_col] == 1].groupby(unit_col)[time_col].min()
        df['treatment_period'] = df[unit_col].map(treatment_start)
        df['event_time'] = df[time_col] - df['treatment_period']

        # For never-treated units, set event_time to NaN
        df.loc[df['treatment_period'].isna(), 'event_time'] = np.nan

    return df


def create_treatment_indicators(
    df: pd.DataFrame,
    unit_col: str = UNIT_ID_COL,
    time_col: str = TIME_COL,
    treatment_col: str = TREATMENT_COL
) -> pd.DataFrame:
    """
    Create additional treatment indicator variables.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with additional treatment indicators
    """
    df = df.copy()

    # Ever treated indicator (unit level)
    ever_treated = df.groupby(unit_col)[treatment_col].max()
    df['ever_treated'] = df[unit_col].map(ever_treated)

    # Post-treatment indicator (based on first treatment)
    if 'treatment_period' in df.columns:
        df['post_treatment'] = (df[time_col] >= df['treatment_period']).astype(int)
        df.loc[df['treatment_period'].isna(), 'post_treatment'] = 0

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_panel(
    df: pd.DataFrame,
    unit_col: str = UNIT_ID_COL,
    time_col: str = TIME_COL
) -> bool:
    """
    Validate panel structure.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame

    Returns
    -------
    bool
        True if validation passes
    """
    print("  Validating panel structure...")

    validator = DataValidator()

    # Required columns exist
    validator.add_rule(no_missing_values([unit_col, time_col]))

    # No duplicate unit-period combinations
    validator.add_rule(no_duplicate_rows([unit_col, time_col]))

    # Minimum observations
    validator.add_rule(row_count(min_rows=10))

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
    balance: bool = False,
    fill_method: str = 'drop',
    create_event_study: bool = True,
    verbose: bool = True
):
    """
    Execute panel construction pipeline.

    Parameters
    ----------
    balance : bool
        Whether to balance the panel
    fill_method : str
        Method for balancing ('drop', 'ffill', 'zero')
    create_event_study : bool
        Whether to create event study variables
    verbose : bool
        Print detailed output
    """
    print("=" * 60)
    print("Stage 02: Panel Construction")
    print("=" * 60)

    # Setup paths
    work_dir = get_data_dir('work')
    diag_dir = get_data_dir('diagnostics')
    input_path = work_dir / INPUT_FILE
    output_path = work_dir / OUTPUT_FILE

    # Load linked data
    print(f"\n  Loading: {INPUT_FILE}")
    if not input_path.exists():
        print(f"  ERROR: Input file not found: {input_path}")
        print("  Run 'link_records' stage first.")
        sys.exit(1)

    df = load_data(input_path)
    print(f"    -> {len(df):,} rows, {len(df.columns)} columns")

    # Check for required columns
    if TIME_COL not in df.columns:
        print(f"\n  Note: No '{TIME_COL}' column found.")
        print("  Panel structure already exists from synthetic data.")

    # Initial diagnostics
    print("\n  Initial panel structure:")
    initial_diag = calculate_panel_diagnostics(df)
    print(f"    Units: {initial_diag.n_units:,}")
    print(f"    Periods: {initial_diag.n_periods}")
    print(f"    Balance rate: {initial_diag.balance_rate:.1%}")

    # Balance panel if requested
    if balance and not initial_diag.is_balanced:
        print(f"\n  Balancing panel (method: {fill_method})...")
        n_before = len(df)
        df = balance_panel(df, fill_method=fill_method)
        n_after = len(df)
        print(f"    Rows: {n_before:,} -> {n_after:,}")

    # Create fixed effects
    print("\n  Creating fixed effects...")
    df = create_fixed_effects(df)

    # Create event study variables
    if create_event_study and TREATMENT_COL in df.columns:
        print("  Creating event study variables...")
        df = create_event_time(df)
        df = create_treatment_indicators(df)

    # Validate
    if not validate_panel(df):
        print("\nERROR: Panel validation failed.")
        sys.exit(1)

    # Final diagnostics
    final_diag = calculate_panel_diagnostics(df)

    # Save diagnostics
    ensure_dir(diag_dir)
    diag_df = pd.DataFrame([final_diag.to_dict()])
    save_diagnostic(diag_df, 'panel_summary')

    # Save output
    print(f"\n  Saving to: {output_path}")
    save_data(df, output_path)

    # Summary
    print("\n" + "-" * 60)
    print("PANEL SUMMARY")
    print("-" * 60)
    print(final_diag.format())

    if verbose:
        print(f"\n  Columns ({len(df.columns)}):")
        for col in df.columns:
            print(f"    - {col}: {df[col].dtype}")

    # Generate QA report
    qa_for_stage('s02_panel', df, output_file=str(output_path))

    print("\n" + "=" * 60)
    print("Stage 02 complete.")
    print("=" * 60)

    return df


if __name__ == '__main__':
    main()
