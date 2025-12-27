#!/usr/bin/env python3
"""
Stage 04: Robustness Checks

Purpose: Run robustness specifications and sensitivity analyses.

This stage handles:
- Alternative specifications
- Placebo tests (time and treatment group)
- Sample restriction tests
- Alternative standard error methods

Input Files
-----------
- data_work/panel.parquet

Output Files
------------
- data_work/diagnostics/robustness_results.csv
- data_work/diagnostics/placebo_results.csv

Usage
-----
    python src/pipeline.py estimate_robustness
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
    save_diagnostic,
    ensure_dir,
    add_significance_stars,
)
from stages._qa_utils import generate_qa_report, QAMetrics


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = 'panel.parquet'
OUTCOME_VAR = 'outcome'
TREATMENT_VAR = 'treatment'


# ============================================================
# ROBUSTNESS RESULT CLASSES
# ============================================================

@dataclass
class RobustnessResult:
    """Result from a robustness check."""
    test_name: str
    test_type: str  # 'specification', 'placebo', 'sample', 'se'
    coefficient: float
    std_error: float
    p_value: float
    n_obs: int
    description: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'test_type': self.test_type,
            'coefficient': self.coefficient,
            'std_error': self.std_error,
            'p_value': self.p_value,
            'n_obs': self.n_obs,
            'significant': self.p_value < 0.05,
            'description': self.description
        }


# ============================================================
# ROBUSTNESS TESTS
# ============================================================

def run_simple_ols(df: pd.DataFrame, y_var: str, x_var: str) -> dict:
    """Run simple OLS and return coefficient, SE, p-value."""
    df_clean = df[[y_var, x_var]].dropna()
    n = len(df_clean)

    if n < 10:
        raise ValueError(f"Insufficient observations: {n}")

    y = df_clean[y_var].values
    X = np.column_stack([np.ones(n), df_clean[x_var].values])

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y

    residuals = y - X @ beta
    s2 = np.sum(residuals ** 2) / (n - 2)
    se = np.sqrt(s2 * XtX_inv[1, 1])

    t_stat = beta[1] / se
    # Approximate p-value
    p_value = 2 * (1 - min(0.99999, 0.5 + 0.5 * (1 - (1 + t_stat**2 / (n-2))**(-(n-2)/2))))

    return {
        'coefficient': beta[1],
        'std_error': se,
        'p_value': p_value,
        'n_obs': n
    }


def run_placebo_time(
    df: pd.DataFrame,
    n_placebos: int = 5,
    pre_period_end: int = 11
) -> list[RobustnessResult]:
    """
    Run placebo tests using fake treatment timing.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    n_placebos : int
        Number of placebo tests to run
    pre_period_end : int
        Last pre-treatment period

    Returns
    -------
    list[RobustnessResult]
        Results from placebo tests
    """
    results = []

    # Get pre-treatment periods for placebo
    if 'period' not in df.columns:
        return results

    periods = sorted(df['period'].unique())
    pre_periods = [p for p in periods if p < pre_period_end]

    if len(pre_periods) < n_placebos + 2:
        n_placebos = max(1, len(pre_periods) - 2)

    # Select placebo periods
    placebo_periods = pre_periods[-(n_placebos + 1):-1]

    for i, placebo_period in enumerate(placebo_periods):
        # Create fake treatment based on placebo period
        df_placebo = df[df['period'] <= pre_period_end].copy()

        if 'ever_treated' in df_placebo.columns:
            df_placebo['placebo_treat'] = (
                (df_placebo['ever_treated'] == 1) &
                (df_placebo['period'] >= placebo_period)
            ).astype(int)
        else:
            df_placebo['placebo_treat'] = (
                (df_placebo['period'] >= placebo_period)
            ).astype(int)

        try:
            result = run_simple_ols(df_placebo, OUTCOME_VAR, 'placebo_treat')
            results.append(RobustnessResult(
                test_name=f'placebo_t{placebo_period}',
                test_type='placebo',
                coefficient=result['coefficient'],
                std_error=result['std_error'],
                p_value=result['p_value'],
                n_obs=result['n_obs'],
                description=f'Placebo treatment at period {placebo_period}'
            ))
        except Exception as e:
            print(f"    Placebo period {placebo_period} failed: {e}")

    return results


def run_sample_restrictions(df: pd.DataFrame) -> list[RobustnessResult]:
    """
    Run tests with various sample restrictions.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data

    Returns
    -------
    list[RobustnessResult]
        Results from sample restriction tests
    """
    results = []

    # Test 1: Excluding first and last periods
    if 'period' in df.columns:
        periods = df['period'].unique()
        if len(periods) > 4:
            df_trim = df[
                (df['period'] > periods.min()) &
                (df['period'] < periods.max())
            ]
            try:
                result = run_simple_ols(df_trim, OUTCOME_VAR, TREATMENT_VAR)
                results.append(RobustnessResult(
                    test_name='trim_endpoints',
                    test_type='sample',
                    coefficient=result['coefficient'],
                    std_error=result['std_error'],
                    p_value=result['p_value'],
                    n_obs=result['n_obs'],
                    description='Excluding first and last periods'
                ))
            except Exception:
                pass

    # Test 2: Random subsample (80%)
    np.random.seed(42)
    df_subsample = df.sample(frac=0.8)
    try:
        result = run_simple_ols(df_subsample, OUTCOME_VAR, TREATMENT_VAR)
        results.append(RobustnessResult(
            test_name='subsample_80',
            test_type='sample',
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            p_value=result['p_value'],
            n_obs=result['n_obs'],
            description='Random 80% subsample'
        ))
    except Exception:
        pass

    # Test 3: Exclude extreme outcomes
    q01 = df[OUTCOME_VAR].quantile(0.01)
    q99 = df[OUTCOME_VAR].quantile(0.99)
    df_trim_outcomes = df[
        (df[OUTCOME_VAR] >= q01) &
        (df[OUTCOME_VAR] <= q99)
    ]
    try:
        result = run_simple_ols(df_trim_outcomes, OUTCOME_VAR, TREATMENT_VAR)
        results.append(RobustnessResult(
            test_name='trim_outliers',
            test_type='sample',
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            p_value=result['p_value'],
            n_obs=result['n_obs'],
            description='Excluding 1st and 99th percentile outcomes'
        ))
    except Exception:
        pass

    return results


def run_alternative_specs(df: pd.DataFrame) -> list[RobustnessResult]:
    """
    Run alternative specifications.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data

    Returns
    -------
    list[RobustnessResult]
        Results from alternative specifications
    """
    results = []

    # Main estimate for comparison
    try:
        main = run_simple_ols(df, OUTCOME_VAR, TREATMENT_VAR)
        results.append(RobustnessResult(
            test_name='main',
            test_type='specification',
            coefficient=main['coefficient'],
            std_error=main['std_error'],
            p_value=main['p_value'],
            n_obs=main['n_obs'],
            description='Main specification (baseline)'
        ))
    except Exception:
        pass

    # Add covariates if available
    covariates = [c for c in df.columns if c.startswith('covariate_')]
    if covariates:
        # Run with first covariate as control
        df_cov = df[[OUTCOME_VAR, TREATMENT_VAR, covariates[0]]].dropna()
        n = len(df_cov)
        if n >= 10:
            y = df_cov[OUTCOME_VAR].values
            X = np.column_stack([
                np.ones(n),
                df_cov[TREATMENT_VAR].values,
                df_cov[covariates[0]].values
            ])

            try:
                XtX_inv = np.linalg.inv(X.T @ X)
                beta = XtX_inv @ X.T @ y
                residuals = y - X @ beta
                s2 = np.sum(residuals ** 2) / (n - 3)
                se = np.sqrt(s2 * XtX_inv[1, 1])
                t_stat = beta[1] / se
                p_value = 2 * (1 - min(0.99999, 0.5 + 0.5 * (1 - (1 + t_stat**2 / (n-3))**(-(n-3)/2))))

                results.append(RobustnessResult(
                    test_name='with_control',
                    test_type='specification',
                    coefficient=beta[1],
                    std_error=se,
                    p_value=p_value,
                    n_obs=n,
                    description=f'Including {covariates[0]} as control'
                ))
            except Exception:
                pass

    return results


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(
    run_placebos: bool = True,
    run_samples: bool = True,
    run_specs: bool = True,
    verbose: bool = True
):
    """
    Execute robustness checks pipeline.

    Parameters
    ----------
    run_placebos : bool
        Run placebo timing tests
    run_samples : bool
        Run sample restriction tests
    run_specs : bool
        Run alternative specification tests
    verbose : bool
        Print detailed output
    """
    print("=" * 60)
    print("Stage 04: Robustness Checks")
    print("=" * 60)

    # Setup paths
    work_dir = get_data_dir('work')
    diag_dir = get_data_dir('diagnostics')
    input_path = work_dir / INPUT_FILE

    ensure_dir(diag_dir)

    # Load data
    print(f"\n  Loading: {INPUT_FILE}")
    if not input_path.exists():
        print(f"  ERROR: Input file not found: {input_path}")
        print("  Run 'build_panel' stage first.")
        sys.exit(1)

    df = load_data(input_path)
    print(f"    -> {len(df):,} rows")

    all_results = []

    # Run alternative specifications
    if run_specs:
        print("\n  Running alternative specifications...")
        spec_results = run_alternative_specs(df)
        all_results.extend(spec_results)
        print(f"    Completed {len(spec_results)} tests")

    # Run placebo tests
    if run_placebos:
        print("\n  Running placebo tests...")
        placebo_results = run_placebo_time(df)
        all_results.extend(placebo_results)
        print(f"    Completed {len(placebo_results)} tests")

    # Run sample restrictions
    if run_samples:
        print("\n  Running sample restriction tests...")
        sample_results = run_sample_restrictions(df)
        all_results.extend(sample_results)
        print(f"    Completed {len(sample_results)} tests")

    # Save results
    if all_results:
        results_df = pd.DataFrame([r.to_dict() for r in all_results])
        save_diagnostic(results_df, 'robustness_results')
        print(f"\n  Results saved to: {diag_dir}")

    # Summary
    print("\n" + "-" * 60)
    print("ROBUSTNESS SUMMARY")
    print("-" * 60)
    print(f"  Total tests: {len(all_results)}")

    if all_results and verbose:
        print("\n  Results:")
        print(f"  {'Test':<25} {'Coef':>10} {'SE':>10} {'p-value':>10}")
        print("  " + "-" * 55)
        for r in all_results:
            stars = add_significance_stars(r.p_value)
            print(f"  {r.test_name:<25} {r.coefficient:>10.4f} {r.std_error:>10.4f} {r.p_value:>10.4f}{stars}")

    # Generate QA report
    metrics = QAMetrics()
    metrics.add('n_tests', len(all_results))
    if all_results:
        significant = sum(1 for r in all_results if r.p_value < 0.05)
        metrics.add('n_significant_05', significant)
        metrics.add_pct('significant', (significant / len(all_results)) * 100 if all_results else 0)
    generate_qa_report('s04_robustness', metrics)

    print("\n" + "=" * 60)
    print("Stage 04 complete.")
    print("=" * 60)

    return all_results


if __name__ == '__main__':
    main()
