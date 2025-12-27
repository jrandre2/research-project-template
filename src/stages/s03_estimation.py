#!/usr/bin/env python3
"""
Stage 03: Estimation

Purpose: Run primary estimation specifications.

This stage handles:
- Specification registry management
- Fixed effects estimation
- Standard error clustering
- Results formatting and export

Input Files
-----------
- data_work/panel.parquet

Output Files
------------
- data_work/diagnostics/estimation_results.csv
- data_work/diagnostics/coefficients.csv

Usage
-----
    python src/pipeline.py run_estimation --specification baseline
    python src/pipeline.py run_estimation -s robust --sample subset
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
    get_data_dir,
    load_data,
    save_data,
    save_diagnostic,
    ensure_dir,
    format_coefficient,
    format_pvalue,
    add_significance_stars,
)
from stages._qa_utils import generate_qa_report, QAMetrics


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = 'panel.parquet'

# Outcome and treatment variables
OUTCOME_VAR = 'outcome'
TREATMENT_VAR = 'treatment'

# Fixed effects columns
UNIT_FE = 'unit_fe'
TIME_FE = 'time_fe'


# ============================================================
# SPECIFICATION REGISTRY
# ============================================================

SPECIFICATIONS = {
    'baseline': {
        'name': 'Baseline',
        'formula': f'{OUTCOME_VAR} ~ {TREATMENT_VAR}',
        'controls': [],
        'fe': [UNIT_FE, TIME_FE],
        'cluster': 'id',
        'description': 'Baseline specification with unit and time FE'
    },
    'no_fe': {
        'name': 'No Fixed Effects',
        'formula': f'{OUTCOME_VAR} ~ {TREATMENT_VAR}',
        'controls': [],
        'fe': [],
        'cluster': None,
        'description': 'Simple OLS without fixed effects'
    },
    'with_controls': {
        'name': 'With Controls',
        'formula': f'{OUTCOME_VAR} ~ {TREATMENT_VAR}',
        'controls': ['covariate_1', 'covariate_2', 'covariate_3'],
        'fe': [UNIT_FE, TIME_FE],
        'cluster': 'id',
        'description': 'Baseline with additional control variables'
    },
    'unit_fe_only': {
        'name': 'Unit FE Only',
        'formula': f'{OUTCOME_VAR} ~ {TREATMENT_VAR}',
        'controls': [],
        'fe': [UNIT_FE],
        'cluster': 'id',
        'description': 'Unit fixed effects only'
    },
}


# ============================================================
# ESTIMATION RESULTS
# ============================================================

@dataclass
class EstimationResult:
    """Results from a single estimation."""
    specification: str
    coefficient: float
    std_error: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_obs: int
    n_units: int
    n_periods: int
    r_squared: float
    controls: list = field(default_factory=list)
    fe: list = field(default_factory=list)
    cluster: Optional[str] = None

    @property
    def significant_05(self) -> bool:
        """Is coefficient significant at 5% level."""
        return self.p_value < 0.05

    @property
    def significant_01(self) -> bool:
        """Is coefficient significant at 1% level."""
        return self.p_value < 0.01

    def format_coefficient(self, decimals: int = 3) -> str:
        """Format coefficient with stars and SE."""
        stars = add_significance_stars(self.p_value)
        return f"{self.coefficient:.{decimals}f}{stars} ({self.std_error:.{decimals}f})"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'specification': self.specification,
            'coefficient': self.coefficient,
            'std_error': self.std_error,
            't_stat': self.t_stat,
            'p_value': self.p_value,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'n_obs': self.n_obs,
            'n_units': self.n_units,
            'n_periods': self.n_periods,
            'r_squared': self.r_squared,
            'controls': ','.join(self.controls),
            'fe': ','.join(self.fe),
            'cluster': self.cluster or 'none',
            'significant_05': self.significant_05,
            'significant_01': self.significant_01
        }


# ============================================================
# ESTIMATION FUNCTIONS
# ============================================================

def run_ols(
    df: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    cluster_var: Optional[str] = None
) -> dict:
    """
    Run OLS regression.

    Parameters
    ----------
    df : pd.DataFrame
        Data
    y_var : str
        Outcome variable
    x_vars : list
        Regressor variables
    cluster_var : str, optional
        Variable for clustered standard errors

    Returns
    -------
    dict
        Regression results
    """
    # Drop missing values
    all_vars = [y_var] + x_vars
    if cluster_var:
        all_vars.append(cluster_var)
    df_clean = df[all_vars].dropna()

    n = len(df_clean)
    if n < len(x_vars) + 1:
        raise ValueError(f"Insufficient observations: {n}")

    # Create design matrix
    y = df_clean[y_var].values
    X = np.column_stack([np.ones(n)] + [df_clean[x].values for x in x_vars])

    # OLS estimation
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ y

    # Residuals and R-squared
    y_hat = X @ beta
    residuals = y - y_hat
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Standard errors
    if cluster_var:
        # Clustered standard errors
        clusters = df_clean[cluster_var].values
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters)

        # Cluster-robust variance
        meat = np.zeros((X.shape[1], X.shape[1]))
        for c in unique_clusters:
            mask = clusters == c
            u_c = residuals[mask]
            X_c = X[mask]
            meat += (X_c.T * u_c) @ (X_c.T * u_c).T

        # Small sample correction
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - X.shape[1]))
        V = correction * XtX_inv @ meat @ XtX_inv
        se = np.sqrt(np.diag(V))
    else:
        # Homoskedastic standard errors
        s2 = ss_res / (n - X.shape[1])
        V = s2 * XtX_inv
        se = np.sqrt(np.diag(V))

    # Treatment coefficient (first regressor after constant)
    coef = beta[1]
    se_coef = se[1]
    t_stat = coef / se_coef
    p_value = 2 * (1 - _t_cdf(abs(t_stat), n - X.shape[1]))

    # Confidence interval
    t_crit = _t_ppf(0.975, n - X.shape[1])
    ci_lower = coef - t_crit * se_coef
    ci_upper = coef + t_crit * se_coef

    return {
        'coefficient': coef,
        'std_error': se_coef,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n,
        'r_squared': r_squared
    }


def _t_cdf(x: float, df: int) -> float:
    """Approximate t-distribution CDF."""
    # Use normal approximation for large df
    if df > 100:
        from math import erf, sqrt
        return 0.5 * (1 + erf(x / sqrt(2)))

    # Simple approximation
    t = x / np.sqrt(df)
    return 0.5 + 0.5 * np.sign(t) * (1 - (1 + t**2/df)**(-df/2))


def _t_ppf(p: float, df: int) -> float:
    """Approximate t-distribution PPF (inverse CDF)."""
    if df > 100:
        # Normal approximation
        from math import sqrt
        # Approximate normal PPF
        a = 8 * (np.pi - 3) / (3 * np.pi * (4 - np.pi))
        x = 2 * p - 1
        inner = (2 / (np.pi * a) + np.log(1 - x**2) / 2)
        return np.sign(x) * sqrt(sqrt(inner**2 - np.log(1 - x**2) / a) - inner)

    # For small df, use approximation
    return 1.96 * (1 + 0.5 / df)


def demean_by_fe(
    df: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    fe_vars: list[str]
) -> pd.DataFrame:
    """
    Demean variables by fixed effects (within transformation).

    Parameters
    ----------
    df : pd.DataFrame
        Data
    y_var : str
        Outcome variable
    x_vars : list
        Regressor variables
    fe_vars : list
        Fixed effect variables

    Returns
    -------
    pd.DataFrame
        Demeaned data
    """
    df_demeaned = df.copy()
    vars_to_demean = [y_var] + x_vars

    for fe in fe_vars:
        for var in vars_to_demean:
            if var in df_demeaned.columns:
                group_mean = df_demeaned.groupby(fe)[var].transform('mean')
                df_demeaned[var] = df_demeaned[var] - group_mean

    return df_demeaned


def run_fe_estimation(
    df: pd.DataFrame,
    spec_name: str
) -> EstimationResult:
    """
    Run fixed effects estimation for a specification.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    spec_name : str
        Specification name from SPECIFICATIONS

    Returns
    -------
    EstimationResult
        Estimation results
    """
    if spec_name not in SPECIFICATIONS:
        raise ValueError(f"Unknown specification: {spec_name}")

    spec = SPECIFICATIONS[spec_name]
    x_vars = [TREATMENT_VAR] + spec['controls']

    # Filter to valid observations
    all_vars = [OUTCOME_VAR] + x_vars + spec['fe']
    if spec['cluster']:
        all_vars.append(spec['cluster'])
    df_valid = df[df[all_vars].notna().all(axis=1)].copy()

    # Demean by fixed effects
    if spec['fe']:
        df_est = demean_by_fe(df_valid, OUTCOME_VAR, x_vars, spec['fe'])
    else:
        df_est = df_valid

    # Run OLS on demeaned data
    results = run_ols(
        df_est,
        OUTCOME_VAR,
        x_vars,
        cluster_var=spec['cluster']
    )

    # Count units and periods
    n_units = df_valid['id'].nunique() if 'id' in df_valid.columns else 0
    n_periods = df_valid['period'].nunique() if 'period' in df_valid.columns else 0

    return EstimationResult(
        specification=spec_name,
        coefficient=results['coefficient'],
        std_error=results['std_error'],
        t_stat=results['t_stat'],
        p_value=results['p_value'],
        ci_lower=results['ci_lower'],
        ci_upper=results['ci_upper'],
        n_obs=results['n_obs'],
        n_units=n_units,
        n_periods=n_periods,
        r_squared=results['r_squared'],
        controls=spec['controls'],
        fe=spec['fe'],
        cluster=spec['cluster']
    )


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(
    specification: str = 'baseline',
    sample: str = 'full',
    run_all: bool = False,
    verbose: bool = True
):
    """
    Execute estimation pipeline.

    Parameters
    ----------
    specification : str
        Specification name to run
    sample : str
        Sample restriction ('full', 'subset', etc.)
    run_all : bool
        Run all specifications
    verbose : bool
        Print detailed output
    """
    print("=" * 60)
    print("Stage 03: Estimation")
    print("=" * 60)

    # Setup paths
    work_dir = get_data_dir('work')
    diag_dir = get_data_dir('diagnostics')
    input_path = work_dir / INPUT_FILE

    # Load panel data
    print(f"\n  Loading: {INPUT_FILE}")
    if not input_path.exists():
        print(f"  ERROR: Input file not found: {input_path}")
        print("  Run 'build_panel' stage first.")
        sys.exit(1)

    df = load_data(input_path)
    print(f"    -> {len(df):,} rows, {len(df.columns)} columns")

    # Check required columns
    required = [OUTCOME_VAR, TREATMENT_VAR]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n  ERROR: Missing required columns: {missing}")
        sys.exit(1)

    # Apply sample restriction
    if sample != 'full':
        print(f"\n  Applying sample restriction: {sample}")
        # Add custom sample restrictions here
        # Example: df = df[df['some_condition'] == True]

    # Determine specifications to run
    if run_all:
        specs_to_run = list(SPECIFICATIONS.keys())
    else:
        if specification not in SPECIFICATIONS:
            print(f"\n  ERROR: Unknown specification: {specification}")
            print(f"  Available: {list(SPECIFICATIONS.keys())}")
            sys.exit(1)
        specs_to_run = [specification]

    # Run estimations
    results = []
    print(f"\n  Running {len(specs_to_run)} specification(s)...")

    for spec_name in specs_to_run:
        spec = SPECIFICATIONS[spec_name]
        print(f"\n  {spec['name']}:")
        print(f"    {spec['description']}")

        try:
            result = run_fe_estimation(df, spec_name)
            results.append(result)

            print(f"    Coefficient: {result.format_coefficient()}")
            print(f"    95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
            print(f"    N: {result.n_obs:,}")
            print(f"    R²: {result.r_squared:.4f}")

        except Exception as e:
            print(f"    ERROR: {e}")

    # Save results
    if results:
        ensure_dir(diag_dir)

        # Save detailed results
        results_df = pd.DataFrame([r.to_dict() for r in results])
        save_diagnostic(results_df, 'estimation_results')

        # Save coefficient table
        coef_df = results_df[['specification', 'coefficient', 'std_error', 'p_value', 'n_obs']]
        save_diagnostic(coef_df, 'coefficients')

        print(f"\n  Results saved to: {diag_dir}")

    # Summary
    print("\n" + "-" * 60)
    print("ESTIMATION SUMMARY")
    print("-" * 60)
    print(f"  Specifications run: {len(results)}")
    print(f"  Sample: {sample}")

    if results:
        print("\n  Results:")
        print(f"  {'Specification':<20} {'Coef':>10} {'SE':>10} {'p-value':>10}")
        print("  " + "-" * 50)
        for r in results:
            print(f"  {r.specification:<20} {r.coefficient:>10.4f} {r.std_error:>10.4f} {r.p_value:>10.4f}")

    # Generate QA report
    metrics = QAMetrics()
    metrics.add('n_specifications', len(results))
    metrics.add('sample', sample)
    if results:
        metrics.add('n_obs', results[0].n_obs)
        significant = sum(1 for r in results if r.p_value < 0.05)
        metrics.add('n_significant_05', significant)
    generate_qa_report('s03_estimation', metrics)

    print("\n" + "=" * 60)
    print("Stage 03 complete.")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()
