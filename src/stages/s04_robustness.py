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

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Callable
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
from utils.cache import CacheManager, hash_dataframe, hash_config
from stages._qa_utils import generate_qa_report, QAMetrics

# Import config settings
try:
    from config import CACHE_ENABLED, PARALLEL_ENABLED, PARALLEL_MAX_WORKERS
except ImportError:
    CACHE_ENABLED = True
    PARALLEL_ENABLED = True
    PARALLEL_MAX_WORKERS = None


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

def _run_test_wrapper(args: tuple) -> list[RobustnessResult]:
    """
    Worker function for parallel test execution.

    Parameters
    ----------
    args : tuple
        (test_func, df, test_name) - function, data, and name

    Returns
    -------
    list[RobustnessResult]
        Test results
    """
    test_func, df, test_name = args
    try:
        return test_func(df)
    except Exception as e:
        print(f"    ERROR in {test_name}: {e}")
        return []


def main(
    run_placebos: bool = True,
    run_samples: bool = True,
    run_specs: bool = True,
    verbose: bool = True,
    use_cache: bool = True,
    parallel: bool = True,
    n_workers: Optional[int] = None,
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
    use_cache : bool
        Use caching for results (default: True)
    parallel : bool
        Use parallel execution for tests (default: True)
    n_workers : int, optional
        Number of parallel workers (default: CPU count)
    """
    start_time = time.time()

    print("=" * 60)
    print("Stage 04: Robustness Checks")
    print("=" * 60)

    # Initialize cache
    cache = CacheManager('s04_robustness', enabled=use_cache and CACHE_ENABLED)
    cache_stats = {'hits': 0, 'misses': 0}

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

    # Compute data hash for cache keys
    data_hash = hash_dataframe(df) if use_cache else None

    # Determine which tests to run
    tests_to_run = []
    if run_specs:
        tests_to_run.append(('specs', run_alternative_specs, 'alternative specifications'))
    if run_placebos:
        tests_to_run.append(('placebos', run_placebo_time, 'placebo tests'))
    if run_samples:
        tests_to_run.append(('samples', run_sample_restrictions, 'sample restrictions'))

    # Determine execution mode
    n_tests = len(tests_to_run)
    use_parallel = parallel and PARALLEL_ENABLED and n_tests > 1
    if n_workers is None:
        n_workers = PARALLEL_MAX_WORKERS or min(n_tests, multiprocessing.cpu_count())

    all_results = []

    if use_parallel and n_tests > 1:
        print(f"\n  Running {n_tests} test groups in parallel ({n_workers} workers)...")

        # Check cache for all test groups first
        tests_to_compute = []
        for test_key, test_func, test_desc in tests_to_run:
            cache_key = f"results_{test_key}"
            depends_on = {'data': data_hash, 'test': test_key}

            found, cached_results = cache.get(cache_key, depends_on)
            if found and cached_results is not None:
                all_results.extend(cached_results)
                cache_stats['hits'] += 1
                print(f"    [cache hit] {test_desc}: {len(cached_results)} tests")
            else:
                tests_to_compute.append((test_key, test_func, test_desc))
                cache_stats['misses'] += 1

        # Run remaining tests in parallel
        if tests_to_compute:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_test = {
                    executor.submit(_run_test_wrapper, (test_func, df, test_desc)): (test_key, test_desc)
                    for test_key, test_func, test_desc in tests_to_compute
                }

                for future in as_completed(future_to_test):
                    test_key, test_desc = future_to_test[future]
                    try:
                        results = future.result()
                        if results:
                            all_results.extend(results)
                            # Cache results
                            cache_key = f"results_{test_key}"
                            depends_on = {'data': data_hash, 'test': test_key}
                            cache.set(cache_key, results, depends_on)
                            print(f"    Completed {test_desc}: {len(results)} tests")
                    except Exception as e:
                        print(f"    ERROR in {test_desc}: {e}")

    else:
        # Sequential execution with caching
        mode_desc = "sequential with caching" if use_cache else "sequential"
        print(f"\n  Running tests ({mode_desc})...")

        for test_key, test_func, test_desc in tests_to_run:
            # Check cache first
            cache_key = f"results_{test_key}"
            depends_on = {'data': data_hash, 'test': test_key} if data_hash else None

            if use_cache and depends_on:
                found, cached_results = cache.get(cache_key, depends_on)
                if found and cached_results is not None:
                    all_results.extend(cached_results)
                    cache_stats['hits'] += 1
                    print(f"\n  {test_desc.title()}: [cache hit] {len(cached_results)} tests")
                    continue

            cache_stats['misses'] += 1
            print(f"\n  Running {test_desc}...")

            try:
                results = test_func(df)
                all_results.extend(results)

                # Cache results
                if use_cache and depends_on:
                    cache.set(cache_key, results, depends_on)

                print(f"    Completed {len(results)} tests")
            except Exception as e:
                print(f"    ERROR: {e}")

    # Save results
    if all_results:
        results_df = pd.DataFrame([r.to_dict() for r in all_results])
        save_diagnostic(results_df, 'robustness_results')
        print(f"\n  Results saved to: {diag_dir}")

    # Timing
    elapsed = time.time() - start_time

    # Summary
    print("\n" + "-" * 60)
    print("ROBUSTNESS SUMMARY")
    print("-" * 60)
    print(f"  Total tests: {len(all_results)}")
    print(f"  Elapsed time: {elapsed:.2f}s")
    if use_cache:
        hit_rate = cache_stats['hits'] / (cache_stats['hits'] + cache_stats['misses']) * 100 if (cache_stats['hits'] + cache_stats['misses']) > 0 else 0
        print(f"  Cache: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({hit_rate:.0f}% hit rate)")

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
    metrics.add('elapsed_sec', round(elapsed, 2))
    metrics.add('cache_hits', cache_stats['hits'])
    metrics.add('cache_misses', cache_stats['misses'])
    metrics.add('parallel_workers', n_workers if use_parallel else 1)
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
