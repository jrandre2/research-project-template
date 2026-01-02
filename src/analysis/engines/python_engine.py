"""
Python/NumPy Analysis Engine.

Native Python implementation using NumPy for matrix operations.
This is the default engine and serves as the reference implementation.

Usage
-----
    from analysis import get_engine

    engine = get_engine('python')
    result = engine.estimate(data_path, specification, output_dir)
"""
from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..base import BaseAnalysisEngine, EstimationResult
from ..factory import register_engine


@register_engine('python')
class PythonEngine(BaseAnalysisEngine):
    """
    Native Python/NumPy estimation engine.

    Uses NumPy for matrix operations and implements fixed effects
    estimation via the within transformation (demeaning).

    Features
    --------
    - Fixed effects via demeaning
    - Clustered standard errors
    - Parallel execution for batch estimation
    """

    def __init__(self):
        super().__init__()
        self._version = f"numpy {np.__version__}"

    @property
    def name(self) -> str:
        return 'python'

    @property
    def version(self) -> str:
        return self._version

    def validate_installation(self) -> tuple[bool, str]:
        """Check that NumPy and Pandas are available."""
        try:
            import numpy as np
            import pandas as pd
            return True, f"Python engine ready (numpy {np.__version__}, pandas {pd.__version__})"
        except ImportError as e:
            return False, f"Missing dependency: {e}"

    def estimate(
        self,
        data_path: Path,
        specification: dict,
        output_dir: Path,
    ) -> EstimationResult:
        """
        Run fixed effects estimation for a single specification.

        Parameters
        ----------
        data_path : Path
            Path to input data (Parquet format)
        specification : dict
            Specification with keys: name, outcome, treatment, controls,
            fixed_effects, cluster
        output_dir : Path
            Directory for output files

        Returns
        -------
        EstimationResult
            Estimation results
        """
        start_time = time.time()

        # Load data
        df = pd.read_parquet(data_path)

        # Extract specification
        spec_name = specification.get('name', 'unnamed')
        outcome_var = specification['outcome']
        treatment_var = specification['treatment']
        controls = specification.get('controls', [])
        fixed_effects = specification.get('fixed_effects', [])
        cluster_var = specification.get('cluster')

        x_vars = [treatment_var] + controls

        # Filter to valid observations
        all_vars = [outcome_var] + x_vars + fixed_effects
        if cluster_var:
            all_vars.append(cluster_var)

        # Only keep columns that exist in the data
        available_vars = [v for v in all_vars if v in df.columns]
        df_valid = df[available_vars].dropna().copy()

        if len(df_valid) == 0:
            raise ValueError("No valid observations after filtering for missing values")

        # Demean by fixed effects
        if fixed_effects:
            available_fe = [fe for fe in fixed_effects if fe in df_valid.columns]
            if available_fe:
                df_est = self._demean_by_fe(df_valid, outcome_var, x_vars, available_fe)
            else:
                df_est = df_valid
        else:
            df_est = df_valid

        # Run OLS on demeaned data
        ols_result = self._run_ols(
            df_est,
            outcome_var,
            x_vars,
            cluster_var=cluster_var if cluster_var in df_est.columns else None,
        )

        # Count units and periods (if applicable columns exist)
        n_units = 0
        n_periods = 0
        if 'id' in df_valid.columns:
            n_units = df_valid['id'].nunique()
        elif fixed_effects and fixed_effects[0] in df_valid.columns:
            n_units = df_valid[fixed_effects[0]].nunique()

        if 'period' in df_valid.columns:
            n_periods = df_valid['period'].nunique()
        elif len(fixed_effects) > 1 and fixed_effects[1] in df_valid.columns:
            n_periods = df_valid[fixed_effects[1]].nunique()

        execution_time = time.time() - start_time

        return EstimationResult(
            specification=spec_name,
            n_obs=ols_result['n_obs'],
            n_units=n_units,
            n_periods=n_periods,
            coefficients={treatment_var: ols_result['coefficient']},
            std_errors={treatment_var: ols_result['std_error']},
            t_stats={treatment_var: ols_result['t_stat']},
            p_values={treatment_var: ols_result['p_value']},
            ci_lower={treatment_var: ols_result['ci_lower']},
            ci_upper={treatment_var: ols_result['ci_upper']},
            r_squared=ols_result['r_squared'],
            fixed_effects=fixed_effects,
            cluster_var=cluster_var,
            controls=controls,
            warnings=[],
            engine='python',
            engine_version=self._version,
            execution_time_seconds=execution_time,
        )

    def estimate_batch(
        self,
        data_path: Path,
        specifications: list[dict],
        output_dir: Path,
        parallel: bool = True,
        n_workers: Optional[int] = None,
    ) -> list[EstimationResult]:
        """
        Run estimation for multiple specifications with parallel support.

        Parameters
        ----------
        data_path : Path
            Path to input data
        specifications : list[dict]
            List of specification dictionaries
        output_dir : Path
            Directory for output files
        parallel : bool
            Whether to run in parallel
        n_workers : int, optional
            Number of parallel workers (default: CPU count)

        Returns
        -------
        list[EstimationResult]
            Results for each specification
        """
        n_specs = len(specifications)

        if not parallel or n_specs == 1:
            # Sequential execution
            return super().estimate_batch(
                data_path, specifications, output_dir, parallel=False
            )

        # Parallel execution
        if n_workers is None:
            n_workers = min(n_specs, multiprocessing.cpu_count())

        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_spec = {
                executor.submit(
                    self.estimate, data_path, spec, output_dir
                ): spec
                for spec in specifications
            }

            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"  ERROR in {spec.get('name', 'unknown')}: {e}")

        return results

    def _run_ols(
        self,
        df: pd.DataFrame,
        y_var: str,
        x_vars: list[str],
        cluster_var: Optional[str] = None,
    ) -> dict:
        """
        Run OLS regression with optional clustered standard errors.

        Parameters
        ----------
        df : pd.DataFrame
            Data (potentially demeaned)
        y_var : str
            Outcome variable
        x_vars : list[str]
            Regressor variables
        cluster_var : str, optional
            Variable for clustered standard errors

        Returns
        -------
        dict
            Regression results with keys: coefficient, std_error, t_stat,
            p_value, ci_lower, ci_upper, n_obs, r_squared
        """
        # Prepare data
        all_vars = [y_var] + x_vars
        if cluster_var and cluster_var in df.columns:
            all_vars.append(cluster_var)
        df_clean = df[all_vars].dropna()

        n = len(df_clean)
        if n < len(x_vars) + 1:
            raise ValueError(f"Insufficient observations: {n}")

        # Create design matrix with intercept
        y = df_clean[y_var].values
        X_cols = [np.ones(n)]
        for x in x_vars:
            if x in df_clean.columns:
                X_cols.append(df_clean[x].values)
        X = np.column_stack(X_cols)

        # OLS estimation: beta = (X'X)^-1 X'y
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse for singular matrices
            XtX_inv = np.linalg.pinv(XtX)

        beta = XtX_inv @ X.T @ y

        # Residuals and R-squared
        y_hat = X @ beta
        residuals = y - y_hat
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Standard errors
        if cluster_var and cluster_var in df_clean.columns:
            se = self._clustered_se(X, residuals, df_clean[cluster_var].values, XtX_inv)
        else:
            # Homoskedastic standard errors
            dof = n - X.shape[1]
            s2 = ss_res / dof if dof > 0 else ss_res
            V = s2 * XtX_inv
            se = np.sqrt(np.diag(V))

        # Treatment coefficient (first regressor after constant)
        coef = beta[1] if len(beta) > 1 else beta[0]
        se_coef = se[1] if len(se) > 1 else se[0]
        t_stat = coef / se_coef if se_coef > 0 else 0.0
        dof = n - X.shape[1]
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), dof))

        # Confidence interval (95%)
        t_crit = self._t_ppf(0.975, dof)
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
            'r_squared': r_squared,
        }

    def _clustered_se(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        clusters: np.ndarray,
        XtX_inv: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cluster-robust standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        residuals : np.ndarray
            OLS residuals
        clusters : np.ndarray
            Cluster identifiers
        XtX_inv : np.ndarray
            Inverse of X'X

        Returns
        -------
        np.ndarray
            Standard errors
        """
        n = len(residuals)
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters)

        # Compute meat of sandwich estimator
        meat = np.zeros((X.shape[1], X.shape[1]))
        for c in unique_clusters:
            mask = clusters == c
            u_c = residuals[mask]
            X_c = X[mask]
            # Sum of X'u within cluster
            Xu_c = (X_c.T * u_c).sum(axis=1)
            meat += np.outer(Xu_c, Xu_c)

        # Small sample correction
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - X.shape[1]))
        V = correction * XtX_inv @ meat @ XtX_inv
        se = np.sqrt(np.diag(V))

        return se

    def _demean_by_fe(
        self,
        df: pd.DataFrame,
        y_var: str,
        x_vars: list[str],
        fe_vars: list[str],
    ) -> pd.DataFrame:
        """
        Demean variables by fixed effects (within transformation).

        Parameters
        ----------
        df : pd.DataFrame
            Data
        y_var : str
            Outcome variable
        x_vars : list[str]
            Regressor variables
        fe_vars : list[str]
            Fixed effect variables

        Returns
        -------
        pd.DataFrame
            Demeaned data
        """
        df_demeaned = df.copy()
        vars_to_demean = [y_var] + [x for x in x_vars if x in df.columns]

        for fe in fe_vars:
            if fe not in df_demeaned.columns:
                continue
            for var in vars_to_demean:
                if var in df_demeaned.columns:
                    group_mean = df_demeaned.groupby(fe)[var].transform('mean')
                    df_demeaned[var] = df_demeaned[var] - group_mean

        return df_demeaned

    @staticmethod
    def _t_cdf(x: float, df: int) -> float:
        """Approximate t-distribution CDF."""
        if df > 100:
            # Normal approximation for large df
            from math import erf, sqrt
            return 0.5 * (1 + erf(x / sqrt(2)))

        # Simple approximation for smaller df
        t = x / np.sqrt(df)
        return 0.5 + 0.5 * np.sign(t) * (1 - (1 + t**2/df)**(-df/2))

    @staticmethod
    def _t_ppf(p: float, df: int) -> float:
        """Approximate t-distribution PPF (inverse CDF)."""
        if df > 100:
            # Normal approximation
            a = 8 * (np.pi - 3) / (3 * np.pi * (4 - np.pi))
            x = 2 * p - 1
            inner = (2 / (np.pi * a) + np.log(1 - x**2) / 2)
            return np.sign(x) * np.sqrt(np.sqrt(inner**2 - np.log(1 - x**2) / a) - inner)

        # For small df, use approximation
        return 1.96 * (1 + 0.5 / df)
