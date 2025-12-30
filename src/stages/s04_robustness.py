#!/usr/bin/env python3
"""
Stage 04: Robustness Checks

Purpose: Run robustness specifications and sensitivity analyses.

This stage handles:
- Alternative specifications
- Placebo tests (time and treatment group)
- Sample restriction tests
- Alternative standard error methods
- Spatial cross-validation vs random CV comparison (if geographic data available)
- Feature ablation studies
- Hyperparameter tuning with nested cross-validation
- Tree-based model comparisons

Input Files
-----------
- data_work/panel.parquet

Output Files
------------
- data_work/diagnostics/robustness_results.csv
- data_work/diagnostics/placebo_results.csv
- data_work/diagnostics/spatial_cv_results.csv (optional)
- data_work/diagnostics/ablation_results.csv (optional)
- data_work/diagnostics/tuning_results.csv (optional)

Usage
-----
    python src/pipeline.py estimate_robustness
    python src/pipeline.py estimate_robustness --spatial-cv
    python src/pipeline.py estimate_robustness --ablation
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
    from config import (
        CACHE_ENABLED, PARALLEL_ENABLED, PARALLEL_MAX_WORKERS,
        SPATIAL_CV_N_GROUPS, SPATIAL_GROUPING_METHOD, RANDOM_STATE,
        TUNING_RIDGE_ALPHAS, TUNING_ENET_ALPHAS, TUNING_ENET_L1_RATIOS,
        TUNING_RF_PARAMS, TUNING_ET_PARAMS, TUNING_GB_PARAMS,
        TUNING_INNER_FOLDS, REPEATED_CV_N_SPLITS, REPEATED_CV_N_REPEATS,
    )
except ImportError:
    CACHE_ENABLED = True
    PARALLEL_ENABLED = True
    PARALLEL_MAX_WORKERS = None
    SPATIAL_CV_N_GROUPS = 5
    SPATIAL_GROUPING_METHOD = 'kmeans'
    RANDOM_STATE = 42
    TUNING_RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
    TUNING_ENET_ALPHAS = [0.01, 0.1, 1.0, 10.0]
    TUNING_ENET_L1_RATIOS = [0.1, 0.5, 0.9]
    TUNING_RF_PARAMS = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    TUNING_ET_PARAMS = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    TUNING_GB_PARAMS = {'n_estimators': [100, 200], 'max_depth': [3, 5]}
    TUNING_INNER_FOLDS = 3
    REPEATED_CV_N_SPLITS = 5
    REPEATED_CV_N_REPEATS = 10

# Optional: scikit-learn models (required for ML-based robustness checks)
try:
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import (
        RandomForestRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import (
        cross_val_score,
        RepeatedKFold,
        GroupKFold,
        GridSearchCV,
    )
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional: Spatial CV module
try:
    from utils.spatial_cv import SpatialCVManager, compare_spatial_vs_random_cv
    SPATIAL_CV_AVAILABLE = True
except ImportError:
    SPATIAL_CV_AVAILABLE = False


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
# SPATIAL CROSS-VALIDATION TESTS
# ============================================================

def run_spatial_cv_comparison(
    df: pd.DataFrame,
    feature_cols: list[str],
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    alpha: float = 1.0,
) -> list[RobustnessResult]:
    """
    Compare spatial CV to random CV to quantify geographic data leakage.

    This test helps identify when standard cross-validation is overly
    optimistic due to spatial autocorrelation in the data.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with geographic coordinates
    feature_cols : list[str]
        Feature columns to use for prediction
    lat_col : str
        Name of latitude column
    lon_col : str
        Name of longitude column
    alpha : float
        Ridge regularization parameter

    Returns
    -------
    list[RobustnessResult]
        Comparison results with leakage estimates
    """
    results = []

    if not SKLEARN_AVAILABLE or not SPATIAL_CV_AVAILABLE:
        print("    Skipping: sklearn or spatial_cv module not available")
        return results

    # Check for required columns
    if lat_col not in df.columns or lon_col not in df.columns:
        print(f"    Skipping: missing geographic columns ({lat_col}, {lon_col})")
        return results

    if OUTCOME_VAR not in df.columns:
        print(f"    Skipping: missing outcome column ({OUTCOME_VAR})")
        return results

    available_features = [c for c in feature_cols if c in df.columns]
    if not available_features:
        print("    Skipping: no feature columns found")
        return results

    # Prepare data
    df_clean = df[[OUTCOME_VAR, lat_col, lon_col] + available_features].dropna()
    if len(df_clean) < 50:
        print(f"    Skipping: insufficient observations ({len(df_clean)})")
        return results

    X = df_clean[available_features].values
    y = df_clean[OUTCOME_VAR].values
    lats = df_clean[lat_col].values
    lons = df_clean[lon_col].values

    # Create model
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    # Compare spatial vs random CV
    try:
        comparison = compare_spatial_vs_random_cv(
            model, X, y, lats, lons,
            n_groups=SPATIAL_CV_N_GROUPS,
            method=SPATIAL_GROUPING_METHOD,
            verbose=False,
        )

        results.append(RobustnessResult(
            test_name='spatial_vs_random_cv',
            test_type='spatial',
            coefficient=comparison['leakage'],
            std_error=comparison['spatial_cv']['std'],
            p_value=0.0,  # Not applicable
            n_obs=len(df_clean),
            description=(
                f"Leakage estimate: random CV R2={comparison['random_cv']['mean']:.3f}, "
                f"spatial CV R2={comparison['spatial_cv']['mean']:.3f}"
            )
        ))
        print(f"    Spatial CV: R2={comparison['spatial_cv']['mean']:.3f} ± {comparison['spatial_cv']['std']:.3f}")
        print(f"    Random CV:  R2={comparison['random_cv']['mean']:.3f} ± {comparison['random_cv']['std']:.3f}")
        print(f"    Leakage:    {comparison['leakage']:+.3f} ({comparison['leakage_pct']:.1f}%)")

    except Exception as e:
        print(f"    ERROR in spatial CV comparison: {e}")

    return results


def run_feature_ablation(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_sets: dict[str, list[str]] = None,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    alpha: float = 1.0,
) -> list[RobustnessResult]:
    """
    Run feature ablation studies to assess feature importance.

    Tests model performance with different feature subsets to understand
    which features contribute most to predictive power.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    feature_cols : list[str]
        All feature columns available
    feature_sets : dict[str, list[str]], optional
        Named feature subsets to test. If None, uses default sets.
    lat_col : str
        Name of latitude column (for spatial CV)
    lon_col : str
        Name of longitude column (for spatial CV)
    alpha : float
        Ridge regularization parameter

    Returns
    -------
    list[RobustnessResult]
        Ablation study results
    """
    results = []

    if not SKLEARN_AVAILABLE:
        print("    Skipping: sklearn not available")
        return results

    if OUTCOME_VAR not in df.columns:
        print(f"    Skipping: missing outcome column ({OUTCOME_VAR})")
        return results

    available_features = [c for c in feature_cols if c in df.columns]
    if not available_features:
        print("    Skipping: no feature columns found")
        return results

    # Default feature sets if none provided
    if feature_sets is None:
        feature_sets = {
            'all_features': available_features,
            'first_half': available_features[:len(available_features)//2],
            'second_half': available_features[len(available_features)//2:],
            'top_5': available_features[:5] if len(available_features) >= 5 else available_features,
        }

    # Prepare data
    df_clean = df[[OUTCOME_VAR] + available_features].dropna()
    if len(df_clean) < 50:
        print(f"    Skipping: insufficient observations ({len(df_clean)})")
        return results

    y = df_clean[OUTCOME_VAR].values

    # Use spatial CV if coordinates available
    use_spatial = (
        SPATIAL_CV_AVAILABLE and
        lat_col in df.columns and
        lon_col in df.columns
    )

    if use_spatial:
        df_clean = df[[OUTCOME_VAR, lat_col, lon_col] + available_features].dropna()
        y = df_clean[OUTCOME_VAR].values
        manager = SpatialCVManager(n_groups=SPATIAL_CV_N_GROUPS, method=SPATIAL_GROUPING_METHOD)
        manager.create_groups_from_coordinates(
            df_clean[lat_col].values,
            df_clean[lon_col].values,
            verbose=False,
        )

    for set_name, features in feature_sets.items():
        set_features = [f for f in features if f in df_clean.columns]
        if not set_features:
            continue

        X = df_clean[set_features].values
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

        try:
            if use_spatial:
                cv_result = manager.cross_validate(model, X, y, scale_features=False)
                mean_r2, std_r2 = cv_result['mean'], cv_result['std']
            else:
                scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                mean_r2, std_r2 = np.mean(scores), np.std(scores)

            results.append(RobustnessResult(
                test_name=f'ablation_{set_name}',
                test_type='ablation',
                coefficient=mean_r2,
                std_error=std_r2,
                p_value=0.0,  # Not applicable
                n_obs=len(df_clean),
                description=f'{len(set_features)} features: R2={mean_r2:.3f} ± {std_r2:.3f}'
            ))
            print(f"    {set_name} ({len(set_features)} features): R2={mean_r2:.3f} ± {std_r2:.3f}")

        except Exception as e:
            print(f"    ERROR in ablation {set_name}: {e}")

    return results


def run_tuned_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
) -> list[RobustnessResult]:
    """
    Run nested cross-validation with hyperparameter tuning.

    Tests Ridge, ElasticNet, and tree-based models with proper
    nested CV to avoid overfitting during hyperparameter selection.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    feature_cols : list[str]
        Feature columns to use
    lat_col : str
        Name of latitude column
    lon_col : str
        Name of longitude column

    Returns
    -------
    list[RobustnessResult]
        Tuning results for each model
    """
    results = []

    if not SKLEARN_AVAILABLE:
        print("    Skipping: sklearn not available")
        return results

    if OUTCOME_VAR not in df.columns:
        print(f"    Skipping: missing outcome column ({OUTCOME_VAR})")
        return results

    available_features = [c for c in feature_cols if c in df.columns]
    if not available_features:
        print("    Skipping: no feature columns found")
        return results

    # Prepare data
    df_clean = df[[OUTCOME_VAR] + available_features].dropna()
    if len(df_clean) < 50:
        print(f"    Skipping: insufficient observations ({len(df_clean)})")
        return results

    X = df_clean[available_features].values
    y = df_clean[OUTCOME_VAR].values

    # Use spatial CV if coordinates available
    use_spatial = (
        SPATIAL_CV_AVAILABLE and
        lat_col in df.columns and
        lon_col in df.columns
    )

    if use_spatial:
        df_clean = df[[OUTCOME_VAR, lat_col, lon_col] + available_features].dropna()
        X = df_clean[available_features].values
        y = df_clean[OUTCOME_VAR].values
        manager = SpatialCVManager(n_groups=SPATIAL_CV_N_GROUPS, method=SPATIAL_GROUPING_METHOD)
        groups = manager.create_groups_from_coordinates(
            df_clean[lat_col].values,
            df_clean[lon_col].values,
            verbose=False,
        )
        outer_cv = GroupKFold(n_splits=SPATIAL_CV_N_GROUPS)
        cv_iter = list(outer_cv.split(X, y, groups=groups))
    else:
        cv_iter = list(RepeatedKFold(
            n_splits=REPEATED_CV_N_SPLITS,
            n_repeats=1,
            random_state=RANDOM_STATE
        ).split(X, y))

    # Define models to tune
    model_configs = [
        ('ridge', make_pipeline(StandardScaler(), Ridge()), {'ridge__alpha': TUNING_RIDGE_ALPHAS}),
        ('elasticnet', make_pipeline(StandardScaler(), ElasticNet(max_iter=10000)),
         {'elasticnet__alpha': TUNING_ENET_ALPHAS, 'elasticnet__l1_ratio': TUNING_ENET_L1_RATIOS}),
        ('random_forest', RandomForestRegressor(random_state=RANDOM_STATE), TUNING_RF_PARAMS),
        ('gradient_boosting', GradientBoostingRegressor(random_state=RANDOM_STATE), TUNING_GB_PARAMS),
    ]

    for model_name, base_model, param_grid in model_configs:
        outer_scores = []
        best_params_list = []

        try:
            for train_idx, test_idx in cv_iter:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Inner CV for tuning
                inner_cv = TUNING_INNER_FOLDS
                grid_search = GridSearchCV(
                    base_model, param_grid,
                    cv=inner_cv, scoring='r2', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)

                # Evaluate on held-out fold
                y_pred = grid_search.best_estimator_.predict(X_test)
                outer_scores.append(r2_score(y_test, y_pred))
                best_params_list.append(grid_search.best_params_)

            mean_r2 = np.mean(outer_scores)
            std_r2 = np.std(outer_scores)

            # Get most common best params
            from collections import Counter
            param_counts = Counter(str(p) for p in best_params_list)
            most_common = param_counts.most_common(1)[0][0] if param_counts else "N/A"

            results.append(RobustnessResult(
                test_name=f'tuned_{model_name}',
                test_type='tuning',
                coefficient=mean_r2,
                std_error=std_r2,
                p_value=0.0,  # Not applicable
                n_obs=len(df_clean),
                description=f'Nested CV R2={mean_r2:.3f} ± {std_r2:.3f}'
            ))
            print(f"    {model_name}: R2={mean_r2:.3f} ± {std_r2:.3f}")

        except Exception as e:
            print(f"    ERROR tuning {model_name}: {e}")

    return results


def run_encoding_comparisons(
    df: pd.DataFrame,
    categorical_col: str = None,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
) -> list[RobustnessResult]:
    """
    Compare categorical vs ordinal encoding for treatment variables.

    Tests whether treating a categorical variable as ordinal (numeric)
    or one-hot encoded produces different results.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    categorical_col : str, optional
        Column to test encodings for. Defaults to TREATMENT_VAR.
    lat_col : str
        Name of latitude column
    lon_col : str
        Name of longitude column

    Returns
    -------
    list[RobustnessResult]
        Encoding comparison results
    """
    results = []

    if not SKLEARN_AVAILABLE:
        print("    Skipping: sklearn not available")
        return results

    if categorical_col is None:
        categorical_col = TREATMENT_VAR

    if categorical_col not in df.columns or OUTCOME_VAR not in df.columns:
        print(f"    Skipping: missing required columns")
        return results

    # Prepare data
    df_clean = df[[OUTCOME_VAR, categorical_col]].dropna()
    if len(df_clean) < 50:
        print(f"    Skipping: insufficient observations ({len(df_clean)})")
        return results

    y = df_clean[OUTCOME_VAR].values

    # Use spatial CV if coordinates available
    use_spatial = (
        SPATIAL_CV_AVAILABLE and
        lat_col in df.columns and
        lon_col in df.columns
    )

    if use_spatial:
        df_clean = df[[OUTCOME_VAR, categorical_col, lat_col, lon_col]].dropna()
        y = df_clean[OUTCOME_VAR].values
        manager = SpatialCVManager(n_groups=SPATIAL_CV_N_GROUPS, method=SPATIAL_GROUPING_METHOD)
        manager.create_groups_from_coordinates(
            df_clean[lat_col].values,
            df_clean[lon_col].values,
            verbose=False,
        )

    # Test 1: Ordinal encoding (treat as numeric)
    X_ordinal = df_clean[[categorical_col]].values
    model_ordinal = Ridge(alpha=1.0)

    try:
        if use_spatial:
            cv_result = manager.cross_validate(model_ordinal, X_ordinal, y)
            mean_r2, std_r2 = cv_result['mean'], cv_result['std']
        else:
            scores = cross_val_score(model_ordinal, X_ordinal, y, cv=5, scoring='r2')
            mean_r2, std_r2 = np.mean(scores), np.std(scores)

        results.append(RobustnessResult(
            test_name=f'{categorical_col}_ordinal',
            test_type='encoding',
            coefficient=mean_r2,
            std_error=std_r2,
            p_value=0.0,
            n_obs=len(df_clean),
            description=f'Ordinal encoding: R2={mean_r2:.3f}'
        ))
        print(f"    Ordinal: R2={mean_r2:.3f} ± {std_r2:.3f}")

    except Exception as e:
        print(f"    ERROR in ordinal encoding: {e}")

    # Test 2: One-hot encoding
    try:
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    try:
        X_onehot = encoder.fit_transform(df_clean[[categorical_col]])
        model_onehot = Ridge(alpha=1.0)

        if use_spatial:
            cv_result = manager.cross_validate(model_onehot, X_onehot, y)
            mean_r2, std_r2 = cv_result['mean'], cv_result['std']
        else:
            scores = cross_val_score(model_onehot, X_onehot, y, cv=5, scoring='r2')
            mean_r2, std_r2 = np.mean(scores), np.std(scores)

        results.append(RobustnessResult(
            test_name=f'{categorical_col}_categorical',
            test_type='encoding',
            coefficient=mean_r2,
            std_error=std_r2,
            p_value=0.0,
            n_obs=len(df_clean),
            description=f'One-hot encoding: R2={mean_r2:.3f}'
        ))
        print(f"    Categorical: R2={mean_r2:.3f} ± {std_r2:.3f}")

    except Exception as e:
        print(f"    ERROR in one-hot encoding: {e}")

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
