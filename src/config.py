#!/usr/bin/env python3
"""
Configuration constants for CENTAUR platform.

This module centralizes paths, methodological parameters, and pipeline settings.
Project-specific values should be customized when adopting for a new research project.

Usage
-----
    from config import PROJECT_ROOT, DATA_WORK_DIR, ENABLE_QA_REPORTS

    # Or import specific sections
    from config import (
        # Paths
        PROJECT_ROOT,
        DATA_RAW_DIR,
        DATA_WORK_DIR,
        FIGURES_DIR,
        MANUSCRIPT_DIR,
        DRAFTS_DIR,

        # QA Settings
        ENABLE_QA_REPORTS,
        QA_REPORTS_DIR,

        # Methodological Parameters
        SIGNIFICANCE_LEVEL,

        # LLM Settings
        LLM_PROVIDER,
        LLM_MODELS,
        LLM_TEMPERATURE,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


# =============================================================================
# PATHS
# =============================================================================

def _find_project_root() -> Path:
    """Find project root by looking for characteristic directories."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'src').exists() and (parent / 'manuscript_quarto').exists():
            return parent
    # Fallback: use parent of src/
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _find_project_root()

# Data directories
DATA_RAW_DIR = PROJECT_ROOT / 'data_raw'
DATA_WORK_DIR = PROJECT_ROOT / 'data_work'
DIAGNOSTICS_DIR = DATA_WORK_DIR / 'diagnostics'

# Output directories
FIGURES_DIR = PROJECT_ROOT / 'figures'
MANUSCRIPT_DIR = PROJECT_ROOT / 'manuscript_quarto'
MANUSCRIPT_FIGURES_DIR = MANUSCRIPT_DIR / 'figures'

# Documentation
DOC_DIR = PROJECT_ROOT / 'doc'
REVIEWS_DIR = DOC_DIR / 'reviews'


# =============================================================================
# QUALITY ASSURANCE
# =============================================================================

# Enable per-stage QA report generation
ENABLE_QA_REPORTS = True

# Output directory for QA reports
QA_REPORTS_DIR = DATA_WORK_DIR / 'quality'

# QA thresholds (customize per project)
QA_THRESHOLDS = {
    'max_missing_pct': 5.0,       # Warn if >5% missing values
    'min_row_count': 10,          # Warn if fewer than 10 rows
    'max_duplicate_pct': 1.0,     # Warn if >1% duplicate rows
}


# =============================================================================
# CACHING SETTINGS
# =============================================================================

# Enable pipeline caching for faster re-runs
CACHE_ENABLED = True

# Cache directory (relative to DATA_WORK_DIR)
CACHE_DIR = DATA_WORK_DIR / '.cache'

# Maximum cache age in hours before automatic invalidation
CACHE_MAX_AGE_HOURS = 168  # 1 week

# Use compression for large cached objects
CACHE_COMPRESSION = False


# =============================================================================
# PARALLEL EXECUTION SETTINGS
# =============================================================================

# Enable parallel execution within stages
PARALLEL_ENABLED = True

# Maximum number of parallel workers (None = use CPU count)
PARALLEL_MAX_WORKERS = None

# Stages that support parallel execution
PARALLEL_STAGES = ['s03_estimation', 's04_robustness', 's05_figures']


# =============================================================================
# SPATIAL CROSS-VALIDATION SETTINGS (Optional)
# =============================================================================

# Number of spatial groups for cross-validation
SPATIAL_CV_N_GROUPS = 5

# Default grouping method
# Options: 'kmeans', 'balanced_kmeans', 'geographic_bands', 'longitude_bands',
#          'spatial_blocks', 'zip_digit', 'contiguity_queen', 'contiguity_rook'
# Note: contiguity methods require geopandas
SPATIAL_GROUPING_METHOD = 'kmeans'

# Methods to test for spatial sensitivity analysis
SPATIAL_SENSITIVITY_METHODS = [
    'kmeans',
    'balanced_kmeans',
    'geographic_bands',
    'longitude_bands',
    'spatial_blocks',
]

# Random state for spatial grouping reproducibility
RANDOM_STATE = 42


# =============================================================================
# GEOSPATIAL SETTINGS
# =============================================================================

# Enable geospatial features (requires geopandas, shapely, pyproj)
SPATIAL_ENABLED = True

# Default coordinate reference system
SPATIAL_DEFAULT_CRS = "EPSG:4326"  # WGS84

# Spatial data directory
SPATIAL_DATA_DIR = DATA_WORK_DIR / 'spatial'

# Geocoding cache directory
SPATIAL_CACHE_DIR = CACHE_DIR / 'spatial'

# Geocoding settings (for future phases)
GEOCODING_PROVIDER = "census"  # 'census', 'nominatim', 'google', 'here'
GEOCODING_CACHE_TTL_DAYS = 30


# =============================================================================
# ML MODEL HYPERPARAMETER GRIDS (for tuning)
# =============================================================================

# Ridge regression alpha values
TUNING_RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]

# ElasticNet parameters
TUNING_ENET_ALPHAS = [0.01, 0.1, 1.0, 10.0]
TUNING_ENET_L1_RATIOS = [0.1, 0.5, 0.9]

# Random Forest parameters
TUNING_RF_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}

# Extra Trees parameters
TUNING_ET_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}

# Gradient Boosting parameters
TUNING_GB_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
}

# Inner CV folds for nested cross-validation
TUNING_INNER_FOLDS = 3


# =============================================================================
# REPEATED CROSS-VALIDATION SETTINGS
# =============================================================================

# Number of splits for repeated k-fold CV
REPEATED_CV_N_SPLITS = 5

# Number of repeats for repeated k-fold CV
REPEATED_CV_N_REPEATS = 10


# =============================================================================
# PIPELINE SETTINGS
# =============================================================================

# Default estimation specification
DEFAULT_SPECIFICATION = 'baseline'

# Available specifications (customize per project)
SPECIFICATIONS = {
    'baseline': 'Main specification with all controls',
    'no_fe': 'No fixed effects',
    'with_controls': 'With additional control variables',
    'unit_fe_only': 'Only unit fixed effects',
}


# =============================================================================
# METHODOLOGICAL PARAMETERS
# =============================================================================

# Statistical thresholds
SIGNIFICANCE_LEVEL = 0.05
CONFIDENCE_LEVEL = 0.95

# Bootstrap settings
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 42

# Winsorization (if applicable)
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

# Sample restrictions (customize per project)
MIN_OBSERVATIONS = 30  # Minimum obs per group for reliable estimates


# =============================================================================
# FILE NAMING CONVENTIONS
# =============================================================================

# Pipeline outputs
PANEL_FILE = 'panel.parquet'
ESTIMATES_FILE = 'estimates.parquet'
ROBUSTNESS_FILE = 'robustness.parquet'

# Full paths
PANEL_PATH = DATA_WORK_DIR / PANEL_FILE
ESTIMATES_PATH = DATA_WORK_DIR / ESTIMATES_FILE
ROBUSTNESS_PATH = DATA_WORK_DIR / ROBUSTNESS_FILE


# =============================================================================
# MANUSCRIPT SETTINGS
# =============================================================================

# Default manuscript configuration
MANUSCRIPTS = {
    'main': {
        'name': 'Main Manuscript',
        'dir': MANUSCRIPT_DIR,
        'reviews_dir': REVIEWS_DIR / 'main',
        'archive_dir': REVIEWS_DIR / 'main' / 'archive',
    },
}

DEFAULT_MANUSCRIPT = 'main'

# Drafts directory for AI-generated content
DRAFTS_DIR = MANUSCRIPT_DIR / 'drafts'


# =============================================================================
# LLM SETTINGS (AI-Assisted Writing)
# =============================================================================

# Default LLM provider ('anthropic' or 'openai')
LLM_PROVIDER = 'anthropic'

# Model configuration per provider
LLM_MODELS = {
    'anthropic': 'claude-sonnet-4-20250514',
    'openai': 'gpt-4-turbo-preview',
}

# Current model (based on default provider)
LLM_MODEL = LLM_MODELS.get(LLM_PROVIDER, 'claude-sonnet-4-20250514')

# Generation settings
LLM_TEMPERATURE = 0.3  # Lower = more deterministic
LLM_MAX_TOKENS = 4096  # Maximum tokens per response


# =============================================================================
# REVIEW MANAGEMENT SETTINGS
# =============================================================================

# Default source type for new reviews ('synthetic' or 'actual')
REVIEW_DEFAULT_SOURCE_TYPE = 'synthetic'

# Git integration for review cycles
REVIEW_GIT_TAGGING_ENABLED = True
REVIEW_GIT_TAG_FORMAT = 'review-{manuscript}-{cycle:02d}-{status}'

# Response letter settings
REVIEW_RESPONSE_DEFAULT_FORMAT = 'markdown'
REVIEW_RESPONSE_TEMPLATE_DIR = DOC_DIR / 'reviews' / 'templates'

# Diff generation settings
REVIEW_DIFF_DEFAULT_FORMAT = 'markdown'
REVIEW_DIFF_CONTEXT_LINES = 3

# Archive structure (True = directory per review, False = legacy single file)
REVIEW_USE_DIRECTORY_ARCHIVE = False  # Keep False for backward compat initially


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config() -> bool:
    """
    Validate configuration settings.

    Returns
    -------
    bool
        True if all validations pass

    Raises
    ------
    ValueError
        If any configuration is invalid
    """
    errors = []

    # Check required directories exist or can be created
    for name, path in [
        ('PROJECT_ROOT', PROJECT_ROOT),
    ]:
        if not path.exists():
            errors.append(f"{name} does not exist: {path}")

    # Check thresholds are valid
    if SIGNIFICANCE_LEVEL <= 0 or SIGNIFICANCE_LEVEL >= 1:
        errors.append(f"SIGNIFICANCE_LEVEL must be between 0 and 1: {SIGNIFICANCE_LEVEL}")

    if BOOTSTRAP_ITERATIONS < 1:
        errors.append(f"BOOTSTRAP_ITERATIONS must be positive: {BOOTSTRAP_ITERATIONS}")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    for path in [DATA_RAW_DIR, DATA_WORK_DIR, DIAGNOSTICS_DIR, FIGURES_DIR, QA_REPORTS_DIR, CACHE_DIR, DRAFTS_DIR, SPATIAL_DATA_DIR, SPATIAL_CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def get_manuscript_config(manuscript: str = None) -> dict:
    """
    Get configuration for a manuscript.

    Parameters
    ----------
    manuscript : str, optional
        Manuscript name. Defaults to DEFAULT_MANUSCRIPT.

    Returns
    -------
    dict
        Manuscript configuration

    Raises
    ------
    ValueError
        If manuscript name is not found
    """
    if manuscript is None:
        manuscript = DEFAULT_MANUSCRIPT

    if manuscript not in MANUSCRIPTS:
        available = ', '.join(MANUSCRIPTS.keys())
        raise ValueError(f"Unknown manuscript '{manuscript}'. Available: {available}")

    return MANUSCRIPTS[manuscript]


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

# For backward compatibility with utils.helpers
def get_project_root() -> Path:
    """Get project root directory (compatibility wrapper)."""
    return PROJECT_ROOT


def get_data_dir(subdir: str = 'work') -> Path:
    """Get data directory (compatibility wrapper)."""
    if subdir == 'raw':
        return DATA_RAW_DIR
    elif subdir == 'work':
        return DATA_WORK_DIR
    elif subdir == 'diagnostics':
        return DIAGNOSTICS_DIR
    else:
        return PROJECT_ROOT / f'data_{subdir}'


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

if __name__ == '__main__':
    # Print configuration when run directly
    print("CENTAUR Configuration")
    print("=" * 50)
    print(f"PROJECT_ROOT:      {PROJECT_ROOT}")
    print(f"DATA_RAW_DIR:      {DATA_RAW_DIR}")
    print(f"DATA_WORK_DIR:     {DATA_WORK_DIR}")
    print(f"FIGURES_DIR:       {FIGURES_DIR}")
    print(f"MANUSCRIPT_DIR:    {MANUSCRIPT_DIR}")
    print()
    print(f"ENABLE_QA_REPORTS: {ENABLE_QA_REPORTS}")
    print(f"QA_REPORTS_DIR:    {QA_REPORTS_DIR}")
    print()
    print("Validating configuration...")
    try:
        validate_config()
        print("Configuration valid.")
    except ValueError as e:
        print(f"Configuration invalid:\n{e}")
