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

        # QA Settings
        ENABLE_QA_REPORTS,
        QA_REPORTS_DIR,

        # Methodological Parameters
        SIGNIFICANCE_LEVEL,
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
    for path in [DATA_RAW_DIR, DATA_WORK_DIR, DIAGNOSTICS_DIR, FIGURES_DIR, QA_REPORTS_DIR, CACHE_DIR]:
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
