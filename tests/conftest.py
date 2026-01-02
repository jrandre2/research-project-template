#!/usr/bin/env python3
"""
Shared pytest fixtures for the test suite.

This module provides common fixtures used across test modules including:
- Temporary directories and files
- Sample DataFrames
- Mock configurations
- Project path fixtures
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to Python path for test imports
_project_root = Path(__file__).parent.parent
_src_path = _project_root / 'src'
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil


# ============================================================
# PATH FIXTURES
# ============================================================

@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_data_dir(temp_dir):
    """Create a temporary data directory structure."""
    data_work = temp_dir / 'data_work'
    data_raw = temp_dir / 'data_raw'
    diagnostics = data_work / 'diagnostics'

    data_work.mkdir()
    data_raw.mkdir()
    diagnostics.mkdir()

    return {
        'root': temp_dir,
        'data_work': data_work,
        'data_raw': data_raw,
        'diagnostics': diagnostics
    }


# ============================================================
# DATA FIXTURES
# ============================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a simple sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'id': range(1, n + 1),
        'value': np.random.randn(n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'date': pd.date_range('2020-01-01', periods=n, freq='D')
    })


@pytest.fixture
def panel_df() -> pd.DataFrame:
    """Create a sample panel DataFrame."""
    np.random.seed(42)
    n_units = 50
    n_periods = 24

    units = list(range(1, n_units + 1))
    periods = list(range(1, n_periods + 1))

    data = []
    for unit in units:
        for period in periods:
            data.append({
                'unit_id': unit,
                'period': period,
                'outcome': np.random.randn() + (0.5 if period > 12 and unit <= 25 else 0),
                'treatment': 1 if period > 12 and unit <= 25 else 0,
                'covariate1': np.random.randn(),
                'covariate2': np.random.choice([0, 1], p=[0.7, 0.3])
            })

    return pd.DataFrame(data)


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """Create a DataFrame with missing values for validation testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'id': range(1, n + 1),
        'value': np.random.randn(n),
        'nullable_col': np.random.randn(n)
    })
    # Introduce missing values
    df.loc[10:15, 'nullable_col'] = np.nan
    return df


@pytest.fixture
def df_with_duplicates() -> pd.DataFrame:
    """Create a DataFrame with duplicate rows."""
    df = pd.DataFrame({
        'id': [1, 2, 2, 3, 4, 4, 4, 5],
        'value': [10, 20, 20, 30, 40, 40, 40, 50]
    })
    return df


# ============================================================
# CONFIGURATION FIXTURES
# ============================================================

@pytest.fixture
def sample_config() -> dict:
    """Create a sample configuration dictionary."""
    return {
        'project_name': 'Test Project',
        'version': '0.1.0',
        'stages': {
            's00_raw': 'data_work/data_raw.parquet',
            's01_linked': 'data_work/data_linked.parquet',
            's02_panel': 'data_work/panel.parquet'
        },
        'estimation': {
            'default_specification': 'baseline',
            'cluster_var': 'unit_id'
        }
    }


@pytest.fixture
def journal_config() -> dict:
    """Create a sample journal configuration."""
    return {
        'journal': {
            'name': 'Test Journal',
            'abbreviation': 'TJ'
        },
        'abstract': {
            'max_words': 250
        },
        'keywords': {
            'min': 4,
            'max': 6
        },
        'artwork': {
            'resolution': {
                'line_art_dpi': 1200,
                'halftone_dpi': 300
            }
        }
    }


# ============================================================
# VALIDATION FIXTURES
# ============================================================

@pytest.fixture
def schema_simple() -> dict:
    """Create a simple schema for validation testing."""
    return {
        'id': {'dtype': 'int64', 'required': True},
        'value': {'dtype': 'float64', 'required': True},
        'category': {'dtype': 'object', 'required': False}
    }


@pytest.fixture
def schema_with_constraints() -> dict:
    """Create a schema with value constraints."""
    return {
        'id': {
            'dtype': 'int64',
            'required': True,
            'unique': True
        },
        'value': {
            'dtype': 'float64',
            'required': True,
            'min': -10,
            'max': 10
        },
        'category': {
            'dtype': 'object',
            'required': True,
            'values': {'A', 'B', 'C'}
        }
    }


# ============================================================
# FILE FIXTURES
# ============================================================

@pytest.fixture
def sample_parquet(temp_dir, sample_df) -> Path:
    """Create a sample parquet file."""
    path = temp_dir / 'sample.parquet'
    sample_df.to_parquet(path)
    return path


@pytest.fixture
def sample_csv(temp_dir, sample_df) -> Path:
    """Create a sample CSV file."""
    path = temp_dir / 'sample.csv'
    sample_df.to_csv(path, index=False)
    return path


# ============================================================
# MOCK FIXTURES
# ============================================================

@pytest.fixture
def mock_project_structure(temp_dir):
    """Create a mock project directory structure."""
    # Create directories
    (temp_dir / 'src').mkdir()
    (temp_dir / 'src' / 'stages').mkdir()
    (temp_dir / 'src' / 'utils').mkdir()
    (temp_dir / 'data_raw').mkdir()
    (temp_dir / 'data_work').mkdir()
    (temp_dir / 'data_work' / 'diagnostics').mkdir()
    (temp_dir / 'manuscript_quarto').mkdir()
    (temp_dir / 'figures').mkdir()

    # Create minimal files
    (temp_dir / 'src' / '__init__.py').touch()
    (temp_dir / 'src' / 'stages' / '__init__.py').touch()
    (temp_dir / 'src' / 'utils' / '__init__.py').touch()

    return temp_dir
