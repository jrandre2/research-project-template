#!/usr/bin/env python3
"""
Tests for src/config.py

Tests cover:
- Configuration imports
- Path validation
- validate_config() function
- Directory creation
- Manuscript configuration
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestConfigImports:
    """Tests for configuration module imports."""

    def test_import_paths(self):
        """All path constants can be imported."""
        from config import (
            PROJECT_ROOT,
            DATA_RAW_DIR,
            DATA_WORK_DIR,
            FIGURES_DIR,
            MANUSCRIPT_DIR,
            DIAGNOSTICS_DIR,
            DOC_DIR,
            REVIEWS_DIR,
        )
        assert PROJECT_ROOT is not None
        assert isinstance(PROJECT_ROOT, Path)
        assert isinstance(DATA_RAW_DIR, Path)
        assert isinstance(DATA_WORK_DIR, Path)

    def test_import_qa_settings(self):
        """QA settings can be imported."""
        from config import (
            ENABLE_QA_REPORTS,
            QA_REPORTS_DIR,
            QA_THRESHOLDS,
        )
        assert isinstance(ENABLE_QA_REPORTS, bool)
        assert isinstance(QA_REPORTS_DIR, Path)
        assert isinstance(QA_THRESHOLDS, dict)

    def test_import_pipeline_settings(self):
        """Pipeline settings can be imported."""
        from config import (
            DEFAULT_SPECIFICATION,
            SPECIFICATIONS,
        )
        assert isinstance(DEFAULT_SPECIFICATION, str)
        assert isinstance(SPECIFICATIONS, dict)
        assert DEFAULT_SPECIFICATION in SPECIFICATIONS

    def test_import_methodological_params(self):
        """Methodological parameters can be imported."""
        from config import (
            SIGNIFICANCE_LEVEL,
            CONFIDENCE_LEVEL,
            BOOTSTRAP_ITERATIONS,
            BOOTSTRAP_SEED,
        )
        assert 0 < SIGNIFICANCE_LEVEL < 1
        assert 0 < CONFIDENCE_LEVEL < 1
        assert BOOTSTRAP_ITERATIONS > 0
        assert isinstance(BOOTSTRAP_SEED, int)

    def test_import_file_paths(self):
        """File path constants can be imported."""
        from config import (
            PANEL_FILE,
            ESTIMATES_FILE,
            ROBUSTNESS_FILE,
            PANEL_PATH,
            ESTIMATES_PATH,
            ROBUSTNESS_PATH,
        )
        assert isinstance(PANEL_FILE, str)
        assert isinstance(PANEL_PATH, Path)
        assert PANEL_PATH.name == PANEL_FILE


class TestPathValidation:
    """Tests for path configuration validation."""

    def test_project_root_exists(self):
        """PROJECT_ROOT should exist."""
        from config import PROJECT_ROOT
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_project_root_has_expected_structure(self):
        """PROJECT_ROOT should contain expected directories."""
        from config import PROJECT_ROOT
        assert (PROJECT_ROOT / 'src').exists()
        # manuscript_quarto is used by _find_project_root
        assert (PROJECT_ROOT / 'manuscript_quarto').exists()

    def test_data_dirs_under_project_root(self):
        """Data directories should be under PROJECT_ROOT."""
        from config import PROJECT_ROOT, DATA_RAW_DIR, DATA_WORK_DIR
        assert DATA_RAW_DIR.is_relative_to(PROJECT_ROOT)
        assert DATA_WORK_DIR.is_relative_to(PROJECT_ROOT)

    def test_figures_dir_under_project_root(self):
        """Figures directory should be under PROJECT_ROOT."""
        from config import PROJECT_ROOT, FIGURES_DIR
        assert FIGURES_DIR.is_relative_to(PROJECT_ROOT)


class TestValidateConfig:
    """Tests for validate_config() function."""

    def test_validate_config_passes(self):
        """validate_config() should pass with current settings."""
        from config import validate_config
        # Should not raise
        result = validate_config()
        assert result is True

    def test_significance_level_valid(self):
        """SIGNIFICANCE_LEVEL must be between 0 and 1."""
        from config import SIGNIFICANCE_LEVEL
        assert 0 < SIGNIFICANCE_LEVEL < 1

    def test_bootstrap_iterations_positive(self):
        """BOOTSTRAP_ITERATIONS must be positive."""
        from config import BOOTSTRAP_ITERATIONS
        assert BOOTSTRAP_ITERATIONS >= 1


class TestEnsureDirectories:
    """Tests for ensure_directories() function."""

    def test_ensure_directories_creates_missing(self, temp_dir):
        """ensure_directories() creates missing directories."""
        from config import ensure_directories
        # This uses the real paths, so we just verify it doesn't error
        # In a real test we'd mock the paths
        ensure_directories()


class TestManuscriptConfig:
    """Tests for manuscript configuration."""

    def test_manuscripts_dict_exists(self):
        """MANUSCRIPTS dictionary should exist."""
        from config import MANUSCRIPTS, DEFAULT_MANUSCRIPT
        assert isinstance(MANUSCRIPTS, dict)
        assert DEFAULT_MANUSCRIPT in MANUSCRIPTS

    def test_get_manuscript_config_default(self):
        """get_manuscript_config() returns default config."""
        from config import get_manuscript_config, DEFAULT_MANUSCRIPT
        config = get_manuscript_config()
        assert config is not None
        assert 'name' in config
        assert 'dir' in config

    def test_get_manuscript_config_specific(self):
        """get_manuscript_config() returns specific manuscript config."""
        from config import get_manuscript_config
        config = get_manuscript_config('main')
        assert config is not None
        assert 'name' in config

    def test_get_manuscript_config_unknown_raises(self):
        """get_manuscript_config() raises for unknown manuscript."""
        from config import get_manuscript_config
        with pytest.raises(ValueError) as exc_info:
            get_manuscript_config('nonexistent_manuscript')
        assert 'Unknown manuscript' in str(exc_info.value)


class TestQAThresholds:
    """Tests for QA threshold configuration."""

    def test_qa_thresholds_has_expected_keys(self):
        """QA_THRESHOLDS has expected threshold keys."""
        from config import QA_THRESHOLDS
        assert 'max_missing_pct' in QA_THRESHOLDS
        assert 'min_row_count' in QA_THRESHOLDS
        assert 'max_duplicate_pct' in QA_THRESHOLDS

    def test_qa_thresholds_values_are_numeric(self):
        """QA threshold values should be numeric."""
        from config import QA_THRESHOLDS
        for key, value in QA_THRESHOLDS.items():
            assert isinstance(value, (int, float)), f"{key} should be numeric"


class TestCompatibilityWrappers:
    """Tests for backward compatibility functions."""

    def test_get_project_root(self):
        """get_project_root() returns PROJECT_ROOT."""
        from config import get_project_root, PROJECT_ROOT
        assert get_project_root() == PROJECT_ROOT

    def test_get_data_dir_work(self):
        """get_data_dir('work') returns DATA_WORK_DIR."""
        from config import get_data_dir, DATA_WORK_DIR
        assert get_data_dir('work') == DATA_WORK_DIR

    def test_get_data_dir_raw(self):
        """get_data_dir('raw') returns DATA_RAW_DIR."""
        from config import get_data_dir, DATA_RAW_DIR
        assert get_data_dir('raw') == DATA_RAW_DIR

    def test_get_data_dir_diagnostics(self):
        """get_data_dir('diagnostics') returns DIAGNOSTICS_DIR."""
        from config import get_data_dir, DIAGNOSTICS_DIR
        assert get_data_dir('diagnostics') == DIAGNOSTICS_DIR
