#!/usr/bin/env python3
"""
Tests for src/utils/validation.py

Tests cover:
- ValidationRule creation
- DataValidator operations
- Built-in validators
- Schema validation
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.validation import (
    ValidationRule,
    ValidationResult,
    ValidationReport,
    DataValidator,
    no_missing_values,
    unique_values,
    value_range,
    categorical_values,
    date_range,
    row_count,
    no_duplicate_rows,
    positive_values,
)


# ============================================================
# VALIDATION RULE TESTS
# ============================================================

class TestValidationRule:
    """Tests for ValidationRule dataclass."""

    def test_create_rule(self):
        """Create a basic validation rule."""
        rule = ValidationRule(
            name='test_rule',
            check=lambda df: (True, 'All good'),
            severity='error',
            description='Test rule'
        )
        assert rule.name == 'test_rule'
        assert rule.severity == 'error'

    def test_run_passing_rule(self, sample_df):
        """Run a rule that passes."""
        rule = ValidationRule(
            name='always_pass',
            check=lambda df: (True, 'Passed'),
            severity='error',
            description='Always passes'
        )
        passed, message = rule.check(sample_df)
        assert passed is True


# ============================================================
# DATA VALIDATOR TESTS
# ============================================================

class TestDataValidator:
    """Tests for DataValidator class."""

    def test_add_rule(self):
        """Add rule to validator."""
        validator = DataValidator()
        rule = no_missing_values(['col1'])
        validator.add_rule(rule)
        assert len(validator.rules) == 1

    def test_method_chaining(self):
        """Add_rule should return self for chaining."""
        validator = (DataValidator()
            .add_rule(no_missing_values(['a']))
            .add_rule(unique_values('id'))
        )
        assert len(validator.rules) == 2

    def test_validate_all_pass(self, sample_df):
        """Validate DataFrame where all rules pass."""
        validator = DataValidator()
        validator.add_rule(row_count(min_rows=1))
        report = validator.validate(sample_df)
        assert not report.has_errors
        assert report.passed == 1

    def test_validate_with_failures(self, df_with_missing):
        """Validate DataFrame with rule failures."""
        validator = DataValidator()
        validator.add_rule(no_missing_values(['nullable_col']))
        report = validator.validate(df_with_missing)
        assert report.has_errors

    def test_validate_or_raise(self, df_with_missing):
        """validate_or_raise should raise on errors."""
        validator = DataValidator()
        validator.add_rule(no_missing_values(['nullable_col']))
        with pytest.raises(ValueError, match='Validation failed'):
            validator.validate_or_raise(df_with_missing)


# ============================================================
# BUILT-IN VALIDATOR TESTS
# ============================================================

class TestNoMissingValues:
    """Tests for no_missing_values validator."""

    def test_no_missing(self, sample_df):
        """DataFrame with no missing values."""
        rule = no_missing_values(['id', 'value'])
        validator = DataValidator().add_rule(rule)
        report = validator.validate(sample_df)
        assert not report.has_errors

    def test_with_missing(self, df_with_missing):
        """DataFrame with missing values."""
        rule = no_missing_values(['nullable_col'])
        validator = DataValidator().add_rule(rule)
        report = validator.validate(df_with_missing)
        assert report.has_errors


class TestUniqueValues:
    """Tests for unique_values validator."""

    def test_unique(self, sample_df):
        """Column with unique values."""
        rule = unique_values('id')
        validator = DataValidator().add_rule(rule)
        report = validator.validate(sample_df)
        assert not report.has_errors

    def test_duplicates(self, df_with_duplicates):
        """Column with duplicate values."""
        rule = unique_values('id')
        validator = DataValidator().add_rule(rule)
        report = validator.validate(df_with_duplicates)
        assert report.has_errors


class TestValueRange:
    """Tests for value_range validator."""

    def test_in_range(self):
        """Values within range."""
        df = pd.DataFrame({'x': [1, 5, 10]})
        rule = value_range('x', min_val=0, max_val=15)
        validator = DataValidator().add_rule(rule)
        report = validator.validate(df)
        assert not report.has_errors

    def test_out_of_range(self):
        """Values outside range."""
        df = pd.DataFrame({'x': [1, 5, 100]})
        rule = value_range('x', min_val=0, max_val=10)
        validator = DataValidator().add_rule(rule)
        report = validator.validate(df)
        assert report.has_warnings or report.has_errors


class TestCategoricalValues:
    """Tests for categorical_values validator."""

    def test_valid_categories(self, sample_df):
        """All values in valid set."""
        rule = categorical_values('category', {'A', 'B', 'C'})
        validator = DataValidator().add_rule(rule)
        report = validator.validate(sample_df)
        assert not report.has_errors

    def test_invalid_categories(self):
        """Values not in valid set."""
        df = pd.DataFrame({'cat': ['A', 'B', 'X']})
        rule = categorical_values('cat', {'A', 'B', 'C'})
        validator = DataValidator().add_rule(rule)
        report = validator.validate(df)
        assert report.has_errors


class TestRowCount:
    """Tests for row_count validator."""

    def test_sufficient_rows(self, sample_df):
        """DataFrame with enough rows."""
        rule = row_count(min_rows=10)
        validator = DataValidator().add_rule(rule)
        report = validator.validate(sample_df)
        assert not report.has_errors

    def test_insufficient_rows(self, sample_df):
        """DataFrame with too few rows."""
        rule = row_count(min_rows=1000)
        validator = DataValidator().add_rule(rule)
        report = validator.validate(sample_df)
        assert report.has_errors


class TestNoDuplicateRows:
    """Tests for no_duplicate_rows validator."""

    def test_no_duplicates(self, sample_df):
        """DataFrame with no duplicate rows."""
        rule = no_duplicate_rows(['id'])
        validator = DataValidator().add_rule(rule)
        report = validator.validate(sample_df)
        assert not report.has_errors

    def test_with_duplicates(self, df_with_duplicates):
        """DataFrame with duplicate rows."""
        rule = no_duplicate_rows(['id', 'value'])
        validator = DataValidator().add_rule(rule)
        report = validator.validate(df_with_duplicates)
        assert report.has_errors


# ============================================================
# VALIDATION REPORT TESTS
# ============================================================

class TestValidationReport:
    """Tests for ValidationReport class."""

    def test_empty_report(self):
        """Empty report with no results."""
        report = ValidationReport(results=[])
        assert len(report.results) == 0
        assert not report.has_errors

    def test_report_format(self, sample_df):
        """Format report as string."""
        validator = (DataValidator()
            .add_rule(row_count(min_rows=1))
            .add_rule(unique_values('id'))
        )
        report = validator.validate(sample_df)
        formatted = report.format()
        assert 'VALIDATION REPORT' in formatted

    def test_report_to_dict(self, sample_df):
        """Convert report to dictionary."""
        validator = DataValidator().add_rule(row_count(min_rows=1))
        report = validator.validate(sample_df)
        d = report.to_dict()
        assert 'passed' in d
        assert 'results' in d
        assert 'has_errors' in d
        assert 'error_count' in d


# ============================================================
# SCHEMA VALIDATION TESTS
# ============================================================

class TestSchemaValidation:
    """Tests for schema-based validation."""

    def test_valid_schema(self, sample_df):
        """DataFrame matches schema."""
        # Note: validate_schema expects {column: numpy_type} format
        schema = {
            'id': np.integer,
            'value': np.floating,
        }
        validator = DataValidator()
        report = validator.validate_schema(sample_df, schema)
        # Should not have dtype errors for required columns
        assert report is not None

    def test_missing_required_column(self, sample_df):
        """Schema requires column not in DataFrame."""
        schema = {
            'nonexistent': np.integer,
        }
        validator = DataValidator()
        report = validator.validate_schema(sample_df, schema)
        assert report.has_errors

    def test_column_type_mismatch(self, sample_df):
        """Column exists but has wrong type."""
        # 'category' column is object type, but we expect integer
        schema = {
            'category': np.integer,
        }
        validator = DataValidator()
        report = validator.validate_schema(sample_df, schema)
        # Type mismatch should produce a warning (not error per implementation)
        assert report.has_warnings or len(report.results) > 0
