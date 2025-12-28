#!/usr/bin/env python3
"""
Tests for src/stages/s04_robustness.py

Tests cover:
- RobustnessResult dataclass
- Placebo tests
- Sample restriction tests
- Simple OLS function
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from stages.s04_robustness import (
    RobustnessResult,
    run_simple_ols,
    run_placebo_time,
    run_sample_restrictions,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def panel_with_treatment():
    """Create panel data for robustness tests."""
    np.random.seed(42)
    data = []
    for unit in range(1, 51):  # 50 units
        for period in range(1, 25):  # 24 periods
            treatment = 1 if period >= 13 and unit <= 25 else 0
            ever_treated = 1 if unit <= 25 else 0
            outcome = np.random.randn() + treatment * 0.5
            data.append({
                'id': unit,
                'period': period,
                'outcome': outcome,
                'treatment': treatment,
                'ever_treated': ever_treated,
            })
    return pd.DataFrame(data)


# ============================================================
# ROBUSTNESS RESULT TESTS
# ============================================================

class TestRobustnessResult:
    """Tests for the RobustnessResult dataclass."""

    def test_creates_result(self):
        """Test creating robustness result."""
        result = RobustnessResult(
            test_name='placebo_t10',
            test_type='placebo',
            coefficient=0.1,
            std_error=0.15,
            p_value=0.5,
            n_obs=500,
            description='Placebo test at period 10'
        )

        assert result.test_name == 'placebo_t10'
        assert result.test_type == 'placebo'
        assert result.coefficient == 0.1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = RobustnessResult(
            test_name='exclude_2020',
            test_type='sample',
            coefficient=0.45,
            std_error=0.12,
            p_value=0.001,
            n_obs=800,
            description='Excluding year 2020'
        )

        d = result.to_dict()

        assert d['test_name'] == 'exclude_2020'
        assert d['test_type'] == 'sample'
        assert d['significant'] is True
        assert d['n_obs'] == 800

    def test_significance_flag(self):
        """Test significance flag in to_dict."""
        sig_result = RobustnessResult(
            test_name='test1', test_type='spec',
            coefficient=0.5, std_error=0.1, p_value=0.01,
            n_obs=100, description='Significant'
        )
        not_sig_result = RobustnessResult(
            test_name='test2', test_type='spec',
            coefficient=0.1, std_error=0.1, p_value=0.6,
            n_obs=100, description='Not significant'
        )

        assert sig_result.to_dict()['significant'] is True
        assert not_sig_result.to_dict()['significant'] is False


# ============================================================
# SIMPLE OLS TESTS
# ============================================================

class TestRunSimpleOLS:
    """Tests for the run_simple_ols function."""

    def test_runs_ols(self, panel_with_treatment):
        """Test running simple OLS."""
        result = run_simple_ols(
            panel_with_treatment,
            y_var='outcome',
            x_var='treatment'
        )

        assert 'coefficient' in result
        assert 'std_error' in result
        assert 'p_value' in result
        assert 'n_obs' in result

    def test_coefficient_reasonable(self, panel_with_treatment):
        """Test that coefficient is reasonable."""
        result = run_simple_ols(
            panel_with_treatment,
            y_var='outcome',
            x_var='treatment'
        )

        # Coefficient should be positive (treatment effect)
        assert result['coefficient'] > 0
        # Coefficient should be less than 2 (reasonable magnitude)
        assert result['coefficient'] < 2

    def test_raises_for_insufficient_data(self):
        """Test that insufficient data raises error."""
        small_df = pd.DataFrame({
            'outcome': [1, 2],
            'treatment': [0, 1]
        })

        with pytest.raises(ValueError, match="Insufficient observations"):
            run_simple_ols(small_df, 'outcome', 'treatment')

    def test_handles_missing_values(self, panel_with_treatment):
        """Test handling of missing values."""
        df = panel_with_treatment.copy()
        df.loc[:10, 'outcome'] = np.nan

        result = run_simple_ols(df, 'outcome', 'treatment')

        # Should work with fewer observations
        assert result['n_obs'] < len(df)


# ============================================================
# PLACEBO TESTS
# ============================================================

class TestRunPlaceboTime:
    """Tests for the run_placebo_time function."""

    def test_runs_placebos(self, panel_with_treatment):
        """Test running placebo tests."""
        results = run_placebo_time(
            panel_with_treatment,
            n_placebos=3,
            pre_period_end=12
        )

        assert len(results) > 0
        assert all(isinstance(r, RobustnessResult) for r in results)

    def test_placebo_type_correct(self, panel_with_treatment):
        """Test that placebo results have correct type."""
        results = run_placebo_time(
            panel_with_treatment,
            n_placebos=2,
            pre_period_end=12
        )

        for r in results:
            assert r.test_type == 'placebo'
            assert 'placebo' in r.test_name.lower()

    def test_returns_empty_for_no_period_col(self):
        """Test returns empty for data without period column."""
        df = pd.DataFrame({
            'id': [1, 2],
            'outcome': [1.0, 2.0],
            'treatment': [0, 1]
        })

        results = run_placebo_time(df)

        assert results == []


# ============================================================
# SAMPLE RESTRICTION TESTS
# ============================================================

class TestRunSampleRestrictions:
    """Tests for the run_sample_restrictions function."""

    def test_runs_restrictions(self, panel_with_treatment):
        """Test running sample restriction tests."""
        results = run_sample_restrictions(panel_with_treatment)

        # Should return a list of results
        assert isinstance(results, list)

    def test_results_are_robustness_results(self, panel_with_treatment):
        """Test that results are RobustnessResult objects."""
        results = run_sample_restrictions(panel_with_treatment)

        for r in results:
            assert isinstance(r, RobustnessResult)
            assert r.test_type in ['sample', 'specification']


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestRobustnessIntegration:
    """Integration tests for robustness checks."""

    def test_full_robustness_workflow(self, panel_with_treatment):
        """Test complete robustness check workflow."""
        all_results = []

        # Run placebo tests
        placebo_results = run_placebo_time(
            panel_with_treatment,
            n_placebos=3,
            pre_period_end=12
        )
        all_results.extend(placebo_results)

        # Run sample restrictions
        sample_results = run_sample_restrictions(panel_with_treatment)
        all_results.extend(sample_results)

        # Convert to DataFrame
        if all_results:
            results_df = pd.DataFrame([r.to_dict() for r in all_results])

            assert 'test_name' in results_df.columns
            assert 'coefficient' in results_df.columns
            assert 'significant' in results_df.columns
