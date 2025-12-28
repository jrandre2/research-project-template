#!/usr/bin/env python3
"""
Tests for src/stages/s03_estimation.py

Tests cover:
- EstimationResult dataclass
- Specification registry
- OLS estimation
- Results formatting
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from stages.s03_estimation import (
    EstimationResult,
    SPECIFICATIONS,
    run_ols,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def panel_with_treatment():
    """Create panel data with treatment effect."""
    np.random.seed(42)
    data = []
    for unit in range(1, 51):  # 50 units
        for period in range(1, 13):  # 12 periods
            treatment = 1 if period >= 7 and unit <= 25 else 0
            # True effect of 0.5 for treated
            outcome = np.random.randn() + treatment * 0.5
            data.append({
                'id': unit,
                'period': period,
                'outcome': outcome,
                'treatment': treatment,
                'unit_fe': unit - 1,
                'time_fe': period - 1,
                'covariate_1': np.random.randn(),
            })
    return pd.DataFrame(data)


# ============================================================
# ESTIMATION RESULT TESTS
# ============================================================

class TestEstimationResult:
    """Tests for the EstimationResult dataclass."""

    def test_creates_result(self):
        """Test creating estimation result."""
        result = EstimationResult(
            specification='baseline',
            coefficient=0.5,
            std_error=0.1,
            t_stat=5.0,
            p_value=0.001,
            ci_lower=0.3,
            ci_upper=0.7,
            n_obs=1000,
            n_units=100,
            n_periods=10,
            r_squared=0.25
        )

        assert result.coefficient == 0.5
        assert result.p_value == 0.001

    def test_significant_05(self):
        """Test 5% significance detection."""
        result = EstimationResult(
            specification='test',
            coefficient=0.5,
            std_error=0.1,
            t_stat=5.0,
            p_value=0.03,
            ci_lower=0.3,
            ci_upper=0.7,
            n_obs=100,
            n_units=10,
            n_periods=10,
            r_squared=0.2
        )

        assert result.significant_05 is True
        assert result.significant_01 is False

    def test_significant_01(self):
        """Test 1% significance detection."""
        result = EstimationResult(
            specification='test',
            coefficient=0.5,
            std_error=0.1,
            t_stat=5.0,
            p_value=0.005,
            ci_lower=0.3,
            ci_upper=0.7,
            n_obs=100,
            n_units=10,
            n_periods=10,
            r_squared=0.2
        )

        assert result.significant_01 is True
        assert result.significant_05 is True

    def test_not_significant(self):
        """Test non-significant result."""
        result = EstimationResult(
            specification='test',
            coefficient=0.1,
            std_error=0.2,
            t_stat=0.5,
            p_value=0.6,
            ci_lower=-0.3,
            ci_upper=0.5,
            n_obs=100,
            n_units=10,
            n_periods=10,
            r_squared=0.01
        )

        assert result.significant_05 is False
        assert result.significant_01 is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = EstimationResult(
            specification='baseline',
            coefficient=0.5,
            std_error=0.1,
            t_stat=5.0,
            p_value=0.001,
            ci_lower=0.3,
            ci_upper=0.7,
            n_obs=1000,
            n_units=100,
            n_periods=10,
            r_squared=0.25,
            controls=['x1', 'x2'],
            fe=['unit', 'time'],
            cluster='id'
        )

        d = result.to_dict()

        assert d['specification'] == 'baseline'
        assert d['coefficient'] == 0.5
        assert d['controls'] == 'x1,x2'
        assert d['fe'] == 'unit,time'
        assert d['cluster'] == 'id'
        assert d['significant_05'] is True

    def test_format_coefficient(self):
        """Test coefficient formatting with stars."""
        result = EstimationResult(
            specification='test',
            coefficient=0.5,
            std_error=0.1,
            t_stat=5.0,
            p_value=0.001,
            ci_lower=0.3,
            ci_upper=0.7,
            n_obs=100,
            n_units=10,
            n_periods=10,
            r_squared=0.2
        )

        formatted = result.format_coefficient()

        assert '0.500' in formatted
        assert '**' in formatted  # Significant at 1%
        assert '(0.100)' in formatted


# ============================================================
# SPECIFICATION REGISTRY TESTS
# ============================================================

class TestSpecificationRegistry:
    """Tests for the specification registry."""

    def test_baseline_exists(self):
        """Test baseline specification exists."""
        assert 'baseline' in SPECIFICATIONS

    def test_specifications_have_required_keys(self):
        """Test all specifications have required keys."""
        required_keys = ['name', 'formula', 'controls', 'fe', 'cluster', 'description']

        for name, spec in SPECIFICATIONS.items():
            for key in required_keys:
                assert key in spec, f"Specification '{name}' missing key '{key}'"

    def test_multiple_specifications_exist(self):
        """Test multiple specifications are registered."""
        assert len(SPECIFICATIONS) >= 2


# ============================================================
# OLS TESTS
# ============================================================

class TestRunOLS:
    """Tests for the run_ols function."""

    def test_runs_ols(self, panel_with_treatment):
        """Test running OLS regression."""
        result = run_ols(
            panel_with_treatment,
            y_var='outcome',
            x_vars=['treatment']
        )

        assert 'coefficient' in result
        assert 'std_error' in result
        assert 'p_value' in result
        assert 'n_obs' in result

    def test_coefficient_direction(self, panel_with_treatment):
        """Test that coefficient has expected direction."""
        result = run_ols(
            panel_with_treatment,
            y_var='outcome',
            x_vars=['treatment']
        )

        # Treatment effect is positive
        assert result['coefficient'] > 0

    def test_handles_controls(self, panel_with_treatment):
        """Test OLS with control variables."""
        result = run_ols(
            panel_with_treatment,
            y_var='outcome',
            x_vars=['treatment', 'covariate_1']
        )

        assert result['n_obs'] > 0


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestEstimationIntegration:
    """Integration tests for estimation."""

    def test_estimate_baseline(self, panel_with_treatment):
        """Test running baseline estimation."""
        spec = SPECIFICATIONS['baseline']

        # Run simplified estimation
        result = run_ols(
            panel_with_treatment,
            y_var='outcome',
            x_vars=['treatment']
        )

        # Create EstimationResult
        est_result = EstimationResult(
            specification='baseline',
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['coefficient'] / result['std_error'],
            p_value=result['p_value'],
            ci_lower=result['coefficient'] - 1.96 * result['std_error'],
            ci_upper=result['coefficient'] + 1.96 * result['std_error'],
            n_obs=result['n_obs'],
            n_units=50,
            n_periods=12,
            r_squared=result.get('r_squared', 0),
            controls=spec['controls'],
            fe=spec['fe'],
            cluster=spec['cluster']
        )

        assert est_result.n_obs == result['n_obs']
        assert est_result.coefficient > 0  # Treatment effect
