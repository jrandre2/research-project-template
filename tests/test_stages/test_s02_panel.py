#!/usr/bin/env python3
"""
Tests for src/stages/s02_panel.py

Tests cover:
- PanelDiagnostics dataclass
- Panel diagnostics calculation
- Panel balancing
- Fixed effects creation
- Event time construction
- Treatment indicators
- Panel validation
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from stages.s02_panel import (
    PanelDiagnostics,
    calculate_panel_diagnostics,
    balance_panel,
    create_fixed_effects,
    create_event_time,
    create_treatment_indicators,
    validate_panel,
    UNIT_ID_COL,
    TIME_COL,
    TREATMENT_COL,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def balanced_panel():
    """Create a balanced panel DataFrame."""
    data = []
    for unit in range(1, 11):  # 10 units
        for period in range(1, 7):  # 6 periods
            data.append({
                'id': unit,
                'period': period,
                'outcome': np.random.randn(),
                'treatment': 1 if period >= 4 and unit <= 5 else 0
            })
    return pd.DataFrame(data)


@pytest.fixture
def unbalanced_panel():
    """Create an unbalanced panel DataFrame."""
    data = []
    for unit in range(1, 11):
        # Some units have fewer periods
        n_periods = 6 if unit <= 7 else 4
        for period in range(1, n_periods + 1):
            data.append({
                'id': unit,
                'period': period,
                'outcome': np.random.randn(),
                'treatment': 1 if period >= 4 and unit <= 5 else 0
            })
    return pd.DataFrame(data)


# ============================================================
# PANEL DIAGNOSTICS TESTS
# ============================================================

class TestPanelDiagnostics:
    """Tests for the PanelDiagnostics dataclass."""

    def test_creates_diagnostics(self):
        """Test creating panel diagnostics."""
        diag = PanelDiagnostics(
            n_units=100,
            n_periods=12,
            n_observations=1200,
            is_balanced=True,
            balance_rate=1.0,
            treatment_share=0.5,
            n_treated_units=50,
            n_control_units=50
        )

        assert diag.n_units == 100
        assert diag.n_periods == 12
        assert diag.is_balanced is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        diag = PanelDiagnostics(
            n_units=100,
            n_periods=12,
            n_observations=1200,
            is_balanced=True,
            balance_rate=1.0,
            treatment_share=0.5,
            n_treated_units=50,
            n_control_units=50
        )

        d = diag.to_dict()

        assert d['n_units'] == 100
        assert d['n_periods'] == 12
        assert d['expected_if_balanced'] == 1200
        assert d['is_balanced'] is True

    def test_format(self):
        """Test string formatting."""
        diag = PanelDiagnostics(
            n_units=100,
            n_periods=12,
            n_observations=1200,
            is_balanced=True,
            balance_rate=1.0,
            treatment_share=0.5,
            n_treated_units=50,
            n_control_units=50
        )

        formatted = diag.format()

        assert 'PANEL DIAGNOSTICS' in formatted
        assert '100' in formatted
        assert '12' in formatted


class TestCalculatePanelDiagnostics:
    """Tests for the calculate_panel_diagnostics function."""

    def test_calculates_for_balanced_panel(self, balanced_panel):
        """Test diagnostics for balanced panel."""
        diag = calculate_panel_diagnostics(balanced_panel)

        assert diag.n_units == 10
        assert diag.n_periods == 6
        assert diag.n_observations == 60
        assert diag.is_balanced is True
        assert diag.balance_rate == 1.0

    def test_calculates_for_unbalanced_panel(self, unbalanced_panel):
        """Test diagnostics for unbalanced panel."""
        diag = calculate_panel_diagnostics(unbalanced_panel)

        assert diag.n_units == 10
        assert diag.n_periods == 6
        assert diag.is_balanced is False
        assert diag.balance_rate < 1.0

    def test_calculates_treatment_share(self, balanced_panel):
        """Test treatment share calculation."""
        diag = calculate_panel_diagnostics(balanced_panel)

        # 5 treated units * 3 treated periods = 15 treated observations
        # 60 total observations
        assert diag.treatment_share == 15 / 60
        assert diag.n_treated_units == 5
        assert diag.n_control_units == 5

    def test_handles_no_treatment_column(self):
        """Test handling when no treatment column exists."""
        df = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'period': [1, 2, 1, 2],
            'value': [10, 20, 30, 40]
        })

        diag = calculate_panel_diagnostics(df)

        assert diag.treatment_share == 0.0
        assert diag.n_treated_units == 0


# ============================================================
# BALANCE PANEL TESTS
# ============================================================

class TestBalancePanel:
    """Tests for the balance_panel function."""

    def test_balance_with_drop(self, unbalanced_panel):
        """Test balancing by dropping incomplete units."""
        balanced = balance_panel(unbalanced_panel, fill_method='drop')

        # Only units with all 6 periods should remain
        assert balanced['id'].nunique() == 7  # units 1-7 have all periods

    def test_balance_with_zero(self, unbalanced_panel):
        """Test balancing by filling with zeros."""
        balanced = balance_panel(unbalanced_panel, fill_method='zero')

        # All unit-period combinations should exist
        assert len(balanced) == 10 * 6  # 10 units * 6 periods

    def test_already_balanced(self, balanced_panel):
        """Test that already balanced panel is unchanged."""
        result = balance_panel(balanced_panel, fill_method='drop')

        assert len(result) == len(balanced_panel)

    def test_returns_dataframe(self, unbalanced_panel):
        """Test that result is a DataFrame."""
        result = balance_panel(unbalanced_panel)

        assert isinstance(result, pd.DataFrame)


# ============================================================
# FIXED EFFECTS TESTS
# ============================================================

class TestCreateFixedEffects:
    """Tests for the create_fixed_effects function."""

    def test_creates_unit_fe(self, balanced_panel):
        """Test creation of unit fixed effects."""
        result = create_fixed_effects(balanced_panel)

        assert 'unit_fe' in result.columns
        assert result['unit_fe'].nunique() == 10

    def test_creates_time_fe(self, balanced_panel):
        """Test creation of time fixed effects."""
        result = create_fixed_effects(balanced_panel)

        assert 'time_fe' in result.columns
        assert result['time_fe'].nunique() == 6

    def test_fe_codes_are_integers(self, balanced_panel):
        """Test that FE codes are integer type."""
        result = create_fixed_effects(balanced_panel)

        assert result['unit_fe'].dtype in ['int8', 'int16', 'int32', 'int64']
        assert result['time_fe'].dtype in ['int8', 'int16', 'int32', 'int64']

    def test_preserves_original_columns(self, balanced_panel):
        """Test that original columns are preserved."""
        original_cols = set(balanced_panel.columns)
        result = create_fixed_effects(balanced_panel)

        assert original_cols.issubset(set(result.columns))


# ============================================================
# EVENT TIME TESTS
# ============================================================

class TestCreateEventTime:
    """Tests for the create_event_time function."""

    def test_creates_event_time_column(self, balanced_panel):
        """Test creation of event_time column."""
        result = create_event_time(balanced_panel)

        assert 'event_time' in result.columns

    def test_event_time_values(self, balanced_panel):
        """Test event time values are correct."""
        result = create_event_time(balanced_panel)

        # For treated units (1-5), treatment starts at period 4
        treated = result[result['id'] <= 5]

        # Event time should be period - 4 for treated units
        for _, row in treated.iterrows():
            expected = row['period'] - 4
            assert row['event_time'] == expected

    def test_never_treated_have_nan(self, balanced_panel):
        """Test that never-treated units have NaN event_time."""
        result = create_event_time(balanced_panel)

        # Units 6-10 are never treated
        never_treated = result[result['id'] > 5]
        assert never_treated['event_time'].isna().all()

    def test_creates_treatment_period(self, balanced_panel):
        """Test creation of treatment_period column."""
        result = create_event_time(balanced_panel)

        assert 'treatment_period' in result.columns


# ============================================================
# TREATMENT INDICATORS TESTS
# ============================================================

class TestCreateTreatmentIndicators:
    """Tests for the create_treatment_indicators function."""

    def test_creates_ever_treated(self, balanced_panel):
        """Test creation of ever_treated indicator."""
        result = create_treatment_indicators(balanced_panel)

        assert 'ever_treated' in result.columns

        # Units 1-5 are ever treated
        for unit in range(1, 6):
            unit_data = result[result['id'] == unit]
            assert (unit_data['ever_treated'] == 1).all()

        # Units 6-10 are never treated
        for unit in range(6, 11):
            unit_data = result[result['id'] == unit]
            assert (unit_data['ever_treated'] == 0).all()

    def test_creates_post_treatment(self, balanced_panel):
        """Test creation of post_treatment indicator."""
        # First add treatment_period
        df = create_event_time(balanced_panel)
        result = create_treatment_indicators(df)

        assert 'post_treatment' in result.columns


# ============================================================
# VALIDATION TESTS
# ============================================================

class TestValidatePanel:
    """Tests for the validate_panel function."""

    def test_valid_panel_passes(self, balanced_panel):
        """Test that valid panel passes validation."""
        result = validate_panel(balanced_panel)

        assert result is True

    def test_empty_panel_fails(self):
        """Test that empty panel fails validation."""
        df = pd.DataFrame({'id': [], 'period': []})

        result = validate_panel(df)

        assert result is False

    def test_missing_unit_id_fails(self):
        """Test that missing unit IDs fail validation."""
        df = pd.DataFrame({
            'id': [1, None, 3],
            'period': [1, 1, 1]
        })

        result = validate_panel(df)

        assert result is False

    def test_duplicate_unit_period_fails(self):
        """Test that duplicate unit-period combinations fail."""
        df = pd.DataFrame({
            'id': [1, 1, 2],
            'period': [1, 1, 1],  # Duplicate (1, 1)
            'value': [10, 20, 30]
        })

        result = validate_panel(df)

        assert result is False


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestPanelStageIntegration:
    """Integration tests for the panel construction stage."""

    def test_full_panel_construction(self, balanced_panel):
        """Test complete panel construction workflow."""
        # Create fixed effects
        df = create_fixed_effects(balanced_panel)

        # Create event time
        df = create_event_time(df)

        # Create treatment indicators
        df = create_treatment_indicators(df)

        # Validate
        is_valid = validate_panel(df)

        assert is_valid is True
        assert 'unit_fe' in df.columns
        assert 'time_fe' in df.columns
        assert 'event_time' in df.columns
        assert 'ever_treated' in df.columns

    def test_diagnostics_after_construction(self, balanced_panel):
        """Test diagnostics after panel construction."""
        df = create_fixed_effects(balanced_panel)
        df = create_event_time(df)
        df = create_treatment_indicators(df)

        diag = calculate_panel_diagnostics(df)

        assert diag.n_units == 10
        assert diag.n_periods == 6
        assert diag.is_balanced is True
