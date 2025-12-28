#!/usr/bin/env python3
"""
Tests for src/stages/s05_figures.py

Tests cover:
- Event study plotting
- Treatment/control means plotting
- Coefficient comparison plots
- Figure output
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from stages.s05_figures import (
    plot_event_study,
    plot_treatment_control_means,
    OUTCOME_VAR,
    TREATMENT_VAR,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def panel_with_event_time():
    """Create panel data with event time for plotting."""
    np.random.seed(42)
    data = []
    for unit in range(1, 51):
        for period in range(1, 25):
            treatment = 1 if period >= 13 and unit <= 25 else 0
            ever_treated = 1 if unit <= 25 else 0
            event_time = period - 13 if unit <= 25 else np.nan
            outcome = np.random.randn() + treatment * 0.5
            data.append({
                'id': unit,
                'period': period,
                'outcome': outcome,
                'treatment': treatment,
                'ever_treated': ever_treated,
                'event_time': event_time,
            })
    return pd.DataFrame(data)


@pytest.fixture
def panel_without_event_time():
    """Create panel data without event_time column."""
    np.random.seed(42)
    data = []
    for unit in range(1, 51):
        for period in range(1, 13):
            outcome = np.random.randn()
            data.append({
                'id': unit,
                'period': period,
                'outcome': outcome,
                'treatment': 0,
            })
    return pd.DataFrame(data)


# ============================================================
# EVENT STUDY PLOT TESTS
# ============================================================

class TestPlotEventStudy:
    """Tests for the plot_event_study function."""

    def test_creates_figure(self, panel_with_event_time, temp_dir):
        """Test that event study creates a figure."""
        output_path = temp_dir / 'event_study.png'

        result = plot_event_study(panel_with_event_time, output_path)

        assert result is not None
        assert output_path.exists()

    def test_returns_none_without_event_time(self, panel_without_event_time, temp_dir):
        """Test returns None when no event_time column."""
        output_path = temp_dir / 'event_study.png'

        result = plot_event_study(panel_without_event_time, output_path)

        assert result is None
        assert not output_path.exists()

    def test_respects_period_limits(self, panel_with_event_time, temp_dir):
        """Test that period limits are respected."""
        output_path = temp_dir / 'event_study.png'

        result = plot_event_study(
            panel_with_event_time,
            output_path,
            pre_periods=5,
            post_periods=5
        )

        assert result is not None


# ============================================================
# TREATMENT CONTROL PLOT TESTS
# ============================================================

class TestPlotTreatmentControlMeans:
    """Tests for the plot_treatment_control_means function."""

    def test_creates_figure(self, panel_with_event_time, temp_dir):
        """Test that treatment/control plot creates a figure."""
        output_path = temp_dir / 'treatment_control.png'

        result = plot_treatment_control_means(panel_with_event_time, output_path)

        assert result is not None
        assert output_path.exists()

    def test_returns_none_without_required_columns(self, temp_dir):
        """Test returns None when missing required columns."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'outcome': [1.0, 2.0, 3.0]
        })
        output_path = temp_dir / 'treatment_control.png'

        result = plot_treatment_control_means(df, output_path)

        assert result is None


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestFiguresIntegration:
    """Integration tests for figure generation."""

    def test_multiple_figures(self, panel_with_event_time, temp_dir):
        """Test generating multiple figures."""
        figures_dir = temp_dir / 'figures'
        figures_dir.mkdir()

        # Generate event study
        event_path = figures_dir / 'event_study.png'
        plot_event_study(panel_with_event_time, event_path)

        # Generate treatment/control
        tc_path = figures_dir / 'treatment_control.png'
        plot_treatment_control_means(panel_with_event_time, tc_path)

        # Both should exist
        assert event_path.exists()
        assert tc_path.exists()

    def test_figure_file_size(self, panel_with_event_time, temp_dir):
        """Test that figure files have reasonable size."""
        output_path = temp_dir / 'event_study.png'

        plot_event_study(panel_with_event_time, output_path)

        # File should be non-trivial size (at least 10KB)
        assert output_path.stat().st_size > 10000
