"""Tests for analysis.base module."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestEstimationResult:
    """Tests for EstimationResult dataclass."""

    def test_basic_creation(self):
        """Test creating a basic EstimationResult."""
        from analysis.base import EstimationResult

        result = EstimationResult(
            specification='baseline',
            n_obs=1000,
            n_units=100,
            n_periods=10,
            coefficients={'treatment': 0.5},
            std_errors={'treatment': 0.1},
            p_values={'treatment': 0.001},
            r_squared=0.8,
            engine='python',
            engine_version='numpy 2.0',
            execution_time_seconds=0.5,
        )

        assert result.specification == 'baseline'
        assert result.n_obs == 1000
        assert result.coefficients['treatment'] == 0.5
        assert result.engine == 'python'

    def test_get_coefficient(self):
        """Test coefficient accessor."""
        from analysis.base import EstimationResult

        result = EstimationResult(
            specification='test',
            n_obs=100,
            coefficients={'treatment': 0.5, 'control': 0.2},
        )

        assert result.get_coefficient('treatment') == 0.5
        assert result.get_coefficient('control') == 0.2
        assert result.get_coefficient('missing') is None

    def test_is_significant(self):
        """Test significance checking."""
        from analysis.base import EstimationResult

        result = EstimationResult(
            specification='test',
            n_obs=100,
            p_values={'significant': 0.01, 'marginal': 0.08, 'not_sig': 0.5},
        )

        assert result.is_significant('significant', level=0.05)
        assert not result.is_significant('not_sig', level=0.05)
        assert result.is_significant('marginal', level=0.10)

    def test_format_coefficient(self):
        """Test coefficient formatting with stars."""
        from analysis.base import EstimationResult

        result = EstimationResult(
            specification='test',
            n_obs=100,
            coefficients={'var': 0.123456},
            std_errors={'var': 0.05},
            p_values={'var': 0.001},
        )

        formatted = result.format_coefficient('var', decimals=3)
        assert '0.123' in formatted
        assert '***' in formatted  # p < 0.01

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from analysis.base import EstimationResult

        result = EstimationResult(
            specification='test',
            n_obs=100,
            coefficients={'treatment': 0.5},
            engine='python',
        )

        d = result.to_dict()
        assert isinstance(d, dict)
        assert d['specification'] == 'test'
        assert d['n_obs'] == 100
        assert d['coefficients'] == {'treatment': 0.5}

    def test_json_serialization(self):
        """Test JSON serialization round-trip."""
        from analysis.base import EstimationResult

        result = EstimationResult(
            specification='test',
            n_obs=100,
            coefficients={'treatment': 0.5},
            std_errors={'treatment': 0.1},
            engine='python',
            engine_version='test',
            execution_time_seconds=1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'result.json'
            result.to_json(path)

            loaded = EstimationResult.from_json(path)

            assert loaded.specification == result.specification
            assert loaded.n_obs == result.n_obs
            assert loaded.coefficients == result.coefficients


class TestAnalysisEngineProtocol:
    """Tests for AnalysisEngine protocol."""

    def test_protocol_isinstance(self):
        """Test that engines satisfy the Protocol."""
        from analysis.base import AnalysisEngine
        from analysis.engines.python_engine import PythonEngine

        engine = PythonEngine()

        # Check that PythonEngine implements the protocol
        assert isinstance(engine, AnalysisEngine)

    def test_engine_has_required_methods(self):
        """Test that engines have required methods."""
        from analysis.engines.python_engine import PythonEngine

        engine = PythonEngine()

        # Check required properties
        assert hasattr(engine, 'name')
        assert hasattr(engine, 'version')

        # Check required methods
        assert callable(getattr(engine, 'validate_installation', None))
        assert callable(getattr(engine, 'estimate', None))
        assert callable(getattr(engine, 'estimate_batch', None))
