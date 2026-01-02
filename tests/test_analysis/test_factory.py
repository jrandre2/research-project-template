"""Tests for analysis.factory module."""
from __future__ import annotations

import pytest


class TestEngineRegistry:
    """Tests for engine registration and retrieval."""

    def test_list_engines(self):
        """Test listing registered engines."""
        from analysis.factory import list_engines

        engines = list_engines()

        assert isinstance(engines, dict)
        assert 'python' in engines
        # Python should always be available
        assert engines['python'] is True

    def test_get_python_engine(self):
        """Test getting Python engine."""
        from analysis.factory import get_engine
        from analysis.base import AnalysisEngine

        engine = get_engine('python')

        assert engine is not None
        assert engine.name == 'python'
        assert isinstance(engine, AnalysisEngine)

    def test_get_unknown_engine_raises(self):
        """Test that unknown engine raises ValueError."""
        from analysis.factory import get_engine

        with pytest.raises(ValueError, match="Unknown engine"):
            get_engine('unknown_engine')

    def test_get_engine_info(self):
        """Test getting engine information."""
        from analysis.factory import get_engine_info

        info = get_engine_info('python')

        assert info['name'] == 'python'
        assert info['available'] is True
        assert 'version' in info
        assert 'message' in info

    def test_get_engine_info_unknown(self):
        """Test getting info for unknown engine."""
        from analysis.factory import get_engine_info

        info = get_engine_info('unknown')

        assert info['available'] is False
        assert 'Unknown engine' in info['message']


class TestEngineValidation:
    """Tests for engine installation validation."""

    def test_python_engine_validation(self):
        """Test Python engine validation."""
        from analysis.factory import get_engine

        engine = get_engine('python')
        available, message = engine.validate_installation()

        assert available is True
        assert 'numpy' in message.lower()

    def test_r_engine_validation_message(self):
        """Test R engine validation provides useful message."""
        from analysis.factory import list_engines, get_engine

        engines = list_engines()

        if engines.get('r'):
            engine = get_engine('r')
            available, message = engine.validate_installation()
            # R is available
            assert available is True
            assert 'ready' in message.lower() or 'fixest' in message.lower()
        else:
            # R not available - that's OK
            pass
