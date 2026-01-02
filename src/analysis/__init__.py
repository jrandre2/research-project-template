"""
Analysis Engine Package for CENTAUR.

Provides a language-agnostic interface for running statistical estimation across
Python, R, Stata, and Julia engines.

Usage
-----
    from analysis import get_engine, list_engines, EstimationResult

    # Get default engine (from config)
    engine = get_engine()

    # Get specific engine
    engine = get_engine('r')

    # List available engines
    engines = list_engines()
    # {'python': True, 'r': True, 'stata': False}

    # Run estimation
    result = engine.estimate(
        data_path=Path('data_work/panel.parquet'),
        specification={'name': 'baseline', ...},
        output_dir=Path('data_work/diagnostics'),
    )
"""
from __future__ import annotations

from .base import AnalysisEngine, BaseAnalysisEngine, EstimationResult
from .factory import get_engine, list_engines, register_engine
from .specifications import load_specifications, get_specification, validate_specification

__all__ = [
    # Base classes and types
    'AnalysisEngine',
    'BaseAnalysisEngine',
    'EstimationResult',
    # Factory functions
    'get_engine',
    'list_engines',
    'register_engine',
    # Specification utilities
    'load_specifications',
    'get_specification',
    'validate_specification',
]
