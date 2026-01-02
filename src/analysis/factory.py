"""
Analysis Engine Factory and Registry.

Provides centralized engine management with automatic registration
and configuration-based defaults.

Usage
-----
    from analysis.factory import get_engine, list_engines, register_engine

    # Get default engine
    engine = get_engine()

    # Get specific engine
    engine = get_engine('r')

    # List available engines
    engines = list_engines()

    # Register custom engine
    @register_engine('custom')
    class CustomEngine(BaseAnalysisEngine):
        ...
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .base import AnalysisEngine

# Engine registry: name -> engine class
_engine_registry: dict[str, type] = {}


def register_engine(name: str):
    """
    Decorator to register an engine implementation.

    Parameters
    ----------
    name : str
        Engine name (e.g., 'python', 'r', 'stata')

    Returns
    -------
    Callable
        Decorator function

    Example
    -------
        @register_engine('r')
        class REngine(BaseAnalysisEngine):
            ...
    """
    def decorator(cls):
        _engine_registry[name] = cls
        return cls
    return decorator


def get_engine(name: Optional[str] = None) -> 'AnalysisEngine':
    """
    Get an analysis engine instance.

    Parameters
    ----------
    name : str, optional
        Engine name ('python', 'r', 'stata', 'julia').
        If not specified, uses ANALYSIS_ENGINE from config.

    Returns
    -------
    AnalysisEngine
        Configured engine instance

    Raises
    ------
    ValueError
        If engine name is not recognized
    ImportError
        If the engine module is not available
    """
    # Import engines to ensure they are registered
    _ensure_engines_loaded()

    # Get config if name not specified
    if name is None:
        name = _get_default_engine()

    name = name.lower()

    if name not in _engine_registry:
        available = ', '.join(sorted(_engine_registry.keys()))
        raise ValueError(
            f"Unknown engine: '{name}'. Available engines: {available}"
        )

    engine_cls = _engine_registry[name]
    return engine_cls()


def list_engines() -> dict[str, bool]:
    """
    List all registered engines and their availability.

    Returns
    -------
    dict[str, bool]
        Dictionary of engine_name -> is_available

    Example
    -------
        >>> list_engines()
        {'python': True, 'r': True, 'stata': False}
    """
    _ensure_engines_loaded()

    result = {}
    for name, engine_cls in sorted(_engine_registry.items()):
        try:
            engine = engine_cls()
            available, _ = engine.validate_installation()
            result[name] = available
        except Exception:
            result[name] = False

    return result


def get_engine_info(name: str) -> dict:
    """
    Get detailed information about an engine.

    Parameters
    ----------
    name : str
        Engine name

    Returns
    -------
    dict
        Engine information including:
        - name: str
        - available: bool
        - version: str
        - message: str
    """
    _ensure_engines_loaded()

    if name not in _engine_registry:
        return {
            'name': name,
            'available': False,
            'version': 'N/A',
            'message': f"Unknown engine: {name}",
        }

    try:
        engine = _engine_registry[name]()
        available, message = engine.validate_installation()
        return {
            'name': name,
            'available': available,
            'version': engine.version if available else 'N/A',
            'message': message,
        }
    except Exception as e:
        return {
            'name': name,
            'available': False,
            'version': 'N/A',
            'message': str(e),
        }


def _get_default_engine() -> str:
    """Get default engine from config."""
    try:
        # Add parent to path for config import
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import ANALYSIS_ENGINE
        return ANALYSIS_ENGINE
    except ImportError:
        return 'python'


def _ensure_engines_loaded() -> None:
    """Ensure all engine modules are imported for registration."""
    if _engine_registry:
        # Already loaded
        return

    # Import engine modules to trigger @register_engine decorators
    try:
        from . import engines  # noqa: F401
    except ImportError:
        pass
