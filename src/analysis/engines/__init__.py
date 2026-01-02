"""
Analysis Engine Implementations.

This package contains engine implementations for different statistical
computing environments.

Available Engines
-----------------
- python: Native Python/NumPy estimation (default)
- r: R with fixest package

Engines are automatically registered via the @register_engine decorator
when this package is imported.
"""
from __future__ import annotations

# Import engines to trigger registration
from . import python_engine

# Conditionally import other engines if their dependencies are available
try:
    from . import r_engine
except ImportError:
    pass
