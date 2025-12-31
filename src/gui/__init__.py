"""
CENTAUR GUI - Lightweight web dashboard for pipeline supervision.

This package provides a FastAPI-based web interface for monitoring
and controlling the CENTAUR research pipeline. The CLI remains the
primary interface; this GUI serves as a human supervision layer.

Usage
-----
    python src/pipeline.py gui
    # or
    uvicorn src.gui.app:app --reload --port 8000
"""
from .app import create_app

__all__ = ['create_app']
