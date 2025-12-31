"""Route handlers for the GUI."""
from .dashboard import router as dashboard_router
from .api import router as api_router
from .websocket import router as websocket_router

__all__ = ['dashboard_router', 'api_router', 'websocket_router']
