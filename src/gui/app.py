"""
CENTAUR GUI - FastAPI Application.

This module creates and configures the FastAPI application
for the CENTAUR research pipeline dashboard.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import STATIC_DIR, TEMPLATES_DIR, GUI_DEBUG
from .routes import dashboard_router, api_router, websocket_router


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        Configured application instance.
    """
    app = FastAPI(
        title="CENTAUR Dashboard",
        description="Lightweight web dashboard for CENTAUR research pipeline",
        version="0.1.0",
        debug=GUI_DEBUG,
    )

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Include routers
    app.include_router(dashboard_router)
    app.include_router(api_router)
    app.include_router(websocket_router)

    return app


# Application instance for uvicorn
app = create_app()


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """
    Run the development server.

    Parameters
    ----------
    host : str
        Host to bind to.
    port : int
        Port to bind to.
    reload : bool
        Enable auto-reload on code changes.
    """
    import os
    import uvicorn
    from .config import PROJECT_ROOT

    # Change to src directory so uvicorn can find gui.app
    src_dir = PROJECT_ROOT / "src"
    os.chdir(src_dir)

    uvicorn.run(
        "gui.app:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[str(src_dir / "gui")] if reload else None,
    )


if __name__ == "__main__":
    run_server()
