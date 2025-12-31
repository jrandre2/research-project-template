"""
Dashboard routes - HTML page endpoints.

These routes serve the main HTML pages using Jinja2 templates.
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..config import TEMPLATES_DIR
from ..services.pipeline_service import get_pipeline_service
from ..services.qa_service import get_qa_service

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page showing pipeline stages and status."""
    pipeline_service = get_pipeline_service()
    qa_service = get_qa_service()

    stages = pipeline_service.discover_stages()

    # Enrich stages with latest QA metrics
    for stage in stages:
        report = qa_service.get_latest_report(stage.name)
        if report:
            stage.last_run = report.timestamp

    return templates.TemplateResponse(
        "pages/dashboard.html",
        {
            "request": request,
            "stages": stages,
            "active_page": "dashboard",
        }
    )


@router.get("/reviews", response_class=HTMLResponse)
async def reviews(request: Request):
    """Review tracker Kanban page."""
    return templates.TemplateResponse(
        "pages/reviews.html",
        {
            "request": request,
            "active_page": "reviews",
        }
    )


@router.get("/supervision", response_class=HTMLResponse)
async def supervision(request: Request):
    """AI supervision controls page."""
    return templates.TemplateResponse(
        "pages/supervision.html",
        {
            "request": request,
            "active_page": "supervision",
        }
    )
