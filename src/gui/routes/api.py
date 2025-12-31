"""
API routes - JSON endpoints for HTMX and frontend.

These routes return JSON data consumed by the frontend
via HTMX requests.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from ..models import StageInfo, QAReport, CacheStats
from ..services.pipeline_service import get_pipeline_service
from ..services.qa_service import get_qa_service
from ..services.cache_service import get_cache_service
from ..services.review_service import get_review_service
from ..services.supervision_service import get_supervision_service

router = APIRouter(prefix="/api")


# =============================================================================
# PIPELINE STAGES
# =============================================================================

@router.get("/stages", response_model=list[StageInfo])
async def list_stages():
    """Get all pipeline stages with their status."""
    service = get_pipeline_service()
    return service.discover_stages()


@router.get("/stages/{stage_name}", response_model=StageInfo)
async def get_stage(stage_name: str):
    """Get details for a specific stage."""
    service = get_pipeline_service()
    stage = service.get_stage(stage_name)
    if not stage:
        raise HTTPException(status_code=404, detail=f"Stage '{stage_name}' not found")
    return stage


@router.post("/stages/{stage_name}/run")
async def run_stage(stage_name: str):
    """
    Trigger execution of a pipeline stage.

    Returns an HTML fragment for HTMX to update the UI.
    """
    service = get_pipeline_service()
    stage = service.get_stage(stage_name)
    if not stage:
        raise HTTPException(status_code=404, detail=f"Stage '{stage_name}' not found")

    # For now, return a placeholder. WebSocket implementation will come later.
    return HTMLResponse(
        content=f"""
        <div class="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p class="text-blue-800">
                Stage <strong>{stage_name}</strong> execution started.
                <span class="animate-pulse">Running...</span>
            </p>
        </div>
        """,
        status_code=202,
    )


# =============================================================================
# QA REPORTS
# =============================================================================

@router.get("/qa", response_model=list[QAReport])
async def list_qa_reports(stage: Optional[str] = None):
    """List QA reports, optionally filtered by stage."""
    service = get_qa_service()
    return service.list_reports(stage=stage)


@router.get("/qa/{stage_name}/latest", response_model=Optional[QAReport])
async def get_latest_qa(stage_name: str):
    """Get the most recent QA report for a stage."""
    service = get_qa_service()
    report = service.get_latest_report(stage_name)
    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"No QA report found for stage '{stage_name}'"
        )
    return report


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

@router.get("/cache", response_model=list[CacheStats])
async def get_cache_stats():
    """Get cache statistics for all stages."""
    service = get_cache_service()
    return service.get_stats()


@router.get("/cache/total")
async def get_cache_total():
    """Get aggregate cache statistics as HTML."""
    service = get_cache_service()
    stats = service.get_total_stats()
    return HTMLResponse(
        content=f"{stats.file_count} files, {stats.total_mb:.2f} MB",
        status_code=200,
    )


@router.post("/cache/clear")
async def clear_cache(stage: Optional[str] = None):
    """Clear cache for a specific stage or all stages."""
    service = get_cache_service()
    count = service.clear(stage)
    return HTMLResponse(
        content=f"""
        <div class="p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg text-green-800 dark:text-green-200">
            Cleared {count} cache files{f' for {stage}' if stage else ''}.
        </div>
        """,
        status_code=200,
    )


# =============================================================================
# REVIEW TRACKER
# =============================================================================

@router.get("/reviews/tracker")
async def get_review_tracker(manuscript: str = 'main'):
    """Get parsed review tracker data as JSON."""
    service = get_review_service()
    tracker = service.parse_tracker(manuscript)

    if not tracker:
        return {"has_review": False, "comments": [], "summary": {}}

    return {
        "has_review": service.has_active_review(manuscript),
        "manuscript": tracker.manuscript,
        "review_number": tracker.review_number,
        "discipline": tracker.discipline,
        "last_updated": tracker.last_updated,
        "comments": [
            {
                "id": c.id,
                "title": c.title,
                "type": c.comment_type,
                "status": c.status,
                "concern": c.concern[:200] + "..." if len(c.concern) > 200 else c.concern,
                "response": c.response[:100] + "..." if len(c.response) > 100 else c.response,
                "files_modified": c.files_modified,
            }
            for c in tracker.comments
        ],
        "summary": tracker.get_summary(),
        "by_status": {
            status: [c.id for c in comments]
            for status, comments in tracker.get_by_status().items()
        },
    }


@router.get("/reviews/comments")
async def get_review_comments(manuscript: str = 'main'):
    """Get all review comments grouped by status for Kanban view."""
    service = get_review_service()
    tracker = service.parse_tracker(manuscript)

    if not tracker:
        return {"groups": {}}

    by_status = tracker.get_by_status()

    return {
        "groups": {
            status: [
                {
                    "id": c.id,
                    "title": c.title,
                    "type": c.comment_type,
                    "concern": c.concern[:150] + "..." if len(c.concern) > 150 else c.concern,
                }
                for c in comments
            ]
            for status, comments in by_status.items()
        }
    }


@router.put("/reviews/comments/{comment_id}/status")
async def update_comment_status(
    comment_id: str,
    new_status: str,
    manuscript: str = 'main'
):
    """Update a comment's status."""
    # Validate status
    valid_statuses = [
        'PENDING',
        'VALID - ACTION NEEDED',
        'ALREADY ADDRESSED',
        'BEYOND SCOPE',
        'INVALID',
    ]

    if new_status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {valid_statuses}"
        )

    service = get_review_service()
    success = service.update_comment_status(manuscript, comment_id, new_status)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Comment '{comment_id}' not found or update failed"
        )

    return {"status": "ok", "comment_id": comment_id, "new_status": new_status}


@router.get("/reviews/summary")
async def get_review_summary(manuscript: str = 'main'):
    """Get review summary statistics."""
    service = get_review_service()
    tracker = service.parse_tracker(manuscript)

    if not tracker:
        return {"has_review": False}

    summary = tracker.get_summary()
    total = sum(s['total'] for s in summary.values())
    addressed = sum(s['addressed'] for s in summary.values())
    action_needed = sum(s['action_needed'] for s in summary.values())

    return {
        "has_review": service.has_active_review(manuscript),
        "total_comments": total,
        "addressed": addressed,
        "action_needed": action_needed,
        "progress_pct": round(addressed / total * 100) if total > 0 else 0,
        "by_type": summary,
    }


# =============================================================================
# SUPERVISION CONTROLS
# =============================================================================

@router.get("/supervision/level")
async def get_supervision_level():
    """Get current supervision level."""
    service = get_supervision_service()
    level = service.get_level()
    return {
        "level": level.level,
        "name": level.name,
        "description": level.description,
        "updated_at": level.updated_at,
        "updated_by": level.updated_by,
    }


@router.put("/supervision/level")
async def set_supervision_level(level: int):
    """Set supervision level (0-4)."""
    if level < 0 or level > 4:
        raise HTTPException(
            status_code=400,
            detail="Level must be between 0 and 4"
        )

    service = get_supervision_service()
    updated = service.set_level(level)
    return {
        "status": "ok",
        "level": updated.level,
        "name": updated.name,
        "description": updated.description,
    }


@router.get("/supervision/pending")
async def get_pending_actions():
    """Get pending actions awaiting approval."""
    service = get_supervision_service()
    actions = service.get_pending_actions()
    return {
        "count": len(actions),
        "actions": [
            {
                "id": a.id,
                "action_type": a.action_type,
                "description": a.description,
                "created_at": a.created_at,
            }
            for a in actions
        ],
    }


@router.post("/supervision/pending/{action_id}/approve")
async def approve_action(action_id: str):
    """Approve a pending action."""
    service = get_supervision_service()
    success = service.approve_action(action_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Action '{action_id}' not found"
        )

    return {"status": "ok", "action_id": action_id, "result": "approved"}


@router.post("/supervision/pending/{action_id}/reject")
async def reject_action(action_id: str, reason: str = ""):
    """Reject a pending action."""
    service = get_supervision_service()
    success = service.reject_action(action_id, reason)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Action '{action_id}' not found"
        )

    return {"status": "ok", "action_id": action_id, "result": "rejected"}


@router.get("/supervision/audit")
async def get_audit_log(limit: int = 50):
    """Get recent audit log entries."""
    service = get_supervision_service()
    entries = service.get_audit_log(limit)
    return {
        "count": len(entries),
        "entries": [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "action": e.action,
                "result": e.result,
                "details": e.details,
            }
            for e in entries
        ],
    }


@router.delete("/supervision/audit")
async def clear_audit_log():
    """Clear the audit log."""
    service = get_supervision_service()
    count = service.clear_audit_log()
    return HTMLResponse(
        content=f"""
        <div class="p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg text-green-800 dark:text-green-200">
            Cleared {count} audit entries.
        </div>
        """,
        status_code=200,
    )
