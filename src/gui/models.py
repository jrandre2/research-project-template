"""
Pydantic models for the GUI API.

These models define the data structures exchanged between
the frontend and backend.
"""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class StageStatus(str, Enum):
    """Pipeline stage execution status."""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    UNKNOWN = "unknown"


class StageInfo(BaseModel):
    """Information about a pipeline stage."""
    name: str
    description: str
    version: Optional[str] = None
    status: StageStatus = StageStatus.UNKNOWN
    last_run: Optional[datetime] = None
    has_qa_report: bool = False
    output_files: list[str] = []


class QAMetric(BaseModel):
    """A single QA metric from a stage report."""
    name: str
    value: float | int | str
    threshold: Optional[float] = None
    is_warning: bool = False


class QAReport(BaseModel):
    """QA report for a pipeline stage."""
    stage: str
    timestamp: datetime
    metrics: list[QAMetric]
    file_path: str


class CacheStats(BaseModel):
    """Cache statistics for a stage or overall."""
    stage: Optional[str] = None
    file_count: int
    total_mb: float
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None


class CommentStatus(str, Enum):
    """Review comment status."""
    PENDING = "PENDING"
    ACTION_NEEDED = "VALID - ACTION NEEDED"
    ADDRESSED = "ALREADY ADDRESSED"
    BEYOND_SCOPE = "BEYOND SCOPE"
    INVALID = "INVALID"


class ReviewComment(BaseModel):
    """A single peer review comment."""
    id: str
    type: str  # major, minor, editorial
    title: str
    concern: str
    status: CommentStatus
    response: Optional[str] = None
    files_modified: list[str] = []


class ReviewSummary(BaseModel):
    """Summary statistics for a review cycle."""
    manuscript: str
    cycle_number: int
    source_type: str
    total_comments: int
    by_status: dict[str, int]
    by_type: dict[str, int]


class SupervisionLevel(BaseModel):
    """AI supervision level configuration."""
    level: int  # 0-4
    name: str
    description: str
    is_current: bool = False


class PendingApproval(BaseModel):
    """An action awaiting human approval."""
    id: str
    action: str
    stage: Optional[str] = None
    description: str
    created_at: datetime
    details: Optional[dict] = None
