#!/usr/bin/env python3
"""
Review Models for Stage 07: Review Management.

This module provides dataclasses for structured review metadata,
supporting both synthetic (AI-generated) and actual (journal) peer reviews.

Usage
-----
    from stages._review_models import ReviewMetadata, ReviewComment

    # Create metadata for an actual review
    metadata = ReviewMetadata(
        manuscript='main',
        cycle_number=1,
        source_type='actual',
        journal='JEEM',
        submission_round='R&R1',
    )

    # Create metadata for a synthetic review
    metadata = ReviewMetadata(
        manuscript='main',
        cycle_number=2,
        source_type='synthetic',
        focus='economics',
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import yaml


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

SourceType = Literal['synthetic', 'actual']
CommentStatus = Literal[
    'VALID - ACTION NEEDED',
    'ALREADY ADDRESSED',
    'BEYOND SCOPE',
    'INVALID',
    'PENDING',  # Not yet triaged
]
CommentType = Literal['major', 'minor', 'editorial']
ValidityAssessment = Literal['VALID', 'PARTIALLY VALID', 'INVALID']


# =============================================================================
# REVIEW METADATA
# =============================================================================

@dataclass
class ReviewMetadata:
    """
    Metadata for a single review cycle.

    Supports both synthetic (AI-generated) and actual (journal) peer reviews
    with appropriate fields for each type.

    Attributes
    ----------
    manuscript : str
        Manuscript key (e.g., 'main', 'short')
    cycle_number : int
        Sequential review number (1, 2, 3...)
    source_type : SourceType
        Either 'synthetic' or 'actual'
    focus : str, optional
        Focus area for synthetic reviews (e.g., 'economics', 'methods')
    journal : str, optional
        Journal name for actual reviews
    submission_round : str, optional
        Submission round (e.g., 'initial', 'R&R1', 'R&R2')
    """

    # Core identifiers
    manuscript: str
    cycle_number: int
    source_type: SourceType = 'synthetic'

    # Generated review ID (computed from manuscript and cycle)
    review_id: str = field(default='', init=False)

    # Synthetic review fields
    focus: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    regeneratable: bool = True

    # Actual review fields
    journal: Optional[str] = None
    journal_abbrev: Optional[str] = None
    submission_round: Optional[str] = None
    decision: Optional[str] = None
    decision_date: Optional[datetime] = None
    editor_name: Optional[str] = None
    reviewer_ids: list[str] = field(default_factory=list)
    source_file: Optional[Path] = None

    # Git tracking
    start_commit: Optional[str] = None
    end_commit: Optional[str] = None
    response_commits: list[str] = field(default_factory=list)
    git_tag: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    archived_at: Optional[datetime] = None

    # File references
    tracker_file: Optional[Path] = None
    response_letter_file: Optional[Path] = None
    diff_file: Optional[Path] = None

    def __post_init__(self):
        """Generate review_id from manuscript and cycle."""
        year = self.created_at.strftime('%Y')
        self.review_id = f"{self.manuscript}-{year}-r{self.cycle_number}"

    def is_synthetic(self) -> bool:
        """Check if this is a synthetic review."""
        return self.source_type == 'synthetic'

    def is_actual(self) -> bool:
        """Check if this is an actual journal review."""
        return self.source_type == 'actual'

    def to_yaml_frontmatter(self) -> str:
        """Convert metadata to YAML frontmatter string."""
        data = {
            'review_id': self.review_id,
            'manuscript': self.manuscript,
            'cycle_number': self.cycle_number,
            'source_type': self.source_type,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }

        # Add source-specific fields
        if self.is_synthetic():
            if self.focus:
                data['focus'] = self.focus
            if self.llm_provider:
                data['llm_provider'] = self.llm_provider
            if self.llm_model:
                data['llm_model'] = self.llm_model
        else:
            if self.journal:
                data['journal'] = self.journal
            if self.journal_abbrev:
                data['journal_abbrev'] = self.journal_abbrev
            if self.submission_round:
                data['submission_round'] = self.submission_round
            if self.decision:
                data['decision'] = self.decision
            if self.decision_date:
                data['decision_date'] = self.decision_date.isoformat()
            if self.reviewer_ids:
                data['reviewer_ids'] = self.reviewer_ids

        # Add git tracking
        if self.start_commit:
            data['start_commit'] = self.start_commit
        if self.end_commit:
            data['end_commit'] = self.end_commit
        if self.git_tag:
            data['git_tag'] = self.git_tag

        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'ReviewMetadata':
        """Create ReviewMetadata from YAML string."""
        data = yaml.safe_load(yaml_str)
        if not data:
            raise ValueError("Empty YAML data")

        # Parse dates
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        updated_at = data.get('updated_at')
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now()

        decision_date = data.get('decision_date')
        if isinstance(decision_date, str):
            decision_date = datetime.fromisoformat(decision_date)

        return cls(
            manuscript=data.get('manuscript', 'main'),
            cycle_number=data.get('cycle_number', 1),
            source_type=data.get('source_type', 'synthetic'),
            focus=data.get('focus'),
            llm_provider=data.get('llm_provider'),
            llm_model=data.get('llm_model'),
            journal=data.get('journal'),
            journal_abbrev=data.get('journal_abbrev'),
            submission_round=data.get('submission_round'),
            decision=data.get('decision'),
            decision_date=decision_date,
            reviewer_ids=data.get('reviewer_ids', []),
            start_commit=data.get('start_commit'),
            end_commit=data.get('end_commit'),
            git_tag=data.get('git_tag'),
            created_at=created_at,
            updated_at=updated_at,
        )


# =============================================================================
# REVIEW COMMENT
# =============================================================================

@dataclass
class ReviewComment:
    """
    A single reviewer comment with tracking.

    Attributes
    ----------
    comment_id : str
        Unique identifier (e.g., 'R1-M1' for Reviewer 1, Major 1)
    reviewer_id : str
        Reviewer identifier ('R1', 'R2', 'synthetic', etc.)
    comment_type : CommentType
        Type of comment ('major', 'minor', 'editorial')
    original_text : str
        Verbatim reviewer comment
    status : CommentStatus
        Current triage status
    """

    comment_id: str
    reviewer_id: str
    comment_type: CommentType
    original_text: str

    # Classification
    status: CommentStatus = 'PENDING'
    validity_assessment: Optional[ValidityAssessment] = None
    validity_explanation: str = ''

    # Response tracking
    response_text: str = ''
    files_modified: list[str] = field(default_factory=list)
    commits: list[str] = field(default_factory=list)
    line_references: dict[str, list[int]] = field(default_factory=dict)

    def is_addressed(self) -> bool:
        """Check if comment has been addressed."""
        return self.status in ('ALREADY ADDRESSED', 'VALID - ACTION NEEDED') and bool(self.response_text)

    def needs_action(self) -> bool:
        """Check if comment still needs action."""
        return self.status == 'VALID - ACTION NEEDED' and not self.response_text


# =============================================================================
# FRONTMATTER UTILITIES
# =============================================================================

def parse_frontmatter(content: str) -> tuple[Optional[dict], str]:
    """
    Parse YAML frontmatter from markdown content.

    Parameters
    ----------
    content : str
        Markdown content potentially containing YAML frontmatter

    Returns
    -------
    tuple[Optional[dict], str]
        Tuple of (frontmatter_dict, remaining_content)
        frontmatter_dict is None if no frontmatter found
    """
    if not content.startswith('---'):
        return None, content

    # Find end of frontmatter
    end_match = content.find('\n---', 3)
    if end_match == -1:
        return None, content

    frontmatter_str = content[4:end_match]
    remaining = content[end_match + 4:].lstrip('\n')

    try:
        frontmatter = yaml.safe_load(frontmatter_str)
        return frontmatter, remaining
    except yaml.YAMLError:
        return None, content


def add_frontmatter(content: str, metadata: ReviewMetadata) -> str:
    """
    Add or replace YAML frontmatter in markdown content.

    Parameters
    ----------
    content : str
        Markdown content
    metadata : ReviewMetadata
        Metadata to add as frontmatter

    Returns
    -------
    str
        Content with frontmatter
    """
    # Remove existing frontmatter if present
    _, body = parse_frontmatter(content)

    # Build new content with frontmatter
    frontmatter = metadata.to_yaml_frontmatter()
    return f"---\n{frontmatter}---\n\n{body}"
