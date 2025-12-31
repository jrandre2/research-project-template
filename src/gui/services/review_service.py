"""
Review service for parsing and managing REVISION_TRACKER.md.

Parses the markdown-based revision tracker and provides
structured access to review comments for the Kanban UI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import PROJECT_ROOT, MANUSCRIPTS


@dataclass
class TrackerComment:
    """A parsed comment from the revision tracker."""
    id: str
    title: str
    comment_type: str  # 'major', 'minor', 'editorial'
    status: str  # 'VALID - ACTION NEEDED', 'ALREADY ADDRESSED', 'BEYOND SCOPE', 'INVALID', 'PENDING'
    concern: str
    response: str = ''
    files_modified: list[str] = field(default_factory=list)
    line_start: int = 0  # Line number in source file
    line_end: int = 0


@dataclass
class TrackerData:
    """Parsed revision tracker data."""
    manuscript: str
    review_number: Optional[int] = None
    discipline: Optional[str] = None
    last_updated: Optional[str] = None
    comments: list[TrackerComment] = field(default_factory=list)
    raw_content: str = ''
    file_path: Optional[Path] = None

    def get_summary(self) -> dict[str, dict[str, int]]:
        """Get summary statistics by type and status."""
        summary = {
            'major': {'total': 0, 'addressed': 0, 'beyond_scope': 0, 'pending': 0, 'action_needed': 0},
            'minor': {'total': 0, 'addressed': 0, 'beyond_scope': 0, 'pending': 0, 'action_needed': 0},
        }

        for comment in self.comments:
            ctype = comment.comment_type
            if ctype not in summary:
                continue

            summary[ctype]['total'] += 1

            if comment.status == 'ALREADY ADDRESSED':
                summary[ctype]['addressed'] += 1
            elif comment.status == 'BEYOND SCOPE':
                summary[ctype]['beyond_scope'] += 1
            elif comment.status == 'VALID - ACTION NEEDED':
                summary[ctype]['action_needed'] += 1
            else:
                summary[ctype]['pending'] += 1

        return summary

    def get_by_status(self) -> dict[str, list[TrackerComment]]:
        """Group comments by status for Kanban view."""
        groups = {
            'PENDING': [],
            'VALID - ACTION NEEDED': [],
            'ALREADY ADDRESSED': [],
            'BEYOND SCOPE': [],
            'INVALID': [],
        }

        for comment in self.comments:
            status = comment.status
            if status in groups:
                groups[status].append(comment)
            else:
                groups['PENDING'].append(comment)

        return groups


class ReviewService:
    """Service for managing review tracker."""

    def __init__(self):
        self.manuscripts = MANUSCRIPTS

    def get_tracker_path(self, manuscript: str = 'main') -> Optional[Path]:
        """Get path to REVISION_TRACKER.md for a manuscript."""
        if manuscript not in self.manuscripts:
            return None

        ms_config = self.manuscripts[manuscript]
        ms_dir = ms_config.get('dir', PROJECT_ROOT / 'manuscript_quarto')
        tracker_path = ms_dir / 'REVISION_TRACKER.md'

        if tracker_path.exists():
            return tracker_path
        return None

    def parse_tracker(self, manuscript: str = 'main') -> Optional[TrackerData]:
        """
        Parse REVISION_TRACKER.md for a manuscript.

        Parameters
        ----------
        manuscript : str
            Manuscript key (e.g., 'main')

        Returns
        -------
        TrackerData or None
            Parsed tracker data, or None if file doesn't exist
        """
        tracker_path = self.get_tracker_path(manuscript)
        if not tracker_path:
            return None

        content = tracker_path.read_text()
        return self._parse_content(content, manuscript, tracker_path)

    def _parse_content(
        self,
        content: str,
        manuscript: str,
        file_path: Optional[Path] = None
    ) -> TrackerData:
        """Parse tracker content into structured data."""
        lines = content.split('\n')
        tracker = TrackerData(
            manuscript=manuscript,
            raw_content=content,
            file_path=file_path,
        )

        # Parse header info
        for line in lines[:20]:
            if '**Review**:' in line:
                match = re.search(r'#(\d+)', line)
                if match:
                    tracker.review_number = int(match.group(1))
            elif '**Discipline**:' in line:
                match = re.search(r'\[([^\]]+)\]', line)
                if match:
                    tracker.discipline = match.group(1)
            elif '**Last Updated**:' in line:
                match = re.search(r'\[([^\]]+)\]', line)
                if match:
                    tracker.last_updated = match.group(1)

        # Parse comments
        tracker.comments = self._parse_comments(lines)

        return tracker

    def _parse_comments(self, lines: list[str]) -> list[TrackerComment]:
        """Parse comments from tracker lines."""
        comments = []
        current_section = None  # 'major' or 'minor'
        current_comment = None
        current_field = None
        comment_num = 0

        i = 0
        while i < len(lines):
            line = lines[i]

            # Detect section headers
            if line.strip() == '## Major Comments':
                current_section = 'major'
                comment_num = 0
            elif line.strip() == '## Minor Comments':
                current_section = 'minor'
                comment_num = 0
            elif line.strip().startswith('## ') and current_section:
                # New section, end current section
                current_section = None

            # Detect comment headers
            elif current_section and line.startswith('### '):
                # Save previous comment
                if current_comment:
                    comments.append(current_comment)

                comment_num += 1
                title_match = re.match(r'###\s+(?:Comment|Minor)\s+\d+:\s*(.+)', line)
                title = title_match.group(1) if title_match else line[4:].strip()

                current_comment = TrackerComment(
                    id=f"{current_section[0].upper()}{comment_num}",
                    title=title,
                    comment_type=current_section,
                    status='PENDING',
                    concern='',
                    line_start=i + 1,
                )
                current_field = None

            # Parse fields within a comment
            elif current_comment:
                if line.startswith('**Status**:'):
                    status_match = re.search(r'\*\*Status\*\*:\s*\[?([^\]]+)\]?', line)
                    if status_match:
                        status = status_match.group(1).strip()
                        # Normalize status
                        if 'ACTION NEEDED' in status.upper():
                            current_comment.status = 'VALID - ACTION NEEDED'
                        elif 'ADDRESSED' in status.upper():
                            current_comment.status = 'ALREADY ADDRESSED'
                        elif 'BEYOND' in status.upper():
                            current_comment.status = 'BEYOND SCOPE'
                        elif 'INVALID' in status.upper():
                            current_comment.status = 'INVALID'
                        else:
                            current_comment.status = 'PENDING'
                    current_field = None

                elif line.startswith("**Reviewer's Concern**:") or line.startswith("**Concern**:"):
                    current_field = 'concern'

                elif line.startswith('**Response**:'):
                    current_field = 'response'

                elif line.startswith('**Files Modified**:'):
                    current_field = 'files'

                elif line.startswith('**'):
                    current_field = None

                elif line.startswith('---'):
                    current_comment.line_end = i
                    current_field = None

                elif current_field:
                    text = line.strip()
                    if text.startswith('>'):
                        text = text[1:].strip()

                    if current_field == 'concern' and text:
                        if current_comment.concern:
                            current_comment.concern += '\n' + text
                        else:
                            current_comment.concern = text

                    elif current_field == 'response' and text:
                        if current_comment.response:
                            current_comment.response += '\n' + text
                        else:
                            current_comment.response = text

                    elif current_field == 'files' and text.startswith('-'):
                        file_match = re.match(r'-\s*`?([^`\s]+)`?', text)
                        if file_match:
                            current_comment.files_modified.append(file_match.group(1))

            i += 1

        # Don't forget the last comment
        if current_comment:
            current_comment.line_end = len(lines)
            comments.append(current_comment)

        return comments

    def update_comment_status(
        self,
        manuscript: str,
        comment_id: str,
        new_status: str
    ) -> bool:
        """
        Update a comment's status in the tracker file.

        Parameters
        ----------
        manuscript : str
            Manuscript key
        comment_id : str
            Comment ID (e.g., 'M1', 'm2')
        new_status : str
            New status value

        Returns
        -------
        bool
            True if update succeeded
        """
        tracker_path = self.get_tracker_path(manuscript)
        if not tracker_path:
            return False

        tracker = self.parse_tracker(manuscript)
        if not tracker:
            return False

        # Find the comment
        target_comment = None
        for comment in tracker.comments:
            if comment.id.upper() == comment_id.upper():
                target_comment = comment
                break

        if not target_comment:
            return False

        # Read file and update the status line
        lines = tracker_path.read_text().split('\n')

        # Find and update the status line within the comment's range
        for i in range(target_comment.line_start - 1, min(target_comment.line_end, len(lines))):
            if lines[i].startswith('**Status**:'):
                lines[i] = f'**Status**: {new_status}'
                break

        # Write back
        tracker_path.write_text('\n'.join(lines))
        return True

    def has_active_review(self, manuscript: str = 'main') -> bool:
        """Check if there's an active review for this manuscript."""
        tracker = self.parse_tracker(manuscript)
        if not tracker:
            return False
        # Check if it's not just a template
        return bool(tracker.comments) or tracker.review_number is not None


# Singleton instance
_review_service: Optional[ReviewService] = None


def get_review_service() -> ReviewService:
    """Get the review service singleton."""
    global _review_service
    if _review_service is None:
        _review_service = ReviewService()
    return _review_service
