"""
Word Document Feedback Extraction

Extract comments and track changes from .docx files for integration
with the CENTAUR revision tracking workflow.

Usage:
    from utils.docx_feedback import extract_feedback, format_as_feedback_tracker

    comments, changes = extract_feedback(Path("manuscript_feedback.docx"))
    markdown = format_as_feedback_tracker(comments, changes, "manuscript_feedback.docx")
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET
import zipfile

# python-docx for basic document access
try:
    from docx import Document
    from docx.oxml.ns import qn
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# Word XML namespaces
NAMESPACES = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'w14': 'http://schemas.microsoft.com/office/word/2010/wordml',
    'w15': 'http://schemas.microsoft.com/office/word/2012/wordml',
}


@dataclass
class ExtractedComment:
    """Single comment from Word document."""
    id: str
    author: str
    date: str
    text: str
    paragraph_text: str = ""
    page_hint: Optional[str] = None
    resolved: bool = False

    def title_from_text(self, max_words: int = 6) -> str:
        """Generate a short title from comment text."""
        words = self.text.split()[:max_words]
        title = " ".join(words)
        if len(self.text.split()) > max_words:
            title += "..."
        return title


@dataclass
class ExtractedChange:
    """Single track change from Word document."""
    id: str
    author: str
    date: str
    change_type: str  # 'insertion' | 'deletion' | 'format'
    original_text: str = ""
    new_text: str = ""
    paragraph_context: str = ""


@dataclass
class FeedbackSummary:
    """Summary of all extracted feedback."""
    source_file: str
    extracted_at: str
    comments: list[ExtractedComment] = field(default_factory=list)
    changes: list[ExtractedChange] = field(default_factory=list)
    authors: set[str] = field(default_factory=set)

    @property
    def total_items(self) -> int:
        return len(self.comments) + len(self.changes)

    @property
    def comment_count(self) -> int:
        return len(self.comments)

    @property
    def change_count(self) -> int:
        return len(self.changes)


def check_docx_available() -> None:
    """Raise error if python-docx is not installed."""
    if not DOCX_AVAILABLE:
        raise ImportError(
            "python-docx is required for Word document processing. "
            "Install with: pip install python-docx"
        )


def extract_comments(docx_path: Path) -> list[ExtractedComment]:
    """
    Extract all comments from a Word document.

    Uses direct XML parsing since python-docx doesn't expose comments directly.
    """
    check_docx_available()

    comments = []

    # Open the docx as a zip file to access raw XML
    with zipfile.ZipFile(docx_path, 'r') as zf:
        # Check if comments.xml exists
        if 'word/comments.xml' not in zf.namelist():
            return comments

        # Parse comments.xml
        with zf.open('word/comments.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()

        # Also parse document.xml to get paragraph context
        with zf.open('word/document.xml') as f:
            doc_tree = ET.parse(f)
            doc_root = doc_tree.getroot()

        # Build a map of comment IDs to their anchored text
        comment_ranges = _extract_comment_ranges(doc_root)

        # Extract each comment
        for comment_elem in root.findall('.//w:comment', NAMESPACES):
            comment_id = comment_elem.get(qn('w:id'))
            author = comment_elem.get(qn('w:author'), 'Unknown')
            date_str = comment_elem.get(qn('w:date'), '')

            # Get comment text (may span multiple paragraphs)
            text_parts = []
            for p in comment_elem.findall('.//w:p', NAMESPACES):
                p_text = ''.join(t.text or '' for t in p.findall('.//w:t', NAMESPACES))
                if p_text:
                    text_parts.append(p_text)
            text = '\n'.join(text_parts)

            # Get the paragraph text this comment is attached to
            paragraph_text = comment_ranges.get(comment_id, '')

            # Parse date
            date_formatted = _format_date(date_str)

            comments.append(ExtractedComment(
                id=comment_id or str(len(comments) + 1),
                author=author,
                date=date_formatted,
                text=text,
                paragraph_text=paragraph_text[:200] + '...' if len(paragraph_text) > 200 else paragraph_text,
                resolved=False,  # Would need commentsExtended.xml for this
            ))

    return comments


def _extract_comment_ranges(doc_root: ET.Element) -> dict[str, str]:
    """Extract the text content that each comment is anchored to."""
    ranges = {}

    # Find all paragraphs
    for para in doc_root.findall('.//w:p', NAMESPACES):
        # Get all text in this paragraph
        para_text = ''.join(t.text or '' for t in para.findall('.//w:t', NAMESPACES))

        # Find comment range starts in this paragraph
        for start in para.findall('.//w:commentRangeStart', NAMESPACES):
            comment_id = start.get(qn('w:id'))
            if comment_id:
                ranges[comment_id] = para_text

    return ranges


def extract_track_changes(docx_path: Path) -> list[ExtractedChange]:
    """
    Extract all track changes (insertions/deletions) from a Word document.
    """
    check_docx_available()

    changes = []
    change_id = 0

    with zipfile.ZipFile(docx_path, 'r') as zf:
        with zf.open('word/document.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()

        # Find insertions
        for ins in root.findall('.//w:ins', NAMESPACES):
            change_id += 1
            author = ins.get(qn('w:author'), 'Unknown')
            date_str = ins.get(qn('w:date'), '')

            # Get inserted text
            text = ''.join(t.text or '' for t in ins.findall('.//w:t', NAMESPACES))

            if text.strip():  # Only include non-empty changes
                changes.append(ExtractedChange(
                    id=str(change_id),
                    author=author,
                    date=_format_date(date_str),
                    change_type='insertion',
                    new_text=text,
                ))

        # Find deletions
        for dele in root.findall('.//w:del', NAMESPACES):
            change_id += 1
            author = dele.get(qn('w:author'), 'Unknown')
            date_str = dele.get(qn('w:date'), '')

            # Get deleted text
            text = ''.join(t.text or '' for t in dele.findall('.//w:delText', NAMESPACES))

            if text.strip():  # Only include non-empty changes
                changes.append(ExtractedChange(
                    id=str(change_id),
                    author=author,
                    date=_format_date(date_str),
                    change_type='deletion',
                    original_text=text,
                ))

    return changes


def _format_date(date_str: str) -> str:
    """Format Word date string to readable format."""
    if not date_str:
        return 'Unknown date'

    try:
        # Word uses ISO format: 2025-12-26T10:30:00Z
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d')
    except (ValueError, AttributeError):
        return date_str[:10] if len(date_str) >= 10 else date_str


def extract_feedback(docx_path: Path) -> FeedbackSummary:
    """
    Extract all feedback (comments + track changes) from a Word document.

    Args:
        docx_path: Path to the .docx file

    Returns:
        FeedbackSummary with all extracted items
    """
    docx_path = Path(docx_path)

    if not docx_path.exists():
        raise FileNotFoundError(f"File not found: {docx_path}")

    if not docx_path.suffix.lower() == '.docx':
        raise ValueError(f"Expected .docx file, got: {docx_path.suffix}")

    comments = extract_comments(docx_path)
    changes = extract_track_changes(docx_path)

    # Collect unique authors
    authors = set()
    for c in comments:
        authors.add(c.author)
    for c in changes:
        authors.add(c.author)

    return FeedbackSummary(
        source_file=docx_path.name,
        extracted_at=datetime.now().strftime('%Y-%m-%d'),
        comments=comments,
        changes=changes,
        authors=authors,
    )


def format_as_feedback_tracker(
    summary: FeedbackSummary,
    manuscript_name: str = "main",
) -> str:
    """
    Format extracted feedback as a FEEDBACK_TRACKER.md file.

    Args:
        summary: FeedbackSummary from extract_feedback()
        manuscript_name: Name of the manuscript for tracking

    Returns:
        Markdown string ready to write to file
    """
    lines = [
        "# External Feedback Tracker",
        "",
        f"**Manuscript**: {manuscript_name}",
        "**Type**: EXTERNAL FEEDBACK",
        f"**Source File**: {summary.source_file}",
        f"**Imported**: {summary.extracted_at}",
        f"**Reviewer(s)**: {', '.join(sorted(summary.authors)) or 'Unknown'}",
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
        "| Category | Total | Addressed | Beyond Scope | Pending |",
        "|----------|-------|-----------|--------------|---------|",
        f"| Comments | {summary.comment_count} | 0 | 0 | {summary.comment_count} |",
        f"| Track Changes | {summary.change_count} | 0 | 0 | {summary.change_count} |",
        "",
        "---",
        "",
    ]

    # Add comments section
    if summary.comments:
        lines.append("## Comments")
        lines.append("")

        for i, comment in enumerate(summary.comments, 1):
            title = comment.title_from_text()
            lines.extend([
                f"### Comment {i}: {title}",
                "",
                "**Status**: PENDING TRIAGE",
                f"**Reviewer**: {comment.author}",
                f"**Received**: {comment.date}",
            ])

            if comment.paragraph_text:
                lines.append(f'**Location**: Paragraph beginning "{comment.paragraph_text[:50]}..."')

            lines.extend([
                "",
                "**Reviewer's Concern**:",
                f"> {comment.text}",
                "",
                "**Validity Assessment**: [TO BE ASSESSED]",
                "",
                "**Response**: [TO BE COMPLETED]",
                "",
                "**Files Modified**:",
                "- [ ] [To be determined]",
                "",
                "---",
                "",
            ])

    # Add track changes section
    if summary.changes:
        lines.append("## Track Changes")
        lines.append("")

        for i, change in enumerate(summary.changes, 1):
            change_desc = f"{change.change_type.title()}"
            if change.change_type == 'insertion':
                preview = change.new_text[:50] + '...' if len(change.new_text) > 50 else change.new_text
            else:
                preview = change.original_text[:50] + '...' if len(change.original_text) > 50 else change.original_text

            lines.extend([
                f"### Change {i}: {change_desc}",
                "",
                "**Status**: PENDING TRIAGE",
                f"**Author**: {change.author}",
                f"**Date**: {change.date}",
                f"**Type**: {change.change_type}",
                "",
            ])

            if change.change_type == 'insertion':
                lines.append(f"**Inserted**: `{change.new_text}`")
            elif change.change_type == 'deletion':
                lines.append(f"**Deleted**: `{change.original_text}`")

            lines.extend([
                "",
                "**Accept/Reject**: [TO BE DECIDED]",
                "",
                "---",
                "",
            ])

    # Add verification checklist
    lines.extend([
        "## Verification Checklist",
        "",
        "- [ ] All comments triaged (VALID/INVALID/BEYOND SCOPE)",
        "- [ ] All track changes accepted or rejected",
        "- [ ] Responses documented for each item",
        "- [ ] Modified files listed",
        "- [ ] Manuscript re-rendered and verified",
        "",
    ])

    return '\n'.join(lines)


# Convenience function for CLI
def ingest_docx_feedback(
    docx_path: Path,
    output_path: Path,
    manuscript_name: str = "main",
    dry_run: bool = False,
) -> FeedbackSummary:
    """
    Main entry point for feedback ingestion.

    Args:
        docx_path: Path to Word document with feedback
        output_path: Path to write FEEDBACK_TRACKER.md
        manuscript_name: Manuscript identifier
        dry_run: If True, don't write file, just return summary

    Returns:
        FeedbackSummary with extraction results
    """
    summary = extract_feedback(docx_path)

    if summary.total_items == 0:
        print(f"No comments or track changes found in {docx_path.name}")
        return summary

    markdown = format_as_feedback_tracker(summary, manuscript_name)

    if dry_run:
        print("=== DRY RUN - Would write to:", output_path)
        print("=" * 60)
        print(markdown)
        print("=" * 60)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown)
        print(f"Wrote feedback tracker to: {output_path}")

    print(f"\nExtracted from: {summary.source_file}")
    print(f"  Comments: {summary.comment_count}")
    print(f"  Track changes: {summary.change_count}")
    print(f"  Authors: {', '.join(sorted(summary.authors))}")

    return summary
