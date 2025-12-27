#!/usr/bin/env python3
"""
Stage 06: Manuscript Validation

Purpose: Validate manuscript against journal requirements.

This stage handles:
- Word count validation
- Abstract length check
- Figure format/resolution checks
- Required section validation
- Citation format validation

Input Files
-----------
- manuscript_quarto/index.qmd
- manuscript_quarto/appendix-*.qmd
- manuscript_quarto/journal_configs/<journal>.yml

Output Files
------------
- data_work/diagnostics/submission_validation.md

Usage
-----
    python src/pipeline.py validate_submission --journal jeem
    python src/pipeline.py validate_submission -j natural_hazards --report
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import re
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import (
    get_project_root,
    get_data_dir,
    load_config,
    ensure_dir,
)
from stages._qa_utils import generate_qa_report, QAMetrics


# ============================================================
# CONFIGURATION
# ============================================================

MANUSCRIPT_DIR = 'manuscript_quarto'
MAIN_FILE = 'index.qmd'


# ============================================================
# VALIDATION RESULT CLASSES
# ============================================================

@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    severity: str = 'error'  # 'error', 'warning', 'info'
    actual_value: Optional[str] = None
    expected_value: Optional[str] = None


@dataclass
class ManuscriptValidation:
    """Complete manuscript validation results."""
    journal: str
    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """All critical checks passed."""
        return not any(c.severity == 'error' and not c.passed for c in self.checks)

    @property
    def n_errors(self) -> int:
        """Number of failed error-level checks."""
        return sum(1 for c in self.checks if c.severity == 'error' and not c.passed)

    @property
    def n_warnings(self) -> int:
        """Number of failed warning-level checks."""
        return sum(1 for c in self.checks if c.severity == 'warning' and not c.passed)

    def format(self) -> str:
        """Format as string."""
        lines = [
            "=" * 60,
            f"MANUSCRIPT VALIDATION: {self.journal.upper()}",
            "=" * 60,
            "",
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            f"Errors: {self.n_errors}",
            f"Warnings: {self.n_warnings}",
            "",
            "-" * 60,
        ]

        # Group by severity
        for severity in ['error', 'warning', 'info']:
            checks = [c for c in self.checks if c.severity == severity]
            if checks:
                lines.append(f"\n{severity.upper()}S:")
                for c in checks:
                    status = "PASS" if c.passed else "FAIL"
                    lines.append(f"  [{status}] {c.name}")
                    lines.append(f"        {c.message}")
                    if c.actual_value and c.expected_value:
                        lines.append(f"        Actual: {c.actual_value}, Expected: {c.expected_value}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Format as markdown."""
        lines = [
            f"# Manuscript Validation Report",
            "",
            f"**Journal:** {self.journal}",
            f"**Status:** {'PASSED' if self.passed else 'FAILED'}",
            "",
            "## Summary",
            "",
            f"- Errors: {self.n_errors}",
            f"- Warnings: {self.n_warnings}",
            f"- Total checks: {len(self.checks)}",
            "",
            "## Details",
            "",
        ]

        for severity in ['error', 'warning', 'info']:
            checks = [c for c in self.checks if c.severity == severity]
            if checks:
                lines.append(f"### {severity.title()}s")
                lines.append("")
                for c in checks:
                    emoji = "x" if not c.passed else "v"
                    lines.append(f"- [{emoji}] **{c.name}**: {c.message}")
                lines.append("")

        return "\n".join(lines)


# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def count_words(text: str) -> int:
    """Count words in text, excluding YAML frontmatter and code blocks."""
    # Remove YAML frontmatter
    text = re.sub(r'^---.*?---', '', text, flags=re.DOTALL)
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r'`[^`]+`', '', text)
    # Remove markdown links (keep text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove images
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Count words
    words = text.split()
    return len(words)


def extract_abstract(text: str) -> Optional[str]:
    """Extract abstract from manuscript."""
    # Look for abstract in YAML frontmatter
    yaml_match = re.search(r'^---.*?abstract:\s*[|>]?\s*(.*?)(?=\n\w+:|---)', text, re.DOTALL)
    if yaml_match:
        return yaml_match.group(1).strip()

    # Look for abstract section
    section_match = re.search(r'#+\s*Abstract\s*\n(.*?)(?=\n#+|\Z)', text, re.DOTALL | re.IGNORECASE)
    if section_match:
        return section_match.group(1).strip()

    return None


def check_word_count(
    manuscript_path: Path,
    max_words: Optional[int] = None
) -> ValidationCheck:
    """Check manuscript word count."""
    text = manuscript_path.read_text()
    word_count = count_words(text)

    if max_words:
        passed = word_count <= max_words
        message = f"Word count: {word_count:,}" + (f" (limit: {max_words:,})" if not passed else "")
    else:
        passed = True
        message = f"Word count: {word_count:,}"

    return ValidationCheck(
        name='Word Count',
        passed=passed,
        message=message,
        severity='error' if max_words else 'info',
        actual_value=str(word_count),
        expected_value=f"<= {max_words}" if max_words else None
    )


def check_abstract_length(
    manuscript_path: Path,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None
) -> ValidationCheck:
    """Check abstract word count."""
    text = manuscript_path.read_text()
    abstract = extract_abstract(text)

    if not abstract:
        return ValidationCheck(
            name='Abstract',
            passed=False,
            message='Abstract not found',
            severity='error'
        )

    word_count = len(abstract.split())

    passed = True
    messages = [f"Abstract word count: {word_count}"]

    if min_words and word_count < min_words:
        passed = False
        messages.append(f"(minimum: {min_words})")
    if max_words and word_count > max_words:
        passed = False
        messages.append(f"(maximum: {max_words})")

    return ValidationCheck(
        name='Abstract Length',
        passed=passed,
        message=' '.join(messages),
        severity='error',
        actual_value=str(word_count),
        expected_value=f"{min_words}-{max_words}" if min_words and max_words else None
    )


def check_required_sections(
    manuscript_path: Path,
    required_sections: list[str]
) -> list[ValidationCheck]:
    """Check for required sections."""
    text = manuscript_path.read_text()
    checks = []

    for section in required_sections:
        # Look for section header
        pattern = rf'#+\s*{re.escape(section)}'
        found = bool(re.search(pattern, text, re.IGNORECASE))

        checks.append(ValidationCheck(
            name=f'Section: {section}',
            passed=found,
            message=f"'{section}' section {'found' if found else 'not found'}",
            severity='warning'
        ))

    return checks


def check_figures(
    manuscript_dir: Path,
    required_formats: Optional[list[str]] = None,
    min_dpi: Optional[int] = None
) -> list[ValidationCheck]:
    """Check figure files."""
    checks = []
    fig_dir = manuscript_dir / 'figures'

    if not fig_dir.exists():
        return [ValidationCheck(
            name='Figures Directory',
            passed=False,
            message='figures/ directory not found',
            severity='warning'
        )]

    figures = list(fig_dir.glob('*.*'))

    # Check count
    checks.append(ValidationCheck(
        name='Figure Count',
        passed=len(figures) > 0,
        message=f"Found {len(figures)} figure files",
        severity='info'
    ))

    # Check formats
    if required_formats:
        for fig in figures:
            format_ok = fig.suffix.lower().lstrip('.') in [f.lower() for f in required_formats]
            if not format_ok:
                checks.append(ValidationCheck(
                    name=f'Figure Format: {fig.name}',
                    passed=False,
                    message=f"Format {fig.suffix} not in allowed formats: {required_formats}",
                    severity='warning'
                ))

    return checks


def check_references(manuscript_path: Path) -> ValidationCheck:
    """Check for references/bibliography."""
    text = manuscript_path.read_text()

    # Look for bibliography in YAML
    has_bib = 'bibliography:' in text

    # Look for citations
    citations = re.findall(r'@[\w-]+', text)

    if has_bib and citations:
        message = f"Bibliography configured, {len(citations)} citations found"
        passed = True
    elif has_bib:
        message = "Bibliography configured but no citations found"
        passed = True
    elif citations:
        message = f"{len(citations)} citations found but no bibliography configured"
        passed = False
    else:
        message = "No bibliography or citations found"
        passed = False

    return ValidationCheck(
        name='References',
        passed=passed,
        message=message,
        severity='warning'
    )


# ============================================================
# MAIN VALIDATION FUNCTION
# ============================================================

def validate(
    journal: str = 'jeem',
    report: bool = False,
    manuscript_path: Optional[Path] = None
) -> ManuscriptValidation:
    """
    Validate manuscript against journal requirements.

    Parameters
    ----------
    journal : str
        Target journal name
    report : bool
        Generate markdown report
    manuscript_path : Path, optional
        Path to manuscript (default: manuscript_quarto/)

    Returns
    -------
    ManuscriptValidation
        Validation results
    """
    print("=" * 60)
    print("Stage 06: Manuscript Validation")
    print("=" * 60)

    # Setup paths
    project_root = get_project_root()
    manuscript_dir = manuscript_path or (project_root / MANUSCRIPT_DIR)
    main_file = manuscript_dir / MAIN_FILE
    diag_dir = get_data_dir('diagnostics')

    print(f"\n  Target journal: {journal}")
    print(f"  Manuscript: {main_file}")

    # Load journal config
    try:
        config = load_config(journal)
        print(f"  Config loaded: {journal}.yml")
    except FileNotFoundError:
        print(f"  WARNING: Journal config not found, using defaults")
        config = {}

    # Check manuscript exists
    if not main_file.exists():
        print(f"\n  ERROR: Manuscript not found: {main_file}")
        return ManuscriptValidation(journal=journal)

    validation = ManuscriptValidation(journal=journal)

    # Run checks
    print("\n  Running validation checks...")

    # Word count
    max_words = config.get('text_limits', {}).get('word_limit')
    validation.checks.append(check_word_count(main_file, max_words))

    # Abstract
    abstract_config = config.get('abstract', {})
    validation.checks.append(check_abstract_length(
        main_file,
        min_words=abstract_config.get('min_words'),
        max_words=abstract_config.get('max_words', 250)
    ))

    # Required sections
    required_sections = ['Introduction', 'Methods', 'Results', 'Discussion', 'Conclusion']
    validation.checks.extend(check_required_sections(main_file, required_sections))

    # Figures
    artwork_config = config.get('artwork', {})
    validation.checks.extend(check_figures(
        manuscript_dir,
        required_formats=artwork_config.get('formats', {}).get('raster_acceptable')
    ))

    # References
    validation.checks.append(check_references(main_file))

    # Print results
    print(validation.format())

    # Save report
    if report:
        ensure_dir(diag_dir)
        report_path = diag_dir / 'submission_validation.md'
        report_path.write_text(validation.to_markdown())
        print(f"\n  Report saved: {report_path}")

    # Generate QA report
    metrics = QAMetrics()
    metrics.add('journal', journal)
    metrics.add('n_checks', len(validation.checks))
    n_passed = sum(1 for c in validation.checks if c.passed)
    n_failed = len(validation.checks) - n_passed
    metrics.add('n_passed', n_passed)
    metrics.add('n_failed', n_failed)
    metrics.add_pct('pass_rate', (n_passed / len(validation.checks) * 100) if validation.checks else 0)
    generate_qa_report('s06_manuscript', metrics)

    print("\n" + "=" * 60)
    print("Stage 06 complete.")
    print("=" * 60)

    return validation


if __name__ == '__main__':
    validate()
