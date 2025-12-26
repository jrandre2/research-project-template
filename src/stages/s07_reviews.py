#!/usr/bin/env python3
"""
Stage 07: Review Management

Purpose: Manage synthetic peer review cycles for manuscript development.

Commands
--------
status : Display current review cycle status
new    : Initialize a new review cycle with discipline-specific template
archive: Archive current cycle and reset for new one
verify : Run verification checklist
report : Generate summary report of all review cycles

Usage
-----
    python src/pipeline.py review_status
    python src/pipeline.py review_new --discipline economics
    python src/pipeline.py review_archive
    python src/pipeline.py review_verify
    python src/pipeline.py review_report
"""
from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOC_DIR = PROJECT_ROOT / 'doc'
REVIEWS_DIR = DOC_DIR / 'reviews'
ARCHIVE_DIR = REVIEWS_DIR / 'archive'
MANUSCRIPT_DIR = PROJECT_ROOT / 'manuscript_quarto'
TRACKER_FILE = MANUSCRIPT_DIR / 'REVISION_TRACKER.md'
CHECKLIST_FILE = DOC_DIR / 'MANUSCRIPT_REVISION_CHECKLIST.md'
REVIEWS_INDEX = REVIEWS_DIR / 'README.md'

# Discipline templates
DISCIPLINE_PROMPTS = {
    'economics': '''Act as a critical peer reviewer for a top economics journal (AER, QJE, JEEM, Econometrica).

Review the following manuscript for:
- **Identification strategy**: Is the causal claim credible? What are threats to identification?
- **Econometric methods**: Are standard errors correctly specified? Is clustering appropriate?
- **Pre-trend tests**: Are parallel trends assumptions tested and satisfied?
- **Robustness checks**: Are alternative specifications explored?
- **Data quality**: Are there measurement concerns? Selection issues?
- **External validity**: How generalizable are findings?

Be specific about threats to identification and suggest diagnostic tests.
Format your response with numbered major and minor comments.''',

    'engineering': '''Act as a technical reviewer for a top engineering journal (IEEE, ASME, Nature Engineering).

Review the following manuscript for:
- **Methodology rigor**: Is the approach well-justified and correctly implemented?
- **Reproducibility**: Can results be replicated from the description?
- **Benchmark comparisons**: Are appropriate baselines used?
- **Computational efficiency**: Is the approach scalable?
- **Technical accuracy**: Are equations, algorithms, and implementations correct?
- **Validation**: Is the experimental design sufficient to support claims?

Identify gaps in experimental design and validation.
Format your response with numbered major and minor comments.''',

    'social_sciences': '''Act as a reviewer for a leading social science journal (ASR, APSR, JCR, AJS).

Review the following manuscript for:
- **Theoretical contribution**: Does this advance theory meaningfully?
- **Generalizability**: How do findings extend beyond this sample/context?
- **Construct validity**: Are concepts measured appropriately?
- **Sampling strategy**: Is the sample representative for the claims made?
- **Measurement**: Are measures reliable and valid?
- **Ethical considerations**: Are there concerns about harm or consent?
- **Policy implications**: Are practical implications overstated?

Assess both methodological and theoretical contributions.
Format your response with numbered major and minor comments.''',

    'general': '''Act as a critical peer reviewer for an academic journal.

Review the following manuscript for:
- **Research question clarity**: Is the central question well-defined?
- **Methodology appropriateness**: Is the method suited to the question?
- **Evidence quality**: Do the data support the claims?
- **Logical flow**: Is the argument coherent and well-structured?
- **Literature contribution**: What does this add to existing knowledge?
- **Presentation**: Is the writing clear and accessible?

Be constructive and specific about improvements needed.
Format your response with numbered major and minor comments.'''
}


def status():
    """Display current review cycle status from REVISION_TRACKER.md."""
    print("Review Status")
    print("=" * 50)

    if not TRACKER_FILE.exists():
        print("\nNo active review cycle.")
        print(f"Start one with: python src/pipeline.py review_new --discipline <name>")
        return

    content = TRACKER_FILE.read_text()

    # Parse summary statistics
    print("\nCurrent Tracker:", TRACKER_FILE)

    # Extract review number and discipline
    review_match = re.search(r'\*\*Review\*\*:\s*#?(\w+)', content)
    discipline_match = re.search(r'\*\*Discipline\*\*:\s*(\w+)', content)

    if review_match:
        print(f"Review: #{review_match.group(1)}")
    if discipline_match:
        print(f"Discipline: {discipline_match.group(1)}")

    # Extract summary table
    summary_match = re.search(
        r'\|\s*Category\s*\|.*?\n\|[-\s|]+\n((?:\|.*?\n)+)',
        content,
        re.MULTILINE
    )

    if summary_match:
        print("\nSummary:")
        rows = summary_match.group(1).strip().split('\n')
        for row in rows:
            cells = [c.strip() for c in row.split('|') if c.strip()]
            if len(cells) >= 5:
                print(f"  {cells[0]}: {cells[1]} total, {cells[2]} addressed, "
                      f"{cells[3]} beyond scope, {cells[4]} pending")

    # Count verification items
    checked = len(re.findall(r'- \[x\]', content, re.IGNORECASE))
    unchecked = len(re.findall(r'- \[ \]', content))
    total = checked + unchecked

    if total > 0:
        print(f"\nVerification: {checked}/{total} items complete")

    print("\n" + "=" * 50)


def new_cycle(discipline: str = 'general'):
    """Initialize a new review cycle with discipline-specific template."""
    if discipline not in DISCIPLINE_PROMPTS:
        print(f"ERROR: Unknown discipline '{discipline}'")
        print(f"Available: {', '.join(DISCIPLINE_PROMPTS.keys())}")
        return

    print(f"Initializing new review cycle (discipline: {discipline})")
    print("=" * 50)

    # Determine review number
    review_num = 1
    if ARCHIVE_DIR.exists():
        existing = list(ARCHIVE_DIR.glob('review_*.md'))
        if existing:
            nums = [int(re.search(r'review_(\d+)', f.name).group(1))
                    for f in existing if re.search(r'review_(\d+)', f.name)]
            if nums:
                review_num = max(nums) + 1

    # Check if there's an active review
    if TRACKER_FILE.exists():
        content = TRACKER_FILE.read_text()
        if 'PENDING' in content.upper() or re.search(r'\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*[1-9]', content):
            print("\nWARNING: Active review has pending items.")
            print("Archive current review first with: python src/pipeline.py review_archive")
            return

    # Create new tracker from template
    today = datetime.now().strftime('%Y-%m-%d')
    template = f'''# Revision Tracker: Response to Synthetic Review

**Document**: [Project Name] manuscript
**Review**: #{review_num}
**Discipline**: {discipline}
**Last Updated**: {today}

---

## Summary Statistics

| Category | Total | Addressed | Beyond Scope | Pending |
|----------|-------|-----------|--------------|---------|
| Major Comments | 0 | 0 | 0 | 0 |
| Minor Comments | 0 | 0 | 0 | 0 |

---

## Prompt Used

```
{DISCIPLINE_PROMPTS[discipline]}
```

---

## Major Comments

### Comment 1: [Title]

**Status**: [VALID - ACTION NEEDED | ALREADY ADDRESSED | BEYOND SCOPE | INVALID]

**Reviewer's Concern**:
> [Paste reviewer comment here]

**Validity Assessment**: [VALID | PARTIALLY VALID | INVALID]

[Explain assessment]

**Response**:

[Describe response]

**Files Modified**:
- [Files]

---

## Minor Comments

[Add minor comments as needed]

---

## Verification Checklist

- [ ] All VALID - ACTION NEEDED items addressed
- [ ] All code runs without errors
- [ ] Manuscript text updated
- [ ] Tables/figures reflect changes
- [ ] Quarto renders without errors
- [ ] Changes committed to git
- [ ] MANUSCRIPT_REVISION_CHECKLIST.md updated

---

*Review generated: {today}*
*Last updated: {today}*
'''

    TRACKER_FILE.write_text(template)
    print(f"\nCreated: {TRACKER_FILE}")
    print(f"\nReview #{review_num} initialized with {discipline} discipline.")
    print("\nNext steps:")
    print("1. Generate a synthetic review using the prompt above")
    print("2. Paste reviewer comments into the tracker")
    print("3. Triage each comment with a status classification")
    print("4. Implement changes and update tracker")
    print("5. Run: python src/pipeline.py review_verify")


def archive():
    """Archive current review cycle and reset for new one."""
    print("Archiving Review Cycle")
    print("=" * 50)

    if not TRACKER_FILE.exists():
        print("No active review to archive.")
        return

    # Ensure archive directory exists
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Determine archive filename
    content = TRACKER_FILE.read_text()
    review_match = re.search(r'\*\*Review\*\*:\s*#?(\d+)', content)

    if review_match:
        review_num = review_match.group(1)
    else:
        # Find next available number
        existing = list(ARCHIVE_DIR.glob('review_*.md'))
        nums = [int(re.search(r'review_(\d+)', f.name).group(1))
                for f in existing if re.search(r'review_(\d+)', f.name)]
        review_num = max(nums) + 1 if nums else 1

    archive_file = ARCHIVE_DIR / f'review_{review_num:02d}.md'

    # Copy to archive
    shutil.copy(TRACKER_FILE, archive_file)
    print(f"Archived to: {archive_file}")

    # Reset tracker
    TRACKER_FILE.unlink()
    print(f"Removed: {TRACKER_FILE}")

    print(f"\nReview #{review_num} archived successfully.")
    print("Start new review with: python src/pipeline.py review_new --discipline <name>")


def verify():
    """Run verification checklist for current review cycle."""
    print("Verification Checklist")
    print("=" * 50)

    if not TRACKER_FILE.exists():
        print("No active review to verify.")
        return

    content = TRACKER_FILE.read_text()

    # Find all checklist items
    checked = re.findall(r'- \[x\]\s+(.+)', content, re.IGNORECASE)
    unchecked = re.findall(r'- \[ \]\s+(.+)', content)

    print(f"\nCompleted ({len(checked)}):")
    for item in checked:
        print(f"  [x] {item}")

    print(f"\nPending ({len(unchecked)}):")
    for item in unchecked:
        print(f"  [ ] {item}")

    total = len(checked) + len(unchecked)
    if total > 0:
        pct = (len(checked) / total) * 100
        print(f"\nProgress: {len(checked)}/{total} ({pct:.0f}%)")

        if len(unchecked) == 0:
            print("\nAll verification items complete!")
            print("Ready to archive: python src/pipeline.py review_archive")
        else:
            print(f"\n{len(unchecked)} items remaining before archive.")


def report():
    """Generate summary report of all review cycles."""
    print("Review Cycles Report")
    print("=" * 50)

    # Check for archived reviews
    if not ARCHIVE_DIR.exists():
        archived = []
    else:
        archived = sorted(ARCHIVE_DIR.glob('review_*.md'))

    # Check for active review
    active = TRACKER_FILE.exists()

    total_cycles = len(archived) + (1 if active else 0)
    print(f"\nTotal review cycles: {total_cycles}")
    print(f"  Archived: {len(archived)}")
    print(f"  Active: {'Yes' if active else 'No'}")

    if archived:
        print("\nArchived Reviews:")
        print("-" * 40)
        for f in archived:
            content = f.read_text()
            discipline_match = re.search(r'\*\*Discipline\*\*:\s*(\w+)', content)
            date_match = re.search(r'\*Review generated:\s*(\d{4}-\d{2}-\d{2})', content)

            discipline = discipline_match.group(1) if discipline_match else 'unknown'
            date = date_match.group(1) if date_match else 'unknown'

            # Count comments
            major = len(re.findall(r'### Comment \d+:', content))
            minor = len(re.findall(r'### Minor \d+:', content))

            print(f"  {f.name}: {discipline}, {date}, {major} major, {minor} minor")

    if active:
        print("\nActive Review:")
        print("-" * 40)
        status()


def main(action: str = 'status', discipline: str = 'general'):
    """Main entry point for review management."""
    if action == 'status':
        status()
    elif action == 'new':
        new_cycle(discipline)
    elif action == 'archive':
        archive()
    elif action == 'verify':
        verify()
    elif action == 'report':
        report()
    else:
        print(f"Unknown action: {action}")
        print("Available: status, new, archive, verify, report")


if __name__ == '__main__':
    main()
