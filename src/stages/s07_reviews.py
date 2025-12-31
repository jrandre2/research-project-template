#!/usr/bin/env python3
"""
Stage 07: Review Management

Purpose: Manage peer review cycles for manuscript development.

Supports both synthetic (AI-generated) and actual (journal) peer reviews,
with multiple manuscripts and independent review tracking.

Commands
--------
status : Display current review cycle status
new    : Initialize a new review cycle (synthetic or actual)
archive: Archive current cycle and reset for new one
verify : Run verification checklist with journal compliance checks
report : Generate summary report of all review cycles

Usage
-----
    # Synthetic review (default)
    python src/pipeline.py review_new --manuscript main --focus economics

    # Actual journal review
    python src/pipeline.py review_new --manuscript main --actual \\
        --journal "JEEM" --round "R&R1" --reviewers R1 R2

    python src/pipeline.py review_status --manuscript main
    python src/pipeline.py review_archive --manuscript main
    python src/pipeline.py review_verify --manuscript main
    python src/pipeline.py review_report
"""
from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from stages._qa_utils import generate_qa_report, QAMetrics
from stages._review_models import (
    ReviewMetadata,
    parse_frontmatter,
    add_frontmatter,
)

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOC_DIR = PROJECT_ROOT / 'doc'
REVIEWS_DIR = DOC_DIR / 'reviews'
CHECKLIST_FILE = DOC_DIR / 'MANUSCRIPT_REVISION_CHECKLIST.md'
REVIEWS_INDEX = REVIEWS_DIR / 'README.md'


# =============================================================================
# MANUSCRIPT CONFIGURATION
# =============================================================================

MANUSCRIPTS = {
    'main': {
        'name': 'Main Manuscript',
        'title': '[Project Title]',
        'dir': PROJECT_ROOT / 'manuscript_quarto',
        'reviews_dir': REVIEWS_DIR / 'main',
        'archive_dir': REVIEWS_DIR / 'main' / 'archive',
        'description': 'Primary submission manuscript',
    },
    # Additional manuscripts can be configured here
    # 'short': {
    #     'name': 'Short Communication',
    #     'title': '[Short Title]',
    #     'dir': PROJECT_ROOT / 'manuscript_short',
    #     'reviews_dir': REVIEWS_DIR / 'short',
    #     'archive_dir': REVIEWS_DIR / 'short' / 'archive',
    #     'description': 'Brief communication version',
    # },
}

DEFAULT_MANUSCRIPT = 'main'


def get_manuscript_paths(manuscript: str = None) -> dict:
    """
    Get paths for a specific manuscript.

    Parameters
    ----------
    manuscript : str, optional
        Manuscript name. Defaults to DEFAULT_MANUSCRIPT.

    Returns
    -------
    dict
        Dictionary with manuscript paths and metadata

    Raises
    ------
    ValueError
        If manuscript name is not found
    """
    if manuscript is None:
        manuscript = DEFAULT_MANUSCRIPT

    if manuscript not in MANUSCRIPTS:
        available = ', '.join(MANUSCRIPTS.keys())
        raise ValueError(f"Unknown manuscript '{manuscript}'. Available: {available}")

    config = MANUSCRIPTS[manuscript]
    return {
        'manuscript_dir': config['dir'],
        'tracker_file': config['dir'] / 'REVISION_TRACKER.md',
        'reviews_dir': config['reviews_dir'],
        'archive_dir': config['archive_dir'],
        'name': config['name'],
        'title': config['title'],
    }


# =============================================================================
# FOCUS PROMPTS (Generic Templates)
# =============================================================================

FOCUS_PROMPTS = {
    # Discipline-based prompts (original)
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
Format your response with numbered major and minor comments.''',

    # Focus-specific prompts (new)
    'methods': '''Act as a methodologist reviewing an academic manuscript.

Focus exclusively on the **statistical and methodological approach**:

**1. Model Specification**
- Is the statistical model appropriate for the research question?
- Are assumptions tested and satisfied?
- Is the functional form justified?

**2. Estimation**
- Are standard errors correctly specified?
- Is clustering/stratification handled appropriately?
- Are confidence intervals properly constructed?

**3. Sample & Power**
- Is the sample size adequate for the analysis?
- Are effect sizes practically meaningful?
- Is there evidence of adequate statistical power?

**4. Diagnostics**
- Are residuals examined?
- Are influential observations identified?
- Are model fit statistics reported?

**5. Robustness**
- Are alternative specifications tested?
- Are results stable across subsamples?
- Are sensitivity analyses performed?

Format your response with numbered major and minor methodological concerns.''',

    'policy': '''Act as a practitioner reviewing an academic manuscript for practical relevance.

Evaluate this manuscript for **practical applicability and actionability**:

**1. Practitioner Relevance**
- Would practitioners find this useful?
- Are recommendations specific enough to implement?
- Is the research question important to the field?

**2. Actionable Insights**
- What specific actions should practitioners take based on findings?
- Are recommendations feasible given real-world constraints?
- Are implementation barriers discussed?

**3. Context & Generalization**
- Do findings apply to the practitioner's context?
- Are boundary conditions clearly stated?
- What contexts might require different approaches?

**4. Trade-offs & Limitations**
- Are costs and benefits honestly assessed?
- Are unintended consequences considered?
- Are limitations clearly communicated?

**5. Communication**
- Is the executive summary accessible to non-academics?
- Are key takeaways clearly stated?
- Would a busy practitioner understand the implications?

Format your response with numbered major and minor practical concerns.''',

    'clarity': '''Act as an editor reviewing an academic manuscript for clarity and accessibility.

Focus on **writing quality and presentation**:

**1. Structure & Organization**
- Is the paper logically organized?
- Do sections flow naturally?
- Are transitions between ideas smooth?

**2. Clarity of Expression**
- Is the writing clear and concise?
- Are complex ideas explained accessibly?
- Is jargon minimized or explained?

**3. Abstract & Introduction**
- Does the abstract accurately summarize the paper?
- Is the research question clearly stated?
- Is the contribution immediately apparent?

**4. Tables & Figures**
- Are visualizations clear and informative?
- Can tables be understood without reading the text?
- Are captions complete and helpful?

**5. Discussion & Conclusion**
- Are limitations honestly acknowledged?
- Are findings interpreted appropriately (not overclaimed)?
- Is the conclusion memorable and impactful?

Format your response with numbered major and minor presentation concerns.''',
}

# Backward compatibility: alias discipline to focus
DISCIPLINE_PROMPTS = FOCUS_PROMPTS


# =============================================================================
# GIT UTILITIES
# =============================================================================

def get_current_commit() -> Optional[str]:
    """Get the current git commit SHA."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]  # Short SHA
    except Exception:
        pass
    return None


def create_review_tag(
    manuscript: str,
    cycle: int,
    status: str = 'complete',
) -> Optional[str]:
    """
    Create a git tag for a review cycle.

    Parameters
    ----------
    manuscript : str
        Manuscript name
    cycle : int
        Review cycle number
    status : str
        Tag status suffix (e.g., 'complete', 'start')

    Returns
    -------
    str or None
        Tag name if created successfully, None otherwise
    """
    tag_name = f"review-{manuscript}-{cycle:02d}-{status}"

    try:
        result = subprocess.run(
            ['git', 'tag', tag_name],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            return tag_name
    except Exception:
        pass
    return None


def get_commits_since(commit_sha: str) -> list[str]:
    """Get list of commits since a given SHA."""
    try:
        result = subprocess.run(
            ['git', 'log', f'{commit_sha}..HEAD', '--format=%h'],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            return [c for c in result.stdout.strip().split('\n') if c]
    except Exception:
        pass
    return []


# =============================================================================
# REVIEW MANAGEMENT FUNCTIONS
# =============================================================================

def status(manuscript: str = None):
    """Display current review cycle status from REVISION_TRACKER.md."""
    paths = get_manuscript_paths(manuscript)
    tracker_file = paths['tracker_file']
    manuscript_key = manuscript or DEFAULT_MANUSCRIPT

    print(f"Review Status: {paths['name']}")
    print("=" * 50)

    if not tracker_file.exists():
        print("\nNo active review cycle.")
        print(f"Start one with: python src/pipeline.py review_new --manuscript {manuscript_key} --focus <name>")
        return

    content = tracker_file.read_text()

    # Try to parse YAML frontmatter for new format
    frontmatter, body = parse_frontmatter(content)

    print(f"\nCurrent Tracker: {tracker_file}")

    if frontmatter:
        # New format with YAML frontmatter
        print(f"Review: #{frontmatter.get('cycle_number', '?')}")
        print(f"Type: {frontmatter.get('source_type', 'synthetic').upper()}")

        if frontmatter.get('source_type') == 'actual':
            if frontmatter.get('journal'):
                print(f"Journal: {frontmatter.get('journal')}")
            if frontmatter.get('submission_round'):
                print(f"Round: {frontmatter.get('submission_round')}")
            if frontmatter.get('reviewer_ids'):
                print(f"Reviewers: {', '.join(frontmatter.get('reviewer_ids', []))}")
        else:
            if frontmatter.get('focus'):
                print(f"Focus: {frontmatter.get('focus')}")

        if frontmatter.get('start_commit'):
            print(f"Start commit: {frontmatter.get('start_commit')}")

        content = body  # Use body for rest of parsing
    else:
        # Legacy format - parse from markdown
        review_match = re.search(r'\*\*Review\*\*:\s*#?(\w+)', content)
        focus_match = re.search(r'\*\*(?:Discipline|Focus)\*\*:\s*(\w+)', content)

        if review_match:
            print(f"Review: #{review_match.group(1)}")
        if focus_match:
            print(f"Focus: {focus_match.group(1)}")

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


def new_cycle(
    manuscript: str = None,
    focus: str = 'general',
    # New parameters for actual reviews
    source_type: Literal['synthetic', 'actual'] = 'synthetic',
    journal: str = None,
    submission_round: str = None,
    decision: str = None,
    reviewer_ids: list[str] = None,
):
    """
    Initialize a new review cycle.

    Parameters
    ----------
    manuscript : str, optional
        Manuscript name. Defaults to DEFAULT_MANUSCRIPT.
    focus : str
        Focus area for synthetic reviews (e.g., 'economics', 'methods')
    source_type : str
        Either 'synthetic' or 'actual'
    journal : str, optional
        Journal name for actual reviews
    submission_round : str, optional
        Submission round (e.g., 'initial', 'R&R1')
    decision : str, optional
        Editor decision (e.g., 'major_revision', 'minor_revision')
    reviewer_ids : list[str], optional
        Reviewer identifiers (e.g., ['R1', 'R2'])
    """
    paths = get_manuscript_paths(manuscript)
    manuscript_key = manuscript or DEFAULT_MANUSCRIPT

    # Validate focus for synthetic reviews
    if source_type == 'synthetic' and focus not in FOCUS_PROMPTS:
        print(f"ERROR: Unknown focus '{focus}'")
        print(f"Available: {', '.join(FOCUS_PROMPTS.keys())}")
        return

    # Print header
    print(f"Initializing new review cycle")
    print(f"  Manuscript: {paths['name']}")
    print(f"  Type: {source_type.upper()}")
    if source_type == 'synthetic':
        print(f"  Focus: {focus}")
    else:
        if journal:
            print(f"  Journal: {journal}")
        if submission_round:
            print(f"  Round: {submission_round}")
        if reviewer_ids:
            print(f"  Reviewers: {', '.join(reviewer_ids)}")
    print("=" * 50)

    # Ensure directories exist
    paths['reviews_dir'].mkdir(parents=True, exist_ok=True)
    paths['archive_dir'].mkdir(parents=True, exist_ok=True)

    # Determine review number
    review_num = 1
    if paths['archive_dir'].exists():
        existing = list(paths['archive_dir'].glob('review_*.md'))
        if existing:
            nums = [int(re.search(r'review_(\d+)', f.name).group(1))
                    for f in existing if re.search(r'review_(\d+)', f.name)]
            if nums:
                review_num = max(nums) + 1

    # Check if there's an active review
    tracker_file = paths['tracker_file']
    if tracker_file.exists():
        content = tracker_file.read_text()
        if 'PENDING' in content.upper() or re.search(r'\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*[1-9]', content):
            print("\nWARNING: Active review has pending items.")
            print(f"Archive current review first with: python src/pipeline.py review_archive --manuscript {manuscript_key}")
            return

    # Create metadata
    metadata = ReviewMetadata(
        manuscript=manuscript_key,
        cycle_number=review_num,
        source_type=source_type,
        focus=focus if source_type == 'synthetic' else None,
        journal=journal,
        submission_round=submission_round,
        decision=decision,
        reviewer_ids=reviewer_ids or [],
        start_commit=get_current_commit(),
        tracker_file=tracker_file,
    )

    # Create new tracker from template
    today = datetime.now().strftime('%Y-%m-%d')

    if source_type == 'synthetic':
        # Synthetic review template
        title = "Revision Tracker: Response to Synthetic Review"
        header_info = f'''**Document**: {paths['title']}
**Review**: #{review_num}
**Type**: Synthetic
**Focus**: {focus}
**Last Updated**: {today}'''
        prompt_section = f'''
---

## Prompt Used

```
{FOCUS_PROMPTS[focus]}
```
'''
    else:
        # Actual review template
        title = "Revision Tracker: Response to Reviewer Comments"
        header_parts = [
            f"**Document**: {paths['title']}",
            f"**Review**: #{review_num}",
            "**Type**: Actual (Journal Review)",
        ]
        if journal:
            header_parts.append(f"**Journal**: {journal}")
        if submission_round:
            header_parts.append(f"**Round**: {submission_round}")
        if decision:
            header_parts.append(f"**Decision**: {decision}")
        if reviewer_ids:
            header_parts.append(f"**Reviewers**: {', '.join(reviewer_ids)}")
        header_parts.append(f"**Last Updated**: {today}")
        header_info = '\n'.join(header_parts)
        prompt_section = ''  # No prompt for actual reviews

    # Build reviewer comment sections for actual reviews
    if source_type == 'actual' and reviewer_ids:
        reviewer_sections = []
        for rid in reviewer_ids:
            reviewer_sections.append(f'''
## {rid} Comments

### {rid} Major 1: [Title]

**Status**: [VALID - ACTION NEEDED | ALREADY ADDRESSED | BEYOND SCOPE | INVALID]

**Reviewer's Comment**:
> [Paste {rid}'s comment here]

**Validity Assessment**: [VALID | PARTIALLY VALID | INVALID]

[Explain assessment]

**Response**:

[Describe response]

**Files Modified**:
- [Files]

---
''')
        comments_section = '\n'.join(reviewer_sections)
    else:
        comments_section = '''
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
'''

    template = f'''# {title}

{header_info}
{prompt_section}
---

## Summary Statistics

| Category | Total | Addressed | Beyond Scope | Pending |
|----------|-------|-----------|--------------|---------|
| Major Comments | 0 | 0 | 0 | 0 |
| Minor Comments | 0 | 0 | 0 | 0 |
{comments_section}
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

    # Add YAML frontmatter with metadata
    final_content = add_frontmatter(template, metadata)
    tracker_file.write_text(final_content)

    print(f"\nCreated: {tracker_file}")
    print(f"\nReview #{review_num} initialized ({source_type}).")

    if metadata.start_commit:
        print(f"Start commit: {metadata.start_commit}")

    print("\nNext steps:")
    if source_type == 'synthetic':
        print("1. Generate a synthetic review using the prompt above")
        print("2. Paste reviewer comments into the tracker")
    else:
        print("1. Paste the reviewer comments from the decision letter")
        print("2. Organize by reviewer (R1, R2, etc.)")
    print("3. Triage each comment with a status classification")
    print("4. Implement changes and update tracker")
    print(f"5. Run: python src/pipeline.py review_verify --manuscript {manuscript_key}")


def archive(
    manuscript: str = None,
    create_tag: bool = True,
    tag_name: str = None,
):
    """
    Archive current review cycle and reset for new one.

    Parameters
    ----------
    manuscript : str, optional
        Manuscript name. Defaults to DEFAULT_MANUSCRIPT.
    create_tag : bool
        Whether to create a git tag for this archive
    tag_name : str, optional
        Custom tag name. If not provided, uses default format.
    """
    paths = get_manuscript_paths(manuscript)
    manuscript_key = manuscript or DEFAULT_MANUSCRIPT

    print(f"Archiving Review Cycle: {paths['name']}")
    print("=" * 50)

    tracker_file = paths['tracker_file']
    if not tracker_file.exists():
        print("No active review to archive.")
        return

    # Ensure archive directory exists
    paths['archive_dir'].mkdir(parents=True, exist_ok=True)

    content = tracker_file.read_text()

    # Try to parse YAML frontmatter for new format
    frontmatter, body = parse_frontmatter(content)

    if frontmatter:
        review_num = frontmatter.get('cycle_number', 1)
        source_type = frontmatter.get('source_type', 'synthetic')
        start_commit = frontmatter.get('start_commit')
    else:
        # Legacy format
        review_match = re.search(r'\*\*Review\*\*:\s*#?(\d+)', content)
        review_num = int(review_match.group(1)) if review_match else 1
        source_type = 'synthetic'
        start_commit = None

    # If no review_num found, find next available
    if not review_num:
        existing = list(paths['archive_dir'].glob('review_*.md'))
        nums = [int(re.search(r'review_(\d+)', f.name).group(1))
                for f in existing if re.search(r'review_(\d+)', f.name)]
        review_num = max(nums) + 1 if nums else 1

    archive_file = paths['archive_dir'] / f'review_{int(review_num):02d}.md'

    # Update metadata with end commit before archiving
    end_commit = get_current_commit()
    if frontmatter and end_commit:
        # Update the frontmatter with end commit
        frontmatter['end_commit'] = end_commit
        frontmatter['archived_at'] = datetime.now().isoformat()
        if start_commit:
            frontmatter['response_commits'] = get_commits_since(start_commit)

        # Rebuild content with updated frontmatter
        import yaml
        updated_frontmatter = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        content = f"---\n{updated_frontmatter}---\n\n{body}"

    # Copy to archive
    with open(archive_file, 'w') as f:
        f.write(content)
    print(f"Archived to: {archive_file}")

    # Create git tag
    if create_tag:
        actual_tag = tag_name or f"review-{manuscript_key}-{review_num:02d}-complete"
        created = create_review_tag(manuscript_key, review_num, 'complete')
        if created:
            print(f"Created git tag: {created}")
        else:
            print(f"Note: Could not create git tag (may already exist or not in git repo)")

    # Show commits in this review cycle
    if start_commit and end_commit:
        commits = get_commits_since(start_commit)
        if commits:
            print(f"\nCommits in this review cycle: {len(commits)}")

    # Reset tracker
    tracker_file.unlink()
    print(f"Removed: {tracker_file}")

    print(f"\nReview #{review_num} archived successfully ({source_type}).")
    print(f"Start new review with: python src/pipeline.py review_new --manuscript {manuscript_key} --focus <name>")


def verify(manuscript: str = None):
    """Run verification checklist for current review cycle."""
    paths = get_manuscript_paths(manuscript)

    print(f"Verification Checklist: {paths['name']}")
    print("=" * 50)

    tracker_file = paths['tracker_file']
    if not tracker_file.exists():
        print("No active review to verify.")
        return

    content = tracker_file.read_text()

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
            print(f"Ready to archive: python src/pipeline.py review_archive --manuscript {manuscript or DEFAULT_MANUSCRIPT}")
        else:
            print(f"\n{len(unchecked)} items remaining before archive.")

    # Journal compliance checks
    print("\n" + "-" * 50)
    print("Journal Compliance Checks")
    print("-" * 50)

    manuscript_dir = paths['manuscript_dir']
    if manuscript_dir.exists():
        # Word count check
        word_count = count_manuscript_words(manuscript_dir)
        if word_count > 0:
            print(f"\n  Word count: {word_count:,}")
            if word_count > 10000:
                print("    WARNING: May exceed typical journal limits")

        # Self-reference check
        self_refs = check_self_references(manuscript_dir)
        if self_refs:
            print(f"\n  Self-references found: {len(self_refs)}")
            for ref in self_refs[:3]:  # Show first 3
                print(f"    - {ref[:60]}...")
            if len(self_refs) > 3:
                print(f"    ... and {len(self_refs) - 3} more")
        else:
            print("\n  Self-references: None found (good)")

        # Abstract length
        abstract_words = check_abstract_length(manuscript_dir)
        if abstract_words > 0:
            print(f"\n  Abstract words: {abstract_words}")
            if abstract_words > 250:
                print("    WARNING: May exceed typical limits (150-250 words)")
    else:
        print(f"\n  Manuscript directory not found: {manuscript_dir}")

    # Generate QA report
    metrics = QAMetrics()
    metrics.add('manuscript', manuscript or DEFAULT_MANUSCRIPT)
    metrics.add('checklist_completed', len(checked))
    metrics.add('checklist_pending', len(unchecked))
    if total > 0:
        metrics.add_pct('checklist_progress', pct)
    generate_qa_report('s07_review_verify', metrics)

    print("\n" + "=" * 50)


def report():
    """Generate summary report of all review cycles across all manuscripts."""
    print("Review Cycles Report")
    print("=" * 50)

    total_cycles = 0
    total_archived = 0

    for ms_name, ms_config in MANUSCRIPTS.items():
        paths = get_manuscript_paths(ms_name)

        # Check for archived reviews
        if paths['archive_dir'].exists():
            archived = sorted(paths['archive_dir'].glob('review_*.md'))
        else:
            archived = []

        # Check for active review
        tracker_file = paths['tracker_file']
        active = tracker_file.exists()

        ms_total = len(archived) + (1 if active else 0)
        total_cycles += ms_total
        total_archived += len(archived)

        if ms_total > 0:
            print(f"\n{paths['name']}:")
            print("-" * 40)
            print(f"  Archived: {len(archived)}")
            print(f"  Active: {'Yes' if active else 'No'}")

            if archived:
                for f in archived:
                    content = f.read_text()
                    focus_match = re.search(r'\*\*(?:Discipline|Focus)\*\*:\s*(\w+)', content)
                    date_match = re.search(r'\*Review generated:\s*(\d{4}-\d{2}-\d{2})', content)

                    focus = focus_match.group(1) if focus_match else 'unknown'
                    date = date_match.group(1) if date_match else 'unknown'

                    # Count comments
                    major = len(re.findall(r'### Comment \d+:', content))
                    minor = len(re.findall(r'### Minor \d+:', content))

                    print(f"    {f.name}: {focus}, {date}, {major} major, {minor} minor")

    print("\n" + "=" * 50)
    print(f"Total across all manuscripts: {total_cycles} cycles ({total_archived} archived)")


# =============================================================================
# JOURNAL COMPLIANCE UTILITIES
# =============================================================================

def count_manuscript_words(manuscript_dir: Path) -> int:
    """Count words in manuscript QMD files, excluding code and YAML."""
    total_words = 0

    for qmd_file in manuscript_dir.glob('*.qmd'):
        try:
            content = qmd_file.read_text()

            # Remove YAML frontmatter
            content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

            # Remove code blocks
            content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)

            # Remove inline code
            content = re.sub(r'`[^`]+`', '', content)

            # Remove markdown links (keep text)
            content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)

            # Count words
            words = len(content.split())
            total_words += words
        except Exception:
            continue

    return total_words


def check_self_references(manuscript_dir: Path) -> list[str]:
    """Check for self-referential phrases like 'this study'."""
    patterns = [
        r'[Tt]his study',
        r'[Tt]his paper',
        r'[Tt]his research',
        r'[Oo]ur study',
        r'[Oo]ur paper',
        r'[Oo]ur research',
        r'[Ww]e find',
        r'[Ww]e show',
        r'[Ww]e demonstrate',
    ]

    matches = []
    for qmd_file in manuscript_dir.glob('*.qmd'):
        try:
            content = qmd_file.read_text()
            for pattern in patterns:
                for match in re.finditer(pattern, content):
                    # Get context around match
                    start = max(0, match.start() - 20)
                    end = min(len(content), match.end() + 40)
                    context = content[start:end].replace('\n', ' ')
                    matches.append(context.strip())
        except Exception:
            continue

    return matches


def check_abstract_length(manuscript_dir: Path) -> int:
    """Count words in the abstract."""
    main_file = manuscript_dir / 'index.qmd'
    if not main_file.exists():
        return 0

    try:
        content = main_file.read_text()

        # Find abstract in YAML frontmatter
        yaml_match = re.search(r'^---\n(.*?)\n---', content, flags=re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)
            abstract_match = re.search(r'abstract:\s*["|](.+?)["|]', yaml_content, flags=re.DOTALL)
            if abstract_match:
                abstract = abstract_match.group(1)
                return len(abstract.split())

        # Alternative: look for abstract section
        abstract_section = re.search(r'## Abstract\n+(.+?)(?:\n## |\Z)', content, flags=re.DOTALL)
        if abstract_section:
            return len(abstract_section.group(1).split())

    except Exception:
        pass

    return 0


# =============================================================================
# DIFF GENERATION
# =============================================================================

def get_manuscript_files(manuscript_dir: Path) -> dict[str, str]:
    """Get content of all manuscript QMD files."""
    files = {}
    for qmd_file in sorted(manuscript_dir.glob('*.qmd')):
        try:
            files[qmd_file.name] = qmd_file.read_text()
        except Exception:
            continue
    return files


def get_files_at_commit(manuscript_dir: Path, commit: str) -> dict[str, str]:
    """Get manuscript files at a specific git commit."""
    files = {}
    relative_dir = manuscript_dir.relative_to(PROJECT_ROOT)

    for qmd_file in manuscript_dir.glob('*.qmd'):
        relative_path = qmd_file.relative_to(PROJECT_ROOT)
        try:
            result = subprocess.run(
                ['git', 'show', f'{commit}:{relative_path}'],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            if result.returncode == 0:
                files[qmd_file.name] = result.stdout
        except Exception:
            continue

    return files


def generate_unified_diff(old_content: str, new_content: str, filename: str) -> str:
    """Generate a unified diff between two versions of a file."""
    import difflib

    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f'a/{filename}',
        tofile=f'b/{filename}',
        lineterm='',
    )
    return ''.join(diff)


def generate_markdown_diff(old_files: dict, new_files: dict) -> str:
    """Generate a markdown-formatted diff report."""
    output = []
    output.append("# Manuscript Changes\n")

    all_files = sorted(set(old_files.keys()) | set(new_files.keys()))

    for filename in all_files:
        old_content = old_files.get(filename, '')
        new_content = new_files.get(filename, '')

        if old_content == new_content:
            continue

        if not old_content:
            output.append(f"\n## {filename} (NEW FILE)\n")
            output.append("```\n")
            output.append(new_content[:500])
            if len(new_content) > 500:
                output.append("\n... (truncated)")
            output.append("\n```\n")
        elif not new_content:
            output.append(f"\n## {filename} (DELETED)\n")
        else:
            diff = generate_unified_diff(old_content, new_content, filename)
            if diff:
                output.append(f"\n## {filename}\n")
                output.append("```diff\n")
                output.append(diff)
                output.append("\n```\n")

    if len(output) == 1:
        output.append("\nNo changes detected.\n")

    return ''.join(output)


def diff(
    manuscript: str = None,
    from_cycle: int = None,
    to_cycle: int = None,
    from_commit: str = None,
    format: str = 'markdown',
) -> Optional[Path]:
    """
    Generate diff between review cycles or commits.

    Parameters
    ----------
    manuscript : str, optional
        Manuscript name. Defaults to DEFAULT_MANUSCRIPT.
    from_cycle : int, optional
        Starting review cycle number. If not provided, uses previous cycle.
    to_cycle : int, optional
        Ending review cycle number. If not provided, uses current.
    from_commit : str, optional
        Git commit to compare from. Overrides from_cycle.
    format : str
        Output format ('markdown' or 'unified')

    Returns
    -------
    Path or None
        Path to generated diff file, or None if generation failed.
    """
    paths = get_manuscript_paths(manuscript)
    manuscript_key = manuscript or DEFAULT_MANUSCRIPT

    print(f"Generating Diff: {paths['name']}")
    print("=" * 50)

    # Get current manuscript content
    current_files = get_manuscript_files(paths['manuscript_dir'])
    if not current_files:
        print("ERROR: No manuscript files found")
        return None

    # Determine comparison point
    if from_commit:
        print(f"Comparing against commit: {from_commit}")
        old_files = get_files_at_commit(paths['manuscript_dir'], from_commit)
    elif from_cycle:
        # Find the commit for that cycle from archive
        archive_file = paths['archive_dir'] / f'review_{from_cycle:02d}.md'
        if not archive_file.exists():
            print(f"ERROR: Review cycle {from_cycle} not found in archive")
            return None

        content = archive_file.read_text()
        frontmatter, _ = parse_frontmatter(content)

        if frontmatter and frontmatter.get('end_commit'):
            from_commit = frontmatter['end_commit']
            print(f"Using end commit from cycle {from_cycle}: {from_commit}")
            old_files = get_files_at_commit(paths['manuscript_dir'], from_commit)
        else:
            print(f"ERROR: No commit info in review cycle {from_cycle}")
            return None
    else:
        # Use start_commit from current tracker if available
        tracker_file = paths['tracker_file']
        if tracker_file.exists():
            content = tracker_file.read_text()
            frontmatter, _ = parse_frontmatter(content)
            if frontmatter and frontmatter.get('start_commit'):
                from_commit = frontmatter['start_commit']
                print(f"Comparing against start of current cycle: {from_commit}")
                old_files = get_files_at_commit(paths['manuscript_dir'], from_commit)
            else:
                print("ERROR: No start_commit in current tracker. Use --commit to specify.")
                return None
        else:
            print("ERROR: No active review cycle. Use --commit to specify comparison point.")
            return None

    if not old_files:
        print("WARNING: Could not retrieve old files from git. Showing current state only.")
        old_files = {}

    # Generate diff
    if format == 'markdown':
        diff_content = generate_markdown_diff(old_files, current_files)
    else:
        # Unified diff format
        diffs = []
        for filename in sorted(set(old_files.keys()) | set(current_files.keys())):
            old = old_files.get(filename, '')
            new = current_files.get(filename, '')
            if old != new:
                diffs.append(generate_unified_diff(old, new, filename))
        diff_content = '\n'.join(diffs)

    # Write to file
    output_file = paths['reviews_dir'] / f'diff_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    paths['reviews_dir'].mkdir(parents=True, exist_ok=True)
    output_file.write_text(diff_content)

    print(f"\nDiff generated: {output_file}")

    # Show summary
    changed_files = sum(1 for f in set(old_files.keys()) | set(current_files.keys())
                        if old_files.get(f, '') != current_files.get(f, ''))
    print(f"Files changed: {changed_files}")

    return output_file


# =============================================================================
# RESPONSE LETTER GENERATION
# =============================================================================

def parse_tracker_comments(content: str) -> list[dict]:
    """
    Parse reviewer comments from REVISION_TRACKER.md content.

    Returns list of comment dictionaries with keys:
    - id: Comment identifier (e.g., "Comment 1", "R1 Major 1")
    - type: 'major' or 'minor'
    - reviewer: Reviewer identifier if present
    - status: Status classification
    - concern: Original reviewer comment
    - response: Author's response
    - files: List of files modified
    """
    comments = []

    # Pattern for major comments
    major_pattern = re.compile(
        r'###\s+(?:(?P<reviewer>R\d+)\s+)?(?:Major\s+)?(?:Comment\s+)?(?P<num>\d+):\s*(?P<title>.+?)\n'
        r'.*?\*\*Status\*\*:\s*(?P<status>[^\n]+)\n'
        r'.*?\*\*(?:Reviewer\'s (?:Concern|Comment)|Original Comment)\*\*:\s*\n>\s*(?P<concern>.+?)\n'
        r'.*?\*\*Response\*\*:\s*\n(?P<response>.+?)\n'
        r'(?:\*\*Files Modified\*\*:\s*\n(?P<files>(?:- .+?\n)+))?',
        re.DOTALL | re.IGNORECASE
    )

    for match in major_pattern.finditer(content):
        files = []
        if match.group('files'):
            files = [f.strip('- \n') for f in match.group('files').strip().split('\n') if f.strip()]

        comments.append({
            'id': f"{match.group('reviewer') or ''} Major {match.group('num')}".strip(),
            'type': 'major',
            'title': match.group('title').strip(),
            'reviewer': match.group('reviewer'),
            'status': match.group('status').strip(),
            'concern': match.group('concern').strip(),
            'response': match.group('response').strip(),
            'files': files,
        })

    # Pattern for minor comments (simpler format)
    minor_pattern = re.compile(
        r'###\s+Minor\s+(?P<num>\d+):\s*(?P<title>.+?)\n'
        r'.*?\*\*Status\*\*:\s*(?P<status>[^\n]+)\n'
        r'.*?\*\*(?:Concern|Comment)\*\*:\s*(?P<concern>.+?)\n'
        r'.*?\*\*Response\*\*:\s*(?P<response>.+?)(?=\n---|$)',
        re.DOTALL | re.IGNORECASE
    )

    for match in minor_pattern.finditer(content):
        comments.append({
            'id': f"Minor {match.group('num')}",
            'type': 'minor',
            'title': match.group('title').strip(),
            'reviewer': None,
            'status': match.group('status').strip(),
            'concern': match.group('concern').strip(),
            'response': match.group('response').strip(),
            'files': [],
        })

    return comments


def generate_response_letter(
    manuscript: str = None,
    format: str = 'markdown',
    include_diffs: bool = False,
) -> Optional[Path]:
    """
    Generate a response letter from the current REVISION_TRACKER.

    Parameters
    ----------
    manuscript : str, optional
        Manuscript name. Defaults to DEFAULT_MANUSCRIPT.
    format : str
        Output format ('markdown')
    include_diffs : bool
        Whether to include file diffs inline

    Returns
    -------
    Path or None
        Path to generated response letter, or None if generation failed.
    """
    paths = get_manuscript_paths(manuscript)
    manuscript_key = manuscript or DEFAULT_MANUSCRIPT

    print(f"Generating Response Letter: {paths['name']}")
    print("=" * 50)

    tracker_file = paths['tracker_file']
    if not tracker_file.exists():
        print("ERROR: No active review to generate response for")
        return None

    content = tracker_file.read_text()
    frontmatter, body = parse_frontmatter(content)

    # Extract metadata
    if frontmatter:
        source_type = frontmatter.get('source_type', 'synthetic')
        cycle_number = frontmatter.get('cycle_number', 1)
        journal = frontmatter.get('journal', '')
        submission_round = frontmatter.get('submission_round', '')
        reviewer_ids = frontmatter.get('reviewer_ids', [])
    else:
        source_type = 'synthetic'
        cycle_number_match = re.search(r'\*\*Review\*\*:\s*#?(\d+)', content)
        cycle_number = int(cycle_number_match.group(1)) if cycle_number_match else 1
        journal = ''
        submission_round = ''
        reviewer_ids = []

    # Parse comments
    comments = parse_tracker_comments(body if frontmatter else content)

    if not comments:
        print("WARNING: No comments found in tracker")

    today = datetime.now().strftime('%Y-%m-%d')

    # Build response letter
    output = []

    if source_type == 'actual':
        output.append("# Response to Reviewers\n\n")
        if journal:
            output.append(f"**Journal**: {journal}\n")
        if submission_round:
            output.append(f"**Submission Round**: {submission_round}\n")
    else:
        output.append("# Response to Synthetic Review\n\n")

    output.append(f"**Manuscript**: {paths['title']}\n")
    output.append(f"**Date**: {today}\n\n")

    output.append("---\n\n")

    # Summary of changes
    addressed = sum(1 for c in comments if 'ADDRESSED' in c['status'].upper() or
                    ('VALID' in c['status'].upper() and c['response'] and
                     c['response'] != '[Describe response]'))
    beyond_scope = sum(1 for c in comments if 'BEYOND SCOPE' in c['status'].upper())
    pending = sum(1 for c in comments if 'ACTION NEEDED' in c['status'].upper() and
                  (not c['response'] or c['response'] == '[Describe response]'))

    output.append("## Summary of Changes\n\n")
    output.append(f"- **Comments Addressed**: {addressed}\n")
    output.append(f"- **Beyond Scope**: {beyond_scope}\n")
    if pending > 0:
        output.append(f"- **Pending**: {pending}\n")
    output.append("\n")

    # Group comments by reviewer if actual review
    if source_type == 'actual' and reviewer_ids:
        for rid in reviewer_ids:
            reviewer_comments = [c for c in comments if c.get('reviewer') == rid]
            if reviewer_comments:
                output.append(f"## Response to {rid}\n\n")
                for comment in reviewer_comments:
                    _add_comment_to_response(output, comment)
    else:
        # All comments together
        major_comments = [c for c in comments if c['type'] == 'major']
        minor_comments = [c for c in comments if c['type'] == 'minor']

        if major_comments:
            output.append("## Major Comments\n\n")
            for comment in major_comments:
                _add_comment_to_response(output, comment)

        if minor_comments:
            output.append("## Minor Comments\n\n")
            for comment in minor_comments:
                _add_comment_to_response(output, comment)

    # Footer
    output.append("---\n\n")
    output.append(f"*Generated by CENTAUR Review Management System on {today}*\n")

    # Write to file
    response_content = ''.join(output)
    output_file = paths['reviews_dir'] / f'response_letter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    paths['reviews_dir'].mkdir(parents=True, exist_ok=True)
    output_file.write_text(response_content)

    print(f"\nResponse letter generated: {output_file}")
    print(f"Comments included: {len(comments)}")

    return output_file


def _add_comment_to_response(output: list, comment: dict) -> None:
    """Helper to add a comment to the response letter output."""
    output.append(f"### {comment['id']}: {comment.get('title', 'Untitled')}\n\n")

    output.append("**Reviewer's Comment:**\n")
    output.append(f"> {comment['concern']}\n\n")

    output.append("**Our Response:**\n")
    response = comment['response']
    if response and response != '[Describe response]':
        output.append(f"{response}\n\n")
    else:
        output.append("*[Response pending]*\n\n")

    if comment.get('files'):
        output.append("**Files Modified:**\n")
        for f in comment['files']:
            output.append(f"- {f}\n")
        output.append("\n")

    output.append("---\n\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(
    action: str = 'status',
    manuscript: str = None,
    focus: str = 'general',
    # New parameters for actual reviews
    source_type: str = 'synthetic',
    journal: str = None,
    submission_round: str = None,
    decision: str = None,
    reviewer_ids: list[str] = None,
    # Archive parameters
    create_tag: bool = True,
    tag_name: str = None,
    # Diff parameters
    from_cycle: int = None,
    to_cycle: int = None,
    from_commit: str = None,
    diff_format: str = 'markdown',
):
    """Main entry point for review management."""
    if action == 'status':
        status(manuscript)
    elif action == 'new':
        new_cycle(
            manuscript=manuscript,
            focus=focus,
            source_type=source_type,
            journal=journal,
            submission_round=submission_round,
            decision=decision,
            reviewer_ids=reviewer_ids,
        )
    elif action == 'archive':
        archive(
            manuscript=manuscript,
            create_tag=create_tag,
            tag_name=tag_name,
        )
    elif action == 'verify':
        verify(manuscript)
    elif action == 'report':
        report()
    elif action == 'diff':
        diff(
            manuscript=manuscript,
            from_cycle=from_cycle,
            to_cycle=to_cycle,
            from_commit=from_commit,
            format=diff_format,
        )
    elif action == 'response':
        generate_response_letter(
            manuscript=manuscript,
            format=diff_format,
        )
    else:
        print(f"Unknown action: {action}")
        print("Available: status, new, archive, verify, report, diff, response")


if __name__ == '__main__':
    main()
