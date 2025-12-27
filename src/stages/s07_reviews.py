#!/usr/bin/env python3
"""
Stage 07: Review Management

Purpose: Manage synthetic peer review cycles for manuscript development.

Supports multiple manuscripts with independent review tracking and focus-specific
prompts for comprehensive pre-submission review.

Commands
--------
status : Display current review cycle status
new    : Initialize a new review cycle with focus-specific template
archive: Archive current cycle and reset for new one
verify : Run verification checklist with journal compliance checks
report : Generate summary report of all review cycles

Usage
-----
    python src/pipeline.py review_status --manuscript main
    python src/pipeline.py review_new --manuscript main --focus economics
    python src/pipeline.py review_archive --manuscript main
    python src/pipeline.py review_verify --manuscript main
    python src/pipeline.py review_report
"""
from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from stages._qa_utils import generate_qa_report, QAMetrics

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
# REVIEW MANAGEMENT FUNCTIONS
# =============================================================================

def status(manuscript: str = None):
    """Display current review cycle status from REVISION_TRACKER.md."""
    paths = get_manuscript_paths(manuscript)
    tracker_file = paths['tracker_file']

    print(f"Review Status: {paths['name']}")
    print("=" * 50)

    if not tracker_file.exists():
        print("\nNo active review cycle.")
        print(f"Start one with: python src/pipeline.py review_new --manuscript {manuscript or DEFAULT_MANUSCRIPT} --focus <name>")
        return

    content = tracker_file.read_text()

    # Parse summary statistics
    print(f"\nCurrent Tracker: {tracker_file}")

    # Extract review number and focus
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


def new_cycle(manuscript: str = None, focus: str = 'general'):
    """Initialize a new review cycle with focus-specific template."""
    paths = get_manuscript_paths(manuscript)

    if focus not in FOCUS_PROMPTS:
        print(f"ERROR: Unknown focus '{focus}'")
        print(f"Available: {', '.join(FOCUS_PROMPTS.keys())}")
        return

    print(f"Initializing new review cycle")
    print(f"  Manuscript: {paths['name']}")
    print(f"  Focus: {focus}")
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
            print(f"Archive current review first with: python src/pipeline.py review_archive --manuscript {manuscript or DEFAULT_MANUSCRIPT}")
            return

    # Create new tracker from template
    today = datetime.now().strftime('%Y-%m-%d')
    template = f'''# Revision Tracker: Response to Synthetic Review

**Document**: {paths['title']}
**Review**: #{review_num}
**Focus**: {focus}
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
{FOCUS_PROMPTS[focus]}
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

    tracker_file.write_text(template)
    print(f"\nCreated: {tracker_file}")
    print(f"\nReview #{review_num} initialized with {focus} focus.")
    print("\nNext steps:")
    print("1. Generate a synthetic review using the prompt above")
    print("2. Paste reviewer comments into the tracker")
    print("3. Triage each comment with a status classification")
    print("4. Implement changes and update tracker")
    print(f"5. Run: python src/pipeline.py review_verify --manuscript {manuscript or DEFAULT_MANUSCRIPT}")


def archive(manuscript: str = None):
    """Archive current review cycle and reset for new one."""
    paths = get_manuscript_paths(manuscript)

    print(f"Archiving Review Cycle: {paths['name']}")
    print("=" * 50)

    tracker_file = paths['tracker_file']
    if not tracker_file.exists():
        print("No active review to archive.")
        return

    # Ensure archive directory exists
    paths['archive_dir'].mkdir(parents=True, exist_ok=True)

    # Determine archive filename
    content = tracker_file.read_text()
    review_match = re.search(r'\*\*Review\*\*:\s*#?(\d+)', content)

    if review_match:
        review_num = review_match.group(1)
    else:
        # Find next available number
        existing = list(paths['archive_dir'].glob('review_*.md'))
        nums = [int(re.search(r'review_(\d+)', f.name).group(1))
                for f in existing if re.search(r'review_(\d+)', f.name)]
        review_num = max(nums) + 1 if nums else 1

    archive_file = paths['archive_dir'] / f'review_{int(review_num):02d}.md'

    # Copy to archive
    shutil.copy(tracker_file, archive_file)
    print(f"Archived to: {archive_file}")

    # Reset tracker
    tracker_file.unlink()
    print(f"Removed: {tracker_file}")

    print(f"\nReview #{review_num} archived successfully.")
    print(f"Start new review with: python src/pipeline.py review_new --manuscript {manuscript or DEFAULT_MANUSCRIPT} --focus <name>")


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
# MAIN ENTRY POINT
# =============================================================================

def main(action: str = 'status', manuscript: str = None, focus: str = 'general'):
    """Main entry point for review management."""
    if action == 'status':
        status(manuscript)
    elif action == 'new':
        new_cycle(manuscript, focus)
    elif action == 'archive':
        archive(manuscript)
    elif action == 'verify':
        verify(manuscript)
    elif action == 'report':
        report()
    else:
        print(f"Unknown action: {action}")
        print("Available: status, new, archive, verify, report")


if __name__ == '__main__':
    main()
