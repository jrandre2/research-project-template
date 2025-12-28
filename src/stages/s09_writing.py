#!/usr/bin/env python3
"""
Stage 09: AI-Assisted Manuscript Writing

Purpose: Generate draft manuscript sections from pipeline outputs using LLMs.

Supports multiple LLM providers (Anthropic Claude, OpenAI GPT-4) with a unified
interface. All outputs require human review before integration into the manuscript.

Commands
--------
draft_results  : Draft results section from estimation tables
draft_captions : Generate figure captions
draft_abstract : Synthesize abstract from manuscript content

Usage
-----
    python src/pipeline.py draft_results --table main_results
    python src/pipeline.py draft_results --table main_results --dry-run
    python src/pipeline.py draft_captions --figure "fig_*.png"
    python src/pipeline.py draft_abstract --max-words 200
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import re

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from stages._qa_utils import generate_qa_report, QAMetrics

# Import config
from config import (
    DRAFTS_DIR,
    DIAGNOSTICS_DIR,
    MANUSCRIPT_DIR,
    MANUSCRIPTS,
    DEFAULT_MANUSCRIPT,
    LLM_PROVIDER,
)


# =============================================================================
# DRAFT METADATA
# =============================================================================

DRAFT_HEADER_TEMPLATE = """<!-- AI-Generated Draft
     Source: {source}
     Provider: {provider}/{model}
     Generated: {timestamp}
     Status: REQUIRES HUMAN REVIEW

     Instructions:
     1. Review and edit this content carefully
     2. Verify all statistics and claims against source data
     3. Integrate into manuscript after review
     4. Delete this header block when finalizing
-->

"""


def _create_draft_header(source: str, provider_name: str, model_name: str) -> str:
    """Create metadata header for draft files."""
    return DRAFT_HEADER_TEMPLATE.format(
        source=source,
        provider=provider_name,
        model=model_name,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )


def _get_output_path(draft_type: str, section: str = None) -> Path:
    """Generate output path for a draft file."""
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if section:
        filename = f"{draft_type}_{section}_{timestamp}.md"
    else:
        filename = f"{draft_type}_{timestamp}.md"
    return DRAFTS_DIR / filename


# =============================================================================
# MANUSCRIPT UTILITIES
# =============================================================================

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
    """
    if manuscript is None:
        manuscript = DEFAULT_MANUSCRIPT

    if manuscript not in MANUSCRIPTS:
        available = ', '.join(MANUSCRIPTS.keys())
        raise ValueError(f"Unknown manuscript '{manuscript}'. Available: {available}")

    config = MANUSCRIPTS[manuscript]
    return {
        'manuscript_dir': config['dir'],
        'name': config['name'],
        'drafts_dir': DRAFTS_DIR,
    }


def _extract_section_content(qmd_path: Path, section_pattern: str) -> str:
    """
    Extract content from a section of a Quarto document.

    Parameters
    ----------
    qmd_path : Path
        Path to the .qmd file.
    section_pattern : str
        Regex pattern to match section header.

    Returns
    -------
    str
        Extracted section content.
    """
    if not qmd_path.exists():
        return ""

    content = qmd_path.read_text()

    # Find section start
    match = re.search(section_pattern, content, re.IGNORECASE | re.MULTILINE)
    if not match:
        return ""

    start = match.end()

    # Find next section (## header)
    next_section = re.search(r'^##\s+', content[start:], re.MULTILINE)
    if next_section:
        end = start + next_section.start()
    else:
        end = len(content)

    return content[start:end].strip()


# =============================================================================
# DRAFT RESULTS
# =============================================================================

def draft_results(
    table_name: str,
    section: str = 'main',
    manuscript: str = None,
    dry_run: bool = False,
    provider: str = None,
    variable_map: Optional[dict] = None,
    additional_context: Optional[str] = None,
) -> Optional[Path]:
    """
    Draft a results section from an estimation table.

    Parameters
    ----------
    table_name : str
        Name of the diagnostic CSV file (without .csv extension).
    section : str
        Section name for output file naming.
    manuscript : str, optional
        Target manuscript.
    dry_run : bool
        If True, show prompt without API call.
    provider : str, optional
        LLM provider override ('anthropic' or 'openai').
    variable_map : dict, optional
        Mapping of variable names to descriptions.
    additional_context : str, optional
        Additional context for the LLM.

    Returns
    -------
    Path or None
        Path to generated draft file, or None if dry_run.
    """
    from llm import get_provider
    from llm.prompts import build_results_prompt

    # Load the estimation table
    table_path = DIAGNOSTICS_DIR / f"{table_name}.csv"
    if not table_path.exists():
        raise FileNotFoundError(
            f"Table not found: {table_path}\n"
            f"Available tables: {list(DIAGNOSTICS_DIR.glob('*.csv'))}"
        )

    df = pd.read_csv(table_path)
    print(f"Loaded table: {table_path} ({len(df)} rows)")

    # Build prompt
    prompt, system = build_results_prompt(
        table_df=df,
        variable_map=variable_map,
        additional_context=additional_context,
    )

    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Prompt Preview")
        print("=" * 60)
        print(f"\n[System Prompt]\n{system}\n")
        print(f"\n[User Prompt]\n{prompt}\n")
        print("=" * 60)
        return None

    # Get provider and generate
    llm = get_provider(provider)
    print(f"Using provider: {provider or LLM_PROVIDER} ({llm.model_name})")

    print("Generating draft...")
    response = llm.complete(prompt, system=system)

    # Create output file
    header = _create_draft_header(
        source=str(table_path),
        provider_name=provider or LLM_PROVIDER,
        model_name=llm.model_name,
    )

    output_path = _get_output_path('results', section)
    output_content = header + f"## {section.title()} Results\n\n" + response

    output_path.write_text(output_content)
    print(f"\nDraft saved to: {output_path}")

    # Generate QA report
    metrics = QAMetrics(
        stage_name='s09_writing',
        row_count=len(df),
        column_count=len(df.columns),
        output_files=[str(output_path)],
        custom_metrics={
            'source_table': table_name,
            'draft_type': 'results',
            'provider': provider or LLM_PROVIDER,
            'word_count': len(response.split()),
        },
    )
    generate_qa_report(metrics)

    return output_path


# =============================================================================
# DRAFT CAPTIONS
# =============================================================================

def draft_captions(
    figure_pattern: str,
    manuscript: str = None,
    dry_run: bool = False,
    provider: str = None,
) -> Optional[Path]:
    """
    Generate captions for figures matching a pattern.

    Parameters
    ----------
    figure_pattern : str
        Glob pattern for figure files (e.g., "fig_*.png").
    manuscript : str, optional
        Target manuscript.
    dry_run : bool
        If True, show prompt without API call.
    provider : str, optional
        LLM provider override.

    Returns
    -------
    Path or None
        Path to generated captions file, or None if dry_run.
    """
    from llm import get_provider
    from llm.prompts import build_caption_prompt

    paths = get_manuscript_paths(manuscript)
    figures_dir = paths['manuscript_dir'] / 'figures'

    # Find matching figures
    figures = list(figures_dir.glob(figure_pattern))
    if not figures:
        raise FileNotFoundError(
            f"No figures matching '{figure_pattern}' in {figures_dir}"
        )

    print(f"Found {len(figures)} figures matching pattern")

    if dry_run:
        # Show prompt for first figure only
        prompt, system = build_caption_prompt(
            filename=figures[0].name,
            context="Context would be extracted from manuscript.",
        )
        print("\n" + "=" * 60)
        print("DRY RUN - Prompt Preview (first figure)")
        print("=" * 60)
        print(f"\n[System Prompt]\n{system}\n")
        print(f"\n[User Prompt]\n{prompt}\n")
        print("=" * 60)
        return None

    # Get provider
    llm = get_provider(provider)
    print(f"Using provider: {provider or LLM_PROVIDER} ({llm.model_name})")

    captions = []
    for fig_path in figures:
        print(f"  Generating caption for: {fig_path.name}")

        prompt, system = build_caption_prompt(
            filename=fig_path.name,
            context=f"Figure from {paths['name']} manuscript.",
        )

        response = llm.complete(prompt, system=system)
        captions.append(f"### {fig_path.name}\n\n{response}\n")

    # Create output file
    header = _create_draft_header(
        source=f"{figures_dir} ({len(figures)} figures)",
        provider_name=provider or LLM_PROVIDER,
        model_name=llm.model_name,
    )

    output_path = _get_output_path('captions')
    output_content = header + "## Figure Captions\n\n" + "\n".join(captions)

    output_path.write_text(output_content)
    print(f"\nCaptions saved to: {output_path}")

    # Generate QA report
    metrics = QAMetrics(
        stage_name='s09_writing',
        row_count=len(figures),
        column_count=1,
        output_files=[str(output_path)],
        custom_metrics={
            'draft_type': 'captions',
            'figure_count': len(figures),
            'provider': provider or LLM_PROVIDER,
        },
    )
    generate_qa_report(metrics)

    return output_path


# =============================================================================
# DRAFT ABSTRACT
# =============================================================================

def draft_abstract(
    manuscript: str = None,
    max_words: int = 250,
    dry_run: bool = False,
    provider: str = None,
) -> Optional[Path]:
    """
    Synthesize an abstract from manuscript sections.

    Parameters
    ----------
    manuscript : str, optional
        Target manuscript.
    max_words : int
        Target word limit for abstract.
    dry_run : bool
        If True, show prompt without API call.
    provider : str, optional
        LLM provider override.

    Returns
    -------
    Path or None
        Path to generated abstract file, or None if dry_run.
    """
    from llm import get_provider
    from llm.prompts import build_abstract_prompt, truncate_text

    paths = get_manuscript_paths(manuscript)
    manuscript_dir = paths['manuscript_dir']

    # Find main manuscript file
    main_qmd = manuscript_dir / 'index.qmd'
    if not main_qmd.exists():
        raise FileNotFoundError(f"Main manuscript not found: {main_qmd}")

    print(f"Extracting sections from: {main_qmd}")

    # Extract sections
    introduction = _extract_section_content(main_qmd, r'^##\s+Introduction')
    methods = _extract_section_content(main_qmd, r'^##\s+(Methods|Methodology|Data and Methods)')
    results = _extract_section_content(main_qmd, r'^##\s+Results')
    conclusion = _extract_section_content(main_qmd, r'^##\s+(Conclusion|Discussion)')

    # Truncate long sections
    introduction = truncate_text(introduction, 2000)
    methods = truncate_text(methods, 2000)
    results = truncate_text(results, 2000)
    conclusion = truncate_text(conclusion, 2000)

    # Build prompt
    prompt, system = build_abstract_prompt(
        introduction=introduction or "Introduction section not found.",
        methods=methods or "Methods section not found.",
        results=results or "Results section not found.",
        conclusion=conclusion or "Conclusion section not found.",
        max_words=max_words,
    )

    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Prompt Preview")
        print("=" * 60)
        print(f"\n[System Prompt]\n{system}\n")
        print(f"\n[User Prompt]\n{prompt}\n")
        print("=" * 60)
        return None

    # Get provider and generate
    llm = get_provider(provider)
    print(f"Using provider: {provider or LLM_PROVIDER} ({llm.model_name})")

    print("Generating abstract...")
    response = llm.complete(prompt, system=system)

    # Count words
    word_count = len(response.split())
    print(f"Generated abstract: {word_count} words (target: {max_words})")

    # Create output file
    header = _create_draft_header(
        source=str(main_qmd),
        provider_name=provider or LLM_PROVIDER,
        model_name=llm.model_name,
    )

    output_path = _get_output_path('abstract')
    output_content = header + "## Abstract\n\n" + response

    output_path.write_text(output_content)
    print(f"\nAbstract saved to: {output_path}")

    # Generate QA report
    metrics = QAMetrics(
        stage_name='s09_writing',
        row_count=1,
        column_count=1,
        output_files=[str(output_path)],
        custom_metrics={
            'draft_type': 'abstract',
            'word_count': word_count,
            'target_words': max_words,
            'provider': provider or LLM_PROVIDER,
        },
    )
    generate_qa_report(metrics)

    return output_path


# =============================================================================
# CLI ENTRY POINTS
# =============================================================================

def main_draft_results(args):
    """CLI entry point for draft_results."""
    return draft_results(
        table_name=args.table,
        section=args.section,
        manuscript=args.manuscript,
        dry_run=args.dry_run,
        provider=args.provider,
    )


def main_draft_captions(args):
    """CLI entry point for draft_captions."""
    return draft_captions(
        figure_pattern=args.figure,
        manuscript=args.manuscript,
        dry_run=args.dry_run,
        provider=args.provider,
    )


def main_draft_abstract(args):
    """CLI entry point for draft_abstract."""
    return draft_abstract(
        manuscript=args.manuscript,
        max_words=args.max_words,
        dry_run=args.dry_run,
        provider=args.provider,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='AI-Assisted Manuscript Writing')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # draft_results
    p_results = subparsers.add_parser('results', help='Draft results section')
    p_results.add_argument('--table', '-t', required=True, help='Diagnostic CSV name')
    p_results.add_argument('--section', '-s', default='main', help='Section name')
    p_results.add_argument('--manuscript', '-m', default=None, help='Target manuscript')
    p_results.add_argument('--dry-run', action='store_true', help='Show prompt only')
    p_results.add_argument('--provider', '-p', default=None, help='LLM provider')

    # draft_captions
    p_captions = subparsers.add_parser('captions', help='Generate figure captions')
    p_captions.add_argument('--figure', '-f', required=True, help='Figure glob pattern')
    p_captions.add_argument('--manuscript', '-m', default=None, help='Target manuscript')
    p_captions.add_argument('--dry-run', action='store_true', help='Show prompt only')
    p_captions.add_argument('--provider', '-p', default=None, help='LLM provider')

    # draft_abstract
    p_abstract = subparsers.add_parser('abstract', help='Synthesize abstract')
    p_abstract.add_argument('--manuscript', '-m', default=None, help='Target manuscript')
    p_abstract.add_argument('--max-words', type=int, default=250, help='Word limit')
    p_abstract.add_argument('--dry-run', action='store_true', help='Show prompt only')
    p_abstract.add_argument('--provider', '-p', default=None, help='LLM provider')

    args = parser.parse_args()

    if args.command == 'results':
        main_draft_results(args)
    elif args.command == 'captions':
        main_draft_captions(args)
    elif args.command == 'abstract':
        main_draft_abstract(args)
    else:
        parser.print_help()
