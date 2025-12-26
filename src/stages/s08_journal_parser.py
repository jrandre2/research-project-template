#!/usr/bin/env python3
"""
Stage 08: Journal Guidelines Parser

Purpose: Parse and manage journal configuration files for manuscript submission.

Commands
--------
parse     : Parse raw journal guidelines into structured YAML config
validate  : Validate a journal config against the comprehensive template
compare   : Compare manuscript against journal requirements
list      : List available journal configurations

Usage
-----
    python src/pipeline.py journal_parse --input guidelines.txt --output journal.yml
    python src/pipeline.py journal_validate --config natural_hazards.yml
    python src/pipeline.py journal_compare --journal natural_hazards
    python src/pipeline.py journal_list
"""
from __future__ import annotations

import re
import yaml
from pathlib import Path
from typing import Optional

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MANUSCRIPT_DIR = PROJECT_ROOT / 'manuscript_quarto'
JOURNAL_CONFIGS = MANUSCRIPT_DIR / 'journal_configs'
TEMPLATE_FILE = JOURNAL_CONFIGS / 'template_comprehensive.yml'

# Extraction prompt for LLM-based parsing
EXTRACTION_PROMPT = '''
Extract journal submission requirements from the following author guidelines.
Output as YAML matching this template structure:

{template}

Guidelines to parse:
{raw_guidelines}

Be comprehensive. Include all numeric limits, format specifications, and policies mentioned.
For fields not specified in the guidelines, use null.
'''


def list_configs() -> None:
    """List available journal configurations."""
    print("Available Journal Configurations")
    print("=" * 50)

    if not JOURNAL_CONFIGS.exists():
        print("\nNo journal configs directory found.")
        return

    configs = sorted(JOURNAL_CONFIGS.glob('*.yml'))
    template_files = ['template.yml', 'template_comprehensive.yml']

    print("\nTemplates:")
    for f in configs:
        if f.name in template_files:
            print(f"  - {f.name}")

    print("\nJournal Configs:")
    for f in configs:
        if f.name not in template_files:
            # Load and show basic info
            try:
                with open(f) as fp:
                    config = yaml.safe_load(fp)
                name = config.get('journal', {}).get('name', 'Unknown')
                abbrev = config.get('journal', {}).get('abbreviation', '')
                print(f"  - {f.stem}: {name} ({abbrev})")
            except Exception:
                print(f"  - {f.stem}: (error reading config)")

    print("\n" + "=" * 50)


def validate_config(config_name: str) -> None:
    """Validate a journal config against the comprehensive template."""
    print(f"Validating: {config_name}")
    print("=" * 50)

    config_file = JOURNAL_CONFIGS / f'{config_name}.yml'
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        return

    if not TEMPLATE_FILE.exists():
        print(f"ERROR: Template file not found: {TEMPLATE_FILE}")
        return

    # Load files
    with open(config_file) as f:
        config = yaml.safe_load(f)
    with open(TEMPLATE_FILE) as f:
        template = yaml.safe_load(f)

    # Check required sections
    required_sections = [
        'journal', 'submission', 'abstract', 'keywords',
        'text_formatting', 'references', 'declarations',
        'tables', 'artwork', 'ethical_responsibilities',
        'authorship', 'data_policy'
    ]

    missing = []
    present = []

    for section in required_sections:
        if section in config and config[section]:
            present.append(section)
        else:
            missing.append(section)

    print(f"\nRequired Sections: {len(present)}/{len(required_sections)} present")
    print("\nPresent:")
    for s in present:
        print(f"  [x] {s}")

    if missing:
        print("\nMissing:")
        for s in missing:
            print(f"  [ ] {s}")

    # Check key fields
    print("\nKey Field Validation:")

    checks = [
        ('journal.name', config.get('journal', {}).get('name')),
        ('journal.publisher', config.get('journal', {}).get('publisher')),
        ('abstract.max_words', config.get('abstract', {}).get('max_words')),
        ('keywords.max', config.get('keywords', {}).get('max')),
        ('references.style', config.get('references', {}).get('style')),
    ]

    for field, value in checks:
        status = "[x]" if value else "[ ]"
        print(f"  {status} {field}: {value}")

    print("\n" + "=" * 50)


def compare_manuscript(journal_name: str, manuscript_path: Optional[str] = None) -> None:
    """Compare manuscript against journal requirements."""
    print(f"Comparing manuscript to: {journal_name}")
    print("=" * 50)

    config_file = JOURNAL_CONFIGS / f'{journal_name}.yml'
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        return

    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Default manuscript location
    if manuscript_path is None:
        manuscript_path = MANUSCRIPT_DIR

    manuscript_dir = Path(manuscript_path)
    if not manuscript_dir.exists():
        print(f"ERROR: Manuscript directory not found: {manuscript_dir}")
        return

    print(f"\nManuscript: {manuscript_dir}")
    print(f"Journal: {config.get('journal', {}).get('name', 'Unknown')}")

    # Check for required files
    print("\nRequired Files:")
    required_files = [
        ('index.qmd', 'Main manuscript'),
        ('references.bib', 'Bibliography'),
    ]

    for filename, desc in required_files:
        filepath = manuscript_dir / filename
        status = "[x]" if filepath.exists() else "[ ]"
        print(f"  {status} {filename} - {desc}")

    # Check abstract requirements
    abstract_config = config.get('abstract', {})
    print(f"\nAbstract Requirements:")
    print(f"  Min words: {abstract_config.get('min_words', 'Not specified')}")
    print(f"  Max words: {abstract_config.get('max_words', 'Not specified')}")
    print(f"  No abbreviations: {abstract_config.get('no_abbreviations', False)}")
    print(f"  No citations: {abstract_config.get('no_citations', False)}")

    # Check keyword requirements
    keywords_config = config.get('keywords', {})
    print(f"\nKeyword Requirements:")
    print(f"  Min: {keywords_config.get('min', 'Not specified')}")
    print(f"  Max: {keywords_config.get('max', 'Not specified')}")

    # Check figure requirements
    artwork_config = config.get('artwork', {})
    resolution = artwork_config.get('resolution', {})
    print(f"\nFigure Resolution Requirements:")
    print(f"  Line art: {resolution.get('line_art_dpi', 'Not specified')} dpi")
    print(f"  Halftone: {resolution.get('halftone_dpi', 'Not specified')} dpi")
    print(f"  Combination: {resolution.get('combination_dpi', 'Not specified')} dpi")

    # Check reference style
    refs_config = config.get('references', {})
    print(f"\nReference Style:")
    print(f"  Style: {refs_config.get('style', 'Not specified')}")
    print(f"  In-text format: {refs_config.get('in_text_format', 'Not specified')}")

    # Submission checklist
    submission = config.get('submission', {})
    checklist = submission.get('checklist', [])
    if checklist:
        print(f"\nSubmission Checklist:")
        for item in checklist:
            print(f"  [ ] {item}")

    print("\n" + "=" * 50)


def parse_guidelines(input_file: str, output_file: str, journal_name: Optional[str] = None) -> None:
    """Parse raw guidelines into structured config.

    Note: This is a stub for LLM-based parsing. In practice, this would use
    an LLM to extract structured data from raw guidelines text.
    """
    print("Journal Guidelines Parser")
    print("=" * 50)

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    output_path = JOURNAL_CONFIGS / output_file
    if not output_file.endswith('.yml'):
        output_path = JOURNAL_CONFIGS / f'{output_file}.yml'

    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")

    # Read raw guidelines
    with open(input_path) as f:
        raw_text = f.read()

    print(f"\nGuidelines length: {len(raw_text)} characters")

    # Load template
    if not TEMPLATE_FILE.exists():
        print(f"ERROR: Template file not found: {TEMPLATE_FILE}")
        return

    with open(TEMPLATE_FILE) as f:
        template = yaml.safe_load(f)

    # TODO: Implement LLM-based extraction
    # For now, create a stub config
    print("\nNOTE: LLM-based extraction not yet implemented.")
    print("Creating stub config from template...")

    # Basic extraction using regex patterns
    config = template.copy()

    # Try to extract journal name
    if journal_name:
        config['journal']['name'] = journal_name

    # Try to extract abstract word limit
    abstract_match = re.search(r'abstract.*?(\d+)\s*(?:to|-)?\s*(\d+)?\s*words', raw_text, re.I)
    if abstract_match:
        if abstract_match.group(2):
            config['abstract']['min_words'] = int(abstract_match.group(1))
            config['abstract']['max_words'] = int(abstract_match.group(2))
        else:
            config['abstract']['max_words'] = int(abstract_match.group(1))

    # Try to extract keyword limit
    keyword_match = re.search(r'(\d+)\s*(?:to|-)?\s*(\d+)?\s*keywords', raw_text, re.I)
    if keyword_match:
        if keyword_match.group(2):
            config['keywords']['min'] = int(keyword_match.group(1))
            config['keywords']['max'] = int(keyword_match.group(2))
        else:
            config['keywords']['max'] = int(keyword_match.group(1))

    # Write output
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nCreated: {output_path}")
    print("\nNext steps:")
    print("1. Review and edit the generated config")
    print("2. Fill in missing fields from the guidelines")
    print("3. Create a Quarto profile if needed (_quarto-{abbrev}.yml)")
    print("4. Test with: python src/pipeline.py journal_validate --config {output_file}")


def main(action: str = 'list', **kwargs) -> None:
    """Main entry point for journal parser."""
    if action == 'list':
        list_configs()
    elif action == 'validate':
        config = kwargs.get('config', 'template_comprehensive')
        validate_config(config)
    elif action == 'compare':
        journal = kwargs.get('journal', 'natural_hazards')
        manuscript = kwargs.get('manuscript')
        compare_manuscript(journal, manuscript)
    elif action == 'parse':
        input_file = kwargs.get('input')
        output = kwargs.get('output', 'new_journal.yml')
        journal = kwargs.get('journal')
        if not input_file:
            print("ERROR: --input file required for parsing")
            return
        parse_guidelines(input_file, output, journal)
    else:
        print(f"Unknown action: {action}")
        print("Available: list, validate, compare, parse")


if __name__ == '__main__':
    main()
