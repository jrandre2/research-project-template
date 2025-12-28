#!/usr/bin/env python3
"""
Module: pipeline.py
Purpose: Main orchestration CLI for the research analysis pipeline.

This module provides a command-line interface to execute individual stages
of the analysis pipeline. Each stage processes data and produces intermediate
or final outputs.

Commands
--------
# Data Processing
ingest_data : Load and preprocess raw data
    Output: data_work/data_raw.parquet
link_records : Link records across data sources
    Output: data_work/data_linked.parquet
build_panel : Create analysis panel
    Output: data_work/panel.parquet

# Analysis
run_estimation : Run primary estimation
    Options: --specification, --sample
estimate_robustness : Run robustness checks
    Output: data_work/diagnostics/

# Figures and Manuscript
make_figures : Generate publication figures
    Output: manuscript_quarto/figures/*.png
validate_submission : Validate against journal requirements
    Options: --journal, --report

# Review Management
review_status : Show current review cycle status
    Options: --manuscript
review_new : Initialize new review cycle
    Options: --manuscript, --focus (or deprecated --discipline)
review_archive : Archive current cycle and reset
    Options: --manuscript
review_verify : Run verification checklist with compliance checks
    Options: --manuscript
review_report : Generate summary of all review cycles

# Journal Configuration
journal_list : List available journal configurations
journal_validate : Validate journal config against template
    Options: --config
journal_compare : Compare manuscript to journal requirements
    Options: --journal, --manuscript
journal_parse : Parse raw guidelines into config
    Options: --input, --output, --journal

# Data Audit
audit_data : Audit pipeline data files
    Options: --full, --report, --output

# Project Migration (AI Agent Tools)
analyze_project : Analyze external project structure
    Options: --path, --output
map_project : Map project structure to template
    Options: --path, --output
plan_migration : Generate migration plan
    Options: --path, --target, --output
migrate_project : Execute migration (interactive)
    Options: --path, --target, --dry-run

# Stage Versioning
run_stage : Run a specific stage by name (supports versioned stages)
    Options: <stage_name>
list_stages : List available stage versions
    Options: --prefix

Usage
-----
    python src/pipeline.py ingest_data
    python src/pipeline.py run_estimation --specification baseline

Notes
-----
Requires activation of project virtual environment before running.
"""
from __future__ import annotations

import os
import sys
import argparse


def ensure_env():
    """Verify virtual environment is activated."""
    venv = os.getenv('VIRTUAL_ENV')
    if not venv or not venv.endswith('/.venv'):
        print(
            'ERROR: Please activate project .venv (source .venv/bin/activate) before running.',
            file=sys.stderr
        )
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description='Research Project Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    # Data Processing Commands
    p_ingest = sub.add_parser('ingest_data', help='Load and preprocess raw data')
    p_ingest.add_argument(
        '--demo',
        action='store_true',
        help='Use synthetic demo data instead of real data'
    )
    sub.add_parser('link_records', help='Link records across data sources')
    sub.add_parser('build_panel', help='Create analysis panel')

    # Estimation Commands
    p_est = sub.add_parser('run_estimation', help='Run primary estimation')
    p_est.add_argument(
        '--specification', '-s',
        default='baseline',
        help='Specification name (default: baseline)'
    )
    p_est.add_argument(
        '--sample',
        default='full',
        help='Sample restriction (default: full)'
    )

    sub.add_parser('estimate_robustness', help='Run robustness checks')

    # Figure and Manuscript Commands
    sub.add_parser('make_figures', help='Generate publication figures')

    p_val = sub.add_parser('validate_submission', help='Validate manuscript')
    p_val.add_argument(
        '--journal', '-j',
        default='jeem',
        help='Target journal (default: jeem)'
    )
    p_val.add_argument(
        '--report',
        action='store_true',
        help='Generate markdown report'
    )

    # Review Management Commands
    p_status = sub.add_parser('review_status', help='Show current review cycle status')
    p_status.add_argument(
        '--manuscript', '-m',
        default='main',
        help='Manuscript to check (default: main)'
    )

    p_new = sub.add_parser('review_new', help='Initialize new review cycle')
    p_new.add_argument(
        '--manuscript', '-m',
        default='main',
        help='Manuscript for review (default: main)'
    )
    p_new.add_argument(
        '--focus', '-f',
        default='general',
        choices=['economics', 'engineering', 'social_sciences', 'general', 'methods', 'policy', 'clarity'],
        help='Focus area for review prompts (default: general)'
    )
    p_new.add_argument(
        '--discipline', '-d',
        dest='focus',
        help='(Deprecated, use --focus) Discipline for review prompts'
    )

    p_archive = sub.add_parser('review_archive', help='Archive current cycle and reset')
    p_archive.add_argument(
        '--manuscript', '-m',
        default='main',
        help='Manuscript to archive (default: main)'
    )

    p_verify = sub.add_parser('review_verify', help='Run verification checklist')
    p_verify.add_argument(
        '--manuscript', '-m',
        default='main',
        help='Manuscript to verify (default: main)'
    )

    sub.add_parser('review_report', help='Generate summary of all review cycles')

    # Journal Configuration Commands
    sub.add_parser('journal_list', help='List available journal configurations')

    p_jval = sub.add_parser('journal_validate', help='Validate journal config')
    p_jval.add_argument(
        '--config', '-c',
        default='natural_hazards',
        help='Config file name without .yml (default: natural_hazards)'
    )

    p_jcmp = sub.add_parser('journal_compare', help='Compare manuscript to journal')
    p_jcmp.add_argument(
        '--journal', '-j',
        default='natural_hazards',
        help='Journal config name (default: natural_hazards)'
    )
    p_jcmp.add_argument(
        '--manuscript', '-m',
        default=None,
        help='Path to manuscript directory (default: manuscript_quarto/)'
    )

    p_jparse = sub.add_parser('journal_parse', help='Parse guidelines into config')
    p_jparse_source = p_jparse.add_mutually_exclusive_group(required=True)
    p_jparse_source.add_argument(
        '--input', '-i',
        help='Input file with raw guidelines'
    )
    p_jparse_source.add_argument(
        '--url', '-u',
        help='URL to journal author guidelines'
    )
    p_jparse.add_argument(
        '--output', '-o',
        default='new_journal.yml',
        help='Output config filename (default: new_journal.yml)'
    )
    p_jparse.add_argument(
        '--journal', '-j',
        default=None,
        help='Journal name (optional)'
    )
    p_jparse.add_argument(
        '--template', '-t',
        default='template_comprehensive',
        help='Template name without .yml (default: template_comprehensive)'
    )
    p_jparse.add_argument(
        '--save-raw',
        action='store_true',
        help='Save fetched guidelines to doc/journal_guidelines/'
    )
    p_jparse.add_argument(
        '--raw-dir',
        default=None,
        help='Directory to save fetched guidelines'
    )
    p_jparse.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files'
    )

    p_jfetch = sub.add_parser('journal_fetch', help='Fetch journal guidelines')
    p_jfetch.add_argument(
        '--url', '-u',
        required=True,
        help='URL to journal author guidelines'
    )
    p_jfetch.add_argument(
        '--output', '-o',
        default=None,
        help='Output filename (default: slug + extension)'
    )
    p_jfetch.add_argument(
        '--journal', '-j',
        default=None,
        help='Journal name for default filename'
    )
    p_jfetch.add_argument(
        '--raw-dir',
        default=None,
        help='Directory to save guidelines'
    )
    p_jfetch.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files'
    )
    p_jfetch.add_argument(
        '--text',
        action='store_true',
        help='Also save a text-only version'
    )

    # Data Audit Commands
    p_audit = sub.add_parser('audit_data', help='Audit pipeline data files')
    p_audit.add_argument(
        '--full', '-f',
        action='store_true',
        help='Run full audit with detailed column info'
    )
    p_audit.add_argument(
        '--report', '-r',
        action='store_true',
        help='Save markdown report to data_work/diagnostics/'
    )
    p_audit.add_argument(
        '--output', '-o',
        default=None,
        help='Output path for JSON report'
    )

    # Project Migration Commands (AI Agent Tools)
    p_analyze = sub.add_parser('analyze_project', help='Analyze external project structure')
    p_analyze.add_argument(
        '--path', '-p',
        required=True,
        help='Path to project to analyze'
    )
    p_analyze.add_argument(
        '--output', '-o',
        default=None,
        help='Output file for JSON analysis (optional)'
    )

    p_map = sub.add_parser('map_project', help='Map project structure to template')
    p_map.add_argument(
        '--path', '-p',
        required=True,
        help='Path to project to map'
    )
    p_map.add_argument(
        '--output', '-o',
        default=None,
        help='Output file for mapping (optional)'
    )

    p_plan = sub.add_parser('plan_migration', help='Generate migration plan')
    p_plan.add_argument(
        '--path', '-p',
        required=True,
        help='Path to source project'
    )
    p_plan.add_argument(
        '--target', '-t',
        required=True,
        help='Path for migrated project'
    )
    p_plan.add_argument(
        '--output', '-o',
        default=None,
        help='Output file for plan (optional)'
    )

    p_migrate = sub.add_parser('migrate_project', help='Execute migration')
    p_migrate.add_argument(
        '--path', '-p',
        required=True,
        help='Path to source project'
    )
    p_migrate.add_argument(
        '--target', '-t',
        required=True,
        help='Path for migrated project'
    )
    p_migrate.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    # Stage Versioning Commands
    p_run_stage = sub.add_parser('run_stage', help='Run a specific stage by name')
    p_run_stage.add_argument(
        'stage_name',
        help='Stage name (e.g., s00_ingest, s00b_standardize)'
    )

    p_list_stages = sub.add_parser('list_stages', help='List available stage versions')
    p_list_stages.add_argument(
        '--prefix', '-p',
        default=None,
        help='Filter by stage prefix (e.g., s00, s01)'
    )

    return p.parse_args()


def main():
    """Main entry point."""
    ensure_env()
    args = parse_args()

    if args.cmd == 'ingest_data':
        from stages import s00_ingest
        s00_ingest.main(use_demo=args.demo)

    elif args.cmd == 'link_records':
        from stages import s01_link
        s01_link.main()

    elif args.cmd == 'build_panel':
        from stages import s02_panel
        s02_panel.main()

    elif args.cmd == 'run_estimation':
        from stages import s03_estimation
        s03_estimation.main(
            specification=args.specification,
            sample=args.sample
        )

    elif args.cmd == 'estimate_robustness':
        from stages import s04_robustness
        s04_robustness.main()

    elif args.cmd == 'make_figures':
        from stages import s05_figures
        s05_figures.main()

    elif args.cmd == 'validate_submission':
        from stages import s06_manuscript
        s06_manuscript.validate(
            journal=args.journal,
            report=args.report
        )

    # Review Management Commands
    elif args.cmd == 'review_status':
        from stages import s07_reviews
        s07_reviews.status(manuscript=args.manuscript)

    elif args.cmd == 'review_new':
        from stages import s07_reviews
        s07_reviews.new_cycle(manuscript=args.manuscript, focus=args.focus)

    elif args.cmd == 'review_archive':
        from stages import s07_reviews
        s07_reviews.archive(manuscript=args.manuscript)

    elif args.cmd == 'review_verify':
        from stages import s07_reviews
        s07_reviews.verify(manuscript=args.manuscript)

    elif args.cmd == 'review_report':
        from stages import s07_reviews
        s07_reviews.report()

    # Journal Configuration Commands
    elif args.cmd == 'journal_list':
        from stages import s08_journal_parser
        s08_journal_parser.list_configs()

    elif args.cmd == 'journal_validate':
        from stages import s08_journal_parser
        s08_journal_parser.validate_config(args.config)

    elif args.cmd == 'journal_compare':
        from stages import s08_journal_parser
        s08_journal_parser.compare_manuscript(args.journal, args.manuscript)

    elif args.cmd == 'journal_parse':
        from stages import s08_journal_parser
        s08_journal_parser.parse_guidelines(
            input_file=args.input,
            output_file=args.output,
            journal_name=args.journal,
            url=args.url,
            template_name=args.template,
            save_raw=args.save_raw,
            raw_dir=args.raw_dir,
            overwrite=args.overwrite
        )

    elif args.cmd == 'journal_fetch':
        from stages import s08_journal_parser
        s08_journal_parser.fetch_guidelines_cli(
            url=args.url,
            output=args.output,
            journal_name=args.journal,
            raw_dir=args.raw_dir,
            overwrite=args.overwrite,
            save_text=args.text
        )

    # Data Audit Commands
    elif args.cmd == 'audit_data':
        import data_audit
        data_audit.main(
            full=args.full,
            report=args.report,
            output=args.output
        )

    # Project Migration Commands (AI Agent Tools)
    elif args.cmd == 'analyze_project':
        from pathlib import Path
        from agents.project_analyzer import ProjectAnalyzer

        analyzer = ProjectAnalyzer(Path(args.path))
        analysis = analyzer.analyze()

        print(analysis.summary())

        if args.output:
            Path(args.output).write_text(analysis.to_json())
            print(f"\nJSON saved to: {args.output}")

    elif args.cmd == 'map_project':
        from pathlib import Path
        from agents.project_analyzer import ProjectAnalyzer
        from agents.structure_mapper import StructureMapper

        analyzer = ProjectAnalyzer(Path(args.path))
        analysis = analyzer.analyze()

        mapper = StructureMapper(analysis)
        mapping = mapper.generate_mapping()

        print(mapping.summary())

        if args.output:
            Path(args.output).write_text(mapping.to_json())
            print(f"\nJSON saved to: {args.output}")

    elif args.cmd == 'plan_migration':
        from pathlib import Path
        from agents.project_analyzer import ProjectAnalyzer
        from agents.structure_mapper import StructureMapper
        from agents.migration_planner import MigrationPlanner

        analyzer = ProjectAnalyzer(Path(args.path))
        analysis = analyzer.analyze()

        mapper = StructureMapper(analysis)
        mapping = mapper.generate_mapping()

        planner = MigrationPlanner(analysis, mapping)
        plan = planner.generate_plan(args.target)

        print(plan.to_markdown())

        if args.output:
            Path(args.output).write_text(plan.to_markdown())
            print(f"\nPlan saved to: {args.output}")

    elif args.cmd == 'migrate_project':
        from pathlib import Path
        from agents.project_analyzer import ProjectAnalyzer
        from agents.structure_mapper import StructureMapper
        from agents.migration_planner import MigrationPlanner
        from agents.migration_executor import MigrationExecutor

        source_path = Path(args.path)
        target_path = Path(args.target)

        # Get the template path (this project's root)
        template_path = Path(__file__).parent.parent

        print(f"Analyzing source project: {source_path}")
        analyzer = ProjectAnalyzer(source_path)
        analysis = analyzer.analyze()

        print(f"Mapping to template structure...")
        mapper = StructureMapper(analysis)
        mapping = mapper.generate_mapping()

        print(f"Generating migration plan...")
        planner = MigrationPlanner(analysis, mapping)
        plan = planner.generate_plan(str(target_path))

        print(f"\n{'DRY RUN - ' if args.dry_run else ''}Executing migration...")
        print(f"  Source: {source_path}")
        print(f"  Target: {target_path}")
        print(f"  Steps: {len(plan.steps)}")
        print()

        executor = MigrationExecutor(
            plan=plan,
            source_path=source_path,
            template_path=template_path,
            dry_run=args.dry_run,
            verbose=True
        )

        report = executor.execute()

        print()
        print("=" * 60)
        print("MIGRATION COMPLETE" if report.overall_success else "MIGRATION FAILED")
        print("=" * 60)
        print(f"  Successful: {report.success_count}/{len(report.results)}")
        print(f"  Failed: {report.failure_count}/{len(report.results)}")

        # Save execution report
        report_path = target_path / 'MIGRATION_REPORT.md'
        if not args.dry_run and target_path.exists():
            report_path.write_text(report.to_markdown())
            print(f"\nReport saved to: {report_path}")

    # Stage Versioning Commands
    elif args.cmd == 'run_stage':
        run_stage_by_name(args.stage_name)

    elif args.cmd == 'list_stages':
        list_available_stages(args.prefix)


def discover_stages(prefix: str = None) -> list[tuple[str, str]]:
    """
    Discover available stage modules.

    Parameters
    ----------
    prefix : str, optional
        Filter by stage prefix (e.g., 's00', 's01')

    Returns
    -------
    list[tuple[str, str]]
        List of (stage_name, description) tuples
    """
    from pathlib import Path

    stages_dir = Path(__file__).parent / 'stages'
    stages = []

    for f in sorted(stages_dir.glob('s*.py')):
        if f.name.startswith('_'):
            continue

        name = f.stem
        if prefix and not name.startswith(prefix):
            continue

        # Try to extract description from docstring
        try:
            content = f.read_text()
            # Look for Purpose: line in docstring
            import re
            match = re.search(r'Purpose:\s*(.+?)(?:\n|$)', content)
            desc = match.group(1).strip() if match else ''
        except Exception:
            desc = ''

        stages.append((name, desc))

    return stages


def list_available_stages(prefix: str = None) -> None:
    """List available stage modules."""
    print("Available Pipeline Stages")
    print("=" * 60)

    stages = discover_stages(prefix)

    if not stages:
        if prefix:
            print(f"No stages found with prefix '{prefix}'")
        else:
            print("No stages found")
        return

    # Group by stage number
    current_num = None
    for name, desc in stages:
        # Extract stage number (e.g., '00' from 's00_ingest' or 's00b_standardize')
        import re
        match = re.match(r's(\d+)', name)
        num = match.group(1) if match else '??'

        if num != current_num:
            if current_num is not None:
                print()
            current_num = num

        print(f"  {name:<30} {desc}")

    print()
    print(f"Total: {len(stages)} stage(s)")
    print()
    print("Run a stage with: python src/pipeline.py run_stage <stage_name>")


def run_stage_by_name(stage_name: str) -> None:
    """
    Run a stage by its module name.

    Parameters
    ----------
    stage_name : str
        Stage module name (e.g., 's00_ingest', 's00b_standardize')
    """
    from pathlib import Path
    import importlib

    stages_dir = Path(__file__).parent / 'stages'
    stage_file = stages_dir / f'{stage_name}.py'

    if not stage_file.exists():
        print(f"ERROR: Stage '{stage_name}' not found")
        print(f"  Expected file: {stage_file}")
        print()
        print("Available stages:")
        for name, _ in discover_stages():
            print(f"  - {name}")
        return

    # Import and run the stage
    try:
        module = importlib.import_module(f'stages.{stage_name}')

        # Look for main() or a similarly named entry point
        if hasattr(module, 'main'):
            module.main()
        elif hasattr(module, 'validate'):
            # s06_manuscript uses validate()
            module.validate()
        else:
            print(f"ERROR: Stage '{stage_name}' has no main() or validate() function")
    except Exception as e:
        print(f"ERROR running stage '{stage_name}': {e}")
        raise


if __name__ == '__main__':
    main()
