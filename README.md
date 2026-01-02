<p align="center">
  <img src="assets/centaur_logo.png?v=3" alt="CENTAUR Logo" width="400">
  <br>
  <strong>C.E.N.T.A.U.R.</strong>
  <br>
  <em>Research, Unbridled</em>
</p>

# CENTAUR

**C**omputational **E**nvironment for **N**avigating **T**asks in **A**utomated **U**niversity **R**esearch

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-blue.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0)
[![Docs License: CC BY-NC 4.0](https://img.shields.io/badge/Docs%20License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Workflow platform for building research pipelines, validating manuscripts, and migrating legacy projects into a standardized structure. This repository provides a complete research workflow platform, not a project-specific analysis.

Licensing: code is PolyForm Noncommercial 1.0.0 and documentation/manuscript content is CC BY-NC 4.0.

## What This Platform Provides

- Modular data and analysis pipeline (ingest, link, panel, estimate, robustness, figures)
- Quarto manuscript with journal profiles and validation checks
- Synthetic peer review workflow and tracking
- Journal configuration parsing and comparison tools (heuristic parsing + URL fetch)
- Project migration tools for onboarding legacy codebases
- Data audit utilities for sample attrition and diagnostics
- Spatial cross-validation for geographic data (optional geopandas dependency)

**Multilanguage Analysis:** R support available (Python default). See [doc/MULTILANGUAGE_SETUP.md](doc/MULTILANGUAGE_SETUP.md) for setup guide.

## Workflow Notes

Journal configuration is optional. If you have not set a journal config yet, skip `validate_submission` and render with Quarto only. When a journal config is available, `validate_submission` checks word count (when configured), abstract length, required sections, figure formats, and bibliography/citation presence.

Review/versioning is handled through `review_new`, `review_verify`, and `review_archive`, with `manuscript_quarto/REVISION_TRACKER.md` and `doc/reviews/archive/` preserving review cycles and responses.

Divergent manuscript drafts can be managed as variants in `manuscript_quarto/variants/` (see `doc/MANUSCRIPT_VARIANTS.md`).

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: Install spatial analysis dependencies
pip install -r requirements-spatial.txt

# Run the pipeline (use --demo for synthetic data, or add real data to data_raw/)
python src/pipeline.py ingest_data --demo
python src/pipeline.py link_records
python src/pipeline.py build_panel
python src/pipeline.py run_estimation --specification baseline
python src/pipeline.py estimate_robustness
python src/pipeline.py make_figures

# Render manuscript
cd manuscript_quarto && ./render_all.sh
```

## Minimal Demo

See `demo/README.md` for a small sample dataset and expected outputs that exercise the full data → manuscript path.

## Using This Platform for a New Project

**New users:** See [doc/GETTING_STARTED.md](doc/GETTING_STARTED.md) for scenario-based guidance.

### Fork vs. Copy

- **Fork** if you plan to contribute improvements back to CENTAUR, or want to pull future updates
- **Copy** (clone + remove `.git`) if you want a completely independent project with fresh git history

### Quick Start Steps

1. Copy or fork this repository as the starting point for a new project
2. Update `CLAUDE.md`, `README.md`, and `doc/` to reflect the new project
3. Configure stage constants in `src/stages/` and define variables in `doc/DATA_DICTIONARY.md`
4. Add raw data to `data_raw/` and run the pipeline
5. Customize journal profiles in `manuscript_quarto/journal_configs/` and write the manuscript

**Detailed checklist:** See [doc/NEW_PROJECT_CHECKLIST.md](doc/NEW_PROJECT_CHECKLIST.md) for step-by-step setup.

## Project Structure

```
├── CLAUDE.md              # AI assistant instructions
├── README.md              # This file
├── requirements.txt       # Python dependencies
│
├── doc/                   # Documentation
│   ├── README.md          # Documentation index
│   ├── PIPELINE.md        # Pipeline stages and CLI
│   ├── ARCHITECTURE.md    # System design
│   ├── METHODOLOGY.md     # Methods template
│   ├── DATA_DICTIONARY.md # Variable definitions template
│   ├── SYNTHETIC_REVIEW_PROCESS.md
│   ├── agents.md          # AI agent guidelines
│   └── skills.md          # Available skills/actions
│
├── demo/                  # Minimal end-to-end demo
│   ├── README.md          # Demo steps and expected outputs
│   └── sample_data.csv    # Small sample dataset
│
├── src/                   # Pipeline and tools
│   ├── pipeline.py        # CLI entry point
│   ├── data_audit.py      # Data audit utilities
│   ├── stages/            # Pipeline and workflow stages (s00-s09)
│   │   ├── s00_ingest.py
│   │   ├── s01_link.py
│   │   ├── s02_panel.py
│   │   ├── s03_estimation.py
│   │   ├── s04_robustness.py
│   │   ├── s05_figures.py
│   │   ├── s06_manuscript.py
│   │   ├── s07_reviews.py
│   │   ├── s08_journal_parser.py
│   │   └── s09_writing.py     # AI-assisted manuscript drafting
│   ├── llm/               # LLM provider abstraction
│   │   ├── anthropic.py   # Claude provider
│   │   ├── openai.py      # GPT-4 provider
│   │   └── prompts.py     # Prompt templates
│   ├── analysis/          # Multilanguage analysis engines
│   │   ├── base.py        # Engine protocol and EstimationResult
│   │   ├── factory.py     # Engine registry
│   │   └── engines/       # Python, R engine implementations
│   ├── agents/            # Project migration tools
│   │   ├── project_analyzer.py
│   │   ├── structure_mapper.py
│   │   ├── migration_planner.py
│   │   └── migration_executor.py
│   └── utils/
│       ├── figure_style.py
│       ├── helpers.py
│       ├── spatial_cv.py       # Spatial cross-validation (optional geopandas)
│       ├── synthetic_data.py
│       └── validation.py
│
├── manuscript_quarto/     # Quarto manuscript
│   ├── _quarto.yml        # Main configuration
│   ├── _quarto-jeem.yml   # JEEM profile
│   ├── _quarto-aer.yml    # AER profile
│   ├── render_all.sh      # Multi-format render script
│   ├── index.qmd          # Main manuscript
│   ├── appendix-*.qmd     # Appendices
│   ├── references.bib     # Bibliography
│   ├── variants/          # Divergent manuscript drafts
│   ├── journal_configs/   # Journal metadata
│   └── code/              # Rendering utilities
│
├── data_raw/              # Raw data (gitignored)
├── data_work/             # Working data (gitignored)
├── figures/               # Optional export figures (not used by default)
└── tools/
    └── bin/quarto         # Quarto wrapper
```

## Key Commands

### Pipeline

| Command | Purpose |
|---------|---------|
| `python src/pipeline.py ingest_data` | Load raw data (fails if empty; use `--demo` for synthetic data) |
| `python src/pipeline.py link_records` | Link data sources |
| `python src/pipeline.py build_panel` | Create analysis panel |
| `python src/pipeline.py run_estimation --specification baseline` | Run estimation |
| `python src/pipeline.py estimate_robustness` | Robustness checks |
| `python src/pipeline.py make_figures` | Generate figures |
| `python src/pipeline.py validate_submission --journal jeem --report` | Validate manuscript |

### Review Management

- `python src/pipeline.py review_status`
- `python src/pipeline.py review_new --focus economics`
- `python src/pipeline.py review_verify`
- `python src/pipeline.py review_archive`
- `python src/pipeline.py review_report`

### Journal Tools

- `python src/pipeline.py journal_list`
- `python src/pipeline.py journal_validate --config natural_hazards`
- `python src/pipeline.py journal_compare --journal natural_hazards`
- `python src/pipeline.py journal_parse --input guidelines.txt --output new_journal.yml`
- `python src/pipeline.py journal_parse --url https://example.com/guidelines --journal "Journal Name" --output journal.yml --save-raw`
- `python src/pipeline.py journal_fetch --url https://example.com/guidelines --journal "Journal Name" --text`

Note: PDF guidelines must be converted to text or HTML before parsing.
Default download location is `doc/journal_guidelines/` (when using `journal_fetch` or `journal_parse --save-raw`).
Parsing is heuristic; review the generated YAML for completeness.

### Data Audit

- `python src/pipeline.py audit_data --full --report`

### AI-Assisted Drafting

Generate draft manuscript sections using LLMs (requires API key):

- `python src/pipeline.py draft_results --table main_results`
- `python src/pipeline.py draft_results --table main_results --dry-run`
- `python src/pipeline.py draft_captions --figure "fig_*.png"`
- `python src/pipeline.py draft_abstract --max-words 200`

Configure provider in `src/config.py` (anthropic or openai). Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` environment variable.

### Analysis Engines

- `python src/pipeline.py engines list` - List available engines
- `python src/pipeline.py engines check` - Validate engine installations
- `python src/pipeline.py run_estimation --engine r` - Use R/fixest for estimation

### Cache Management

- `python src/pipeline.py cache stats`
- `python src/pipeline.py cache clear`
- `python src/pipeline.py cache clear -s s03_estimation`

### Project Migration

- `python src/pipeline.py analyze_project --path /path/to/project`
- `python src/pipeline.py map_project --path /path/to/project`
- `python src/pipeline.py plan_migration --path /source --target /target`
- `python src/pipeline.py migrate_project --path /source --target /target --dry-run`

## Journal Profiles

Render with journal-specific profiles:

```bash
cd manuscript_quarto
./render_all.sh                    # Default (HTML + PDF + DOCX)
./render_all.sh --profile jeem     # JEEM submission format
./render_all.sh --profile aer      # AER submission format
```

If Quarto is not installed system-wide, use the wrapper at `tools/bin/quarto`.

## Documentation

See `doc/README.md` for the full documentation index.

## License

Copyright (c) 2025 Jesse R. Andrews <jesseand@ttu.edu>

Code in this repository is licensed under the PolyForm Noncommercial License 1.0.0 (see `LICENSE`).
Documentation and manuscript content (including `doc/` and `manuscript_quarto/`) are licensed under CC BY-NC 4.0 (see `LICENSE-DOCS`).
