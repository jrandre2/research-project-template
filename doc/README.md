# Documentation Index

**Project**: CENTAUR (Computational Environment for Navigating Tasks in Automated University Research)
**Scope**: Research workflow platform (not project-specific analysis)
**Last Updated**: 2025-12-30

---

## New to CENTAUR?

Start here based on your situation:

| I have... | Start here |
|-----------|------------|
| Nothing yet, want to explore | [GETTING_STARTED.md](GETTING_STARTED.md) → Demo Walkthrough |
| Data files ready for analysis | [GETTING_STARTED.md](GETTING_STARTED.md) → Data-First Workflow |
| Manuscript draft, no data yet | [GETTING_STARTED.md](GETTING_STARTED.md) → Manuscript-First Workflow |
| Both data and manuscript | [GETTING_STARTED.md](GETTING_STARTED.md) → Integration Workflow |
| Existing analysis codebase | [AGENT_TOOLS.md](AGENT_TOOLS.md) → Migration Tools |
| Need step-by-step setup | [NEW_PROJECT_CHECKLIST.md](NEW_PROJECT_CHECKLIST.md) |

---

## Quick Start

| Document | Location | Purpose |
|----------|----------|---------|
| **CLAUDE.md** | Root | AI agent project instructions |
| **README.md** | Root | Project overview, setup, key commands |
| **[TUTORIAL.md](TUTORIAL.md)** | doc/ | Step-by-step getting started guide |

---

## Core References

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, data flow, extension points | Understanding the system |
| [PIPELINE.md](PIPELINE.md) | Pipeline stages and CLI commands | Running the pipeline |
| [METHODOLOGY.md](METHODOLOGY.md) | Statistical methods and equations | Methodology review |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | Variable definitions and conventions | Variable lookups |

---

## Guides

| Document | Purpose |
|----------|---------|
| [TUTORIAL.md](TUTORIAL.md) | Getting started with demo data |
| [CUSTOMIZATION.md](CUSTOMIZATION.md) | Adapting the platform to your project |
| [MANUSCRIPT_VARIANTS.md](MANUSCRIPT_VARIANTS.md) | Managing divergent manuscript drafts |
| [REPRODUCTION.md](REPRODUCTION.md) | Running analysis from scratch |
| [ANALYSIS_JOURNEY.md](ANALYSIS_JOURNEY.md) | Research process documentation template |
| [GIT_BRANCHING.md](GIT_BRANCHING.md) | Branching strategy for analysis variants |
| [agents.md](agents.md) | AI agent guidelines |
| [skills.md](skills.md) | Available skills/actions |

---

## Peer Review System

| Document | Purpose |
|----------|---------|
| [SYNTHETIC_REVIEW_PROCESS.md](SYNTHETIC_REVIEW_PROCESS.md) | Review methodology and prompts |
| [MANUSCRIPT_REVISION_CHECKLIST.md](MANUSCRIPT_REVISION_CHECKLIST.md) | High-level revision status |
| [reviews/README.md](reviews/README.md) | Review cycles index |

Supports both **synthetic** (AI-generated) and **actual** (journal) reviews.

**CLI Commands:**

```bash
# Status and reports
python src/pipeline.py review_status [-m main]
python src/pipeline.py review_report

# Synthetic reviews (pre-submission)
python src/pipeline.py review_new --focus methods
python src/pipeline.py review_new -m main -f economics

# Actual reviews (from journals)
python src/pipeline.py review_new --actual --journal "JEEM" --round "R&R1"

# Workflow commands
python src/pipeline.py review_verify [-m main]
python src/pipeline.py review_archive [-m main]
python src/pipeline.py review_diff [-m main]
python src/pipeline.py review_response [-m main]
```

**Focus Options (synthetic):** economics, engineering, social_sciences, general, methods, policy, clarity

---

## Pipeline Commands

### Data Processing

```bash
python src/pipeline.py ingest_data     # Load raw data (use --demo for synthetic data)
python src/pipeline.py link_records    # Link data sources
python src/pipeline.py build_panel     # Construct panel
```

### Analysis

```bash
python src/pipeline.py run_estimation           # Run main estimation
python src/pipeline.py estimate_robustness      # Robustness checks
```

### Output

```bash
python src/pipeline.py make_figures             # Generate figures
python src/pipeline.py validate_submission      # Check manuscript
python src/pipeline.py audit_data [--full]      # Audit data files
```

### Stage Versioning

```bash
python src/pipeline.py list_stages              # List all available stages
python src/pipeline.py list_stages -p s00       # List s00 versions
python src/pipeline.py run_stage s00_ingest     # Run specific stage
```

---

## Configuration & QA

**Configuration:** `src/config.py` centralizes paths, methodological parameters, and QA settings.

**QA Reports:** Each stage generates quality reports in `data_work/quality/`. Controlled by `ENABLE_QA_REPORTS` in config.

**Extended Scripts:** `scripts/` directory for exploratory analyses separate from the core pipeline. See `scripts/README.md`.

---

## Geospatial Analysis

The `src/spatial/` module provides core utilities for working with geographic data.

| Document | Purpose |
|----------|---------|
| [GEOSPATIAL_ANALYSIS.md](GEOSPATIAL_ANALYSIS.md) | Module guide and API reference |
| [design/GEOSPATIAL_MODULE.md](design/GEOSPATIAL_MODULE.md) | Full design and roadmap |

**Quick Usage:**

```python
from spatial import load_spatial, haversine_distance, ensure_crs

gdf = load_spatial('data.gpkg')
gdf = ensure_crs(gdf, 'EPSG:4326')
dist = haversine_distance(40.7, -74.0, 34.1, -118.2)  # NYC to LA
```

**Key Functions:**
- `load_spatial()`, `save_spatial()` - Spatial data I/O (GeoPackage, Shapefile, GeoJSON)
- `haversine_distance()`, `haversine_matrix()` - Distance calculations
- `nearest_neighbor()`, `distance_to_nearest()` - Proximity analysis
- `ensure_crs()`, `to_projected()` - CRS handling

**Dependencies:** `geopandas`, `shapely`, `pyproj` (install with `pip install -r requirements-spatial.txt`)

---

## Journal Configuration

**CLI Commands:**
- `python src/pipeline.py journal_list` - List available configs
- `python src/pipeline.py journal_validate --config natural_hazards` - Validate config
- `python src/pipeline.py journal_compare --journal natural_hazards` - Compare manuscript
- `python src/pipeline.py journal_parse --input guidelines.txt --output new_journal.yml` - Parse guidelines
- `python src/pipeline.py journal_parse --url https://example.com/guidelines --journal "Journal Name" --output journal.yml --save-raw` - Fetch + parse guidelines
- `python src/pipeline.py journal_fetch --url https://example.com/guidelines --journal "Journal Name" --text` - Download guidelines

**Note:** PDF guidelines must be converted to text or HTML before parsing.
Downloads are saved to `doc/journal_guidelines/` by default when using `journal_fetch` or `journal_parse --save-raw`.
Parsing is heuristic; review the generated YAML for completeness.

---

## Status Tracking

| Document | Purpose |
|----------|---------|
| [CHANGELOG.md](CHANGELOG.md) | Change history |

---

## Design Documents

Future feature specifications and roadmap documents.

| Document | Purpose |
|----------|---------|
| [design/MULTILANGUAGE_ANALYSIS.md](design/MULTILANGUAGE_ANALYSIS.md) | R, Stata, Julia analysis engine support |
| [design/QUALITATIVE_MODULE.md](design/QUALITATIVE_MODULE.md) | Qualitative & mixed methods module design |
| [design/GEOSPATIAL_MODULE.md](design/GEOSPATIAL_MODULE.md) | Geospatial analysis & spatial econometrics design |

---

## Testing

| Document | Purpose |
|----------|---------|
| [TESTING.md](TESTING.md) | Test structure, conventions, and examples |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common errors and solutions |
| [STAGE_DEVELOPMENT.md](STAGE_DEVELOPMENT.md) | How to create and extend pipeline stages |

---

## Source Code

| Directory | Purpose |
|-----------|---------|
| `src/pipeline.py` | Main CLI entry point |
| `src/config.py` | Centralized configuration (paths, QA, methodology, LLM) |
| `src/data_audit.py` | Data auditing utilities |
| `src/stages/` | Pipeline and workflow stages (s00-s09) |
| `src/stages/_qa_utils.py` | QA report generation utilities |
| `src/stages/_review_models.py` | Review metadata and comment dataclasses |
| `src/llm/` | LLM provider abstraction (Anthropic, OpenAI) |
| `src/agents/` | Project migration tools |
| `src/utils/` | Shared utilities |
| `scripts/` | Extended analysis scripts |
| `tests/` | Test suite |

### Utilities

| Module | Purpose |
|--------|---------|
| `utils/helpers.py` | Path handling, data I/O, formatters |
| `utils/validation.py` | Data validation framework |
| `utils/figure_style.py` | Matplotlib styling |
| `utils/synthetic_data.py` | Demo data generation |
| `utils/spatial_cv.py` | Spatial cross-validation grouping |
| `utils/cache.py` | Result caching utilities |
| `utils/docx_feedback.py` | DOCX document feedback extraction |

---

## Project Migration (AI Agent Tools)

AI-powered tools for analyzing and migrating external research projects.

| Document | Purpose |
|----------|---------|
| [AGENT_TOOLS.md](AGENT_TOOLS.md) | Agent module reference and API |
| [skills.md](skills.md) | Migration skills (/analyze-project, /map-project, etc.) |

### Agent Modules

| Module | Purpose |
|--------|---------|
| `agents/project_analyzer.py` | Scan and analyze project structures |
| `agents/structure_mapper.py` | Map modules to platform stages |
| `agents/migration_planner.py` | Generate migration plans |
| `agents/migration_executor.py` | Execute migrations |

### CLI Commands

```bash
python src/pipeline.py analyze_project --path /path/to/project
python src/pipeline.py map_project --path /path/to/project
python src/pipeline.py plan_migration --path /source --target /target
python src/pipeline.py migrate_project --path /source --target /target --dry-run
```

---

## Adding New Documentation

When adding a new document:

1. Create the file in `doc/`
2. Add an entry to this index
3. Add a related link header to the new file:

```markdown
**Related**: [File1](file1.md) | [File2](file2.md)
**Status**: Active
**Last Updated**: YYYY-MM-DD
```

## Document Status Legend

- **Active** - In use, regularly updated
- **Template** - Placeholder for project-specific content
- **Reference** - Stable, infrequently changed
- **Archive** - Historical, moved to `archive/`
