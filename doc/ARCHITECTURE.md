# System Architecture

**Related**: [PIPELINE.md](PIPELINE.md) | [README.md](README.md)
**Status**: Active
**Last Updated**: 2025-12-30

---

This document describes the architecture of CENTAUR (Computational Environment for Navigating Tasks in Automated University Research), including data flow, component relationships, and extension points.

## Overview

The system is designed as a modular research pipeline with clear separation between:

1. **Data Processing** - Ingestion, linkage, panel construction
2. **Analysis** - Estimation, robustness checks
3. **Output Generation** - Figures, manuscript validation
4. **Infrastructure** - Utilities, validation, configuration
5. **Workflow Tools** - Review management, journal configuration, migration tools

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   pipeline.py   │    │  Quarto CLI     │    │   pytest        │ │
│  │   (main CLI)    │    │  (manuscript)   │    │   (testing)     │ │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘ │
└───────────┼──────────────────────┼──────────────────────┼───────────┘
            │                      │                      │
┌───────────▼──────────────────────▼──────────────────────▼───────────┐
│                        PIPELINE STAGES                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │s00_ingest│→ │s01_link  │→ │s02_panel │→ │s03_estim │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│                                               ↓                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │s06_manusc│← │s05_figure│← │s04_robust│← │          │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────┐
│                         UTILITIES                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  helpers.py │  │validation.py│  │figure_style │  │synthetic_   │ │
│  │             │  │             │  │     .py     │  │   data.py   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────┐
│                         DATA LAYER                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────────┐ │
│  │  data_raw/  │  │ data_work/  │  │ manuscript_quarto/figures/   │ │
│  │  (input)    │  │ (processed) │  │ (primary figure outputs)     │ │
│  └─────────────┘  └─────────────┘  └──────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

Workflow tools (review management, journal configuration, and migration) sit alongside the core pipeline and operate on `doc/`, `manuscript_quarto/`, and external project directories.

### AI Agent Orchestration (Design)

CENTAUR is designed to be operated by AI agents of varying capabilities at different supervision levels. The orchestration layer (see [design/AI_AGENT_ORCHESTRATION.md](design/AI_AGENT_ORCHESTRATION.md)) specifies:

- **Graduated supervision**: Five levels from fully human (0) to autonomous (4)
- **Multi-model routing**: Provider-agnostic capability tiers (simple/standard/complex)
- **Operator patterns**: Human, AI assistant, autonomous agent, multi-agent coordination

This enables the same pipeline to be operated by:
- A human researcher running CLI commands directly
- An AI assistant (Claude Code) executing on the human's behalf
- An autonomous agent running scheduled tasks

## Directory Structure

For the complete project structure, see [README.md](../README.md#project-structure) in the project root.

**Key directories:**

| Directory | Purpose |
|-----------|---------|
| `src/stages/` | Pipeline stages (s00-s09) |
| `src/utils/` | Shared utilities (helpers, validation, cache, spatial_cv, etc.) |
| `src/llm/` | LLM provider abstraction |
| `src/spatial/` | Geospatial analysis module |
| `src/agents/` | Project migration tools |
| `data_raw/` | Raw input data (gitignored) |
| `data_work/` | Processed data and diagnostics (gitignored) |
| `manuscript_quarto/` | Quarto manuscript and figures |

## Data Flow

### Stage Dependencies

```
data_raw/input.csv
        │
        ▼
┌───────────────────┐
│  s00_ingest.py    │  Load, clean, validate
│  → data_raw.parquet
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  s01_link.py      │  Match records across sources
│  → data_linked.parquet + diagnostics/linkage_summary.csv
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  s02_panel.py     │  Construct analysis panel
│  → panel.parquet + diagnostics/panel_summary.csv
└─────────┬─────────┘
          │
          ├────────────────────────────┐
          ▼                            ▼
┌───────────────────┐        ┌───────────────────┐
│  s03_estimation   │        │  s04_robustness   │
│  → diagnostics/   │        │  → diagnostics/   │
│     estimation_   │        │     robustness_   │
│     results.csv   │        │     results.csv   │
│  → diagnostics/   │        │  → diagnostics/   │
│     coefficients  │        │     placebo_      │
│     .csv          │        │     results.csv   │
└─────────┬─────────┘        └─────────┬─────────┘
          │                            │
          └──────────┬─────────────────┘
                     ▼
          ┌───────────────────┐
          │  s05_figures.py   │  Generate plots
          │  → manuscript_quarto/figures/*.png  │
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │  s06_manuscript   │  Validate submission
          │  → diagnostics/   │
          │     submission_   │
          │     validation.md │
          └───────────────────┘
```

Review management and journal configuration are workflow tools that do not depend on data flow outputs, but they do read from and write to `manuscript_quarto/` and `doc/`.

### File Naming Conventions

| Stage | Input | Output |
|-------|-------|--------|
| s00 | `data_raw/*.csv` | `data_work/data_raw.parquet` |
| s01 | `data_work/data_raw.parquet` | `data_work/data_linked.parquet`<br>`data_work/diagnostics/linkage_summary.csv` |
| s02 | `data_work/data_linked.parquet` | `data_work/panel.parquet`<br>`data_work/diagnostics/panel_summary.csv` |
| s03 | `data_work/panel.parquet` | `data_work/diagnostics/estimation_results.csv`<br>`data_work/diagnostics/coefficients.csv` |
| s04 | `data_work/panel.parquet` | `data_work/diagnostics/robustness_results.csv`<br>`data_work/diagnostics/placebo_results.csv` |
| s05 | `data_work/panel.parquet` | `manuscript_quarto/figures/*.png` |
| s06 | `manuscript_quarto/*.qmd` | `data_work/diagnostics/submission_validation.md` |

## Component Relationships

### CLI Commands → Stages

```python
# src/pipeline.py command routing
COMMANDS = {
    'ingest_data': 's00_ingest.main()',
    'link_records': 's01_link.main()',
    'build_panel': 's02_panel.main()',
    'run_estimation': 's03_estimation.main()',
    'estimate_robustness': 's04_robustness.main()',
    'make_figures': 's05_figures.main()',
    'validate_submission': 's06_manuscript.validate()',
    'review_new': 's07_reviews.new_cycle()',
    'review_status': 's07_reviews.status()',
    'review_verify': 's07_reviews.verify()',
    'review_archive': 's07_reviews.archive()',
    'review_report': 's07_reviews.report()',
    'journal_list': 's08_journal_parser.list_configs()',
    'journal_validate': 's08_journal_parser.validate_config()',
    'journal_compare': 's08_journal_parser.compare_manuscript()',
    'journal_fetch': 's08_journal_parser.fetch_guidelines_cli()',
    'journal_parse': 's08_journal_parser.parse_guidelines()',
    'draft_results': 's09_writing.draft_results()',
    'draft_captions': 's09_writing.draft_captions()',
    'draft_abstract': 's09_writing.draft_abstract()',
    'audit_data': 'data_audit.main()',
    'cache': 'cache_management()',
}
```


### Utility Dependencies

```
stages/s00_ingest.py
    ├── utils/helpers.py (load_data, save_data, get_data_dir)
    ├── utils/validation.py (DataValidator)
    └── utils/synthetic_data.py (SyntheticDataGenerator)

stages/s03_estimation.py
    ├── utils/helpers.py (load_data, save_diagnostic, add_significance_stars)
    └── numpy (matrix operations)

stages/s05_figures.py
    ├── utils/helpers.py (load_data, get_figures_dir)
    ├── utils/figure_style.py (apply_style, get_color_palette, save_figure)
    └── matplotlib.pyplot

stages/s06_manuscript.py
    ├── utils/helpers.py (get_project_root, load_config)
    └── manuscript_quarto/journal_configs/*.yml
```

## Extension Points

### 1. Adding New Pipeline Stages

Create a new stage module in `src/stages/`:

```python
# src/stages/s07_new_stage.py
"""
Stage 07: New Stage Name

Purpose: Description of what this stage does.

Input Files
-----------
- data_work/previous_output.parquet

Output Files
------------
- data_work/new_output.parquet
- data_work/diagnostics/new_diagnostics.csv

Usage
-----
    python src/pipeline.py new_command
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import get_data_dir, load_data, save_data

def main(verbose: bool = True):
    """Execute the stage."""
    print("=" * 60)
    print("Stage 07: New Stage Name")
    print("=" * 60)

    # Implementation

    print("Stage 07 complete.")

if __name__ == '__main__':
    main()
```

Register in `src/pipeline.py`:

```python
# Add subparser
p_new = sub.add_parser('new_command', help='Description')
p_new.add_argument('--option', '-o', help='Option description')

# Add handler in main()
elif args.cmd == 'new_command':
    from stages import s07_new_stage
    s07_new_stage.main()
```

### 2. Adding Validation Rules

Extend `src/utils/validation.py`:

```python
# Add new rule type
def value_pattern(column: str, pattern: str) -> ValidationRule:
    """Check that values match a regex pattern."""
    import re
    compiled = re.compile(pattern)

    def check(df):
        return df[column].astype(str).str.match(compiled).all()

    return ValidationRule(
        name=f'{column}_pattern',
        check=check,
        severity='error',
        message=f'{column} values must match pattern: {pattern}'
    )
```

### 3. Adding Figure Types

Extend `src/stages/s05_figures.py`:

```python
def plot_new_figure(
    df: pd.DataFrame,
    output_path: Path,
    **kwargs
) -> Path:
    """Create a new figure type."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Your plotting code

    save_figure(fig, output_path, formats=['png'])
    plt.close(fig)
    return output_path

# Register in all_figures dict
all_figures = {
    # ... existing figures ...
    'new_figure': lambda: plot_new_figure(df, fig_dir / 'fig_new'),
}
```

### 4. Adding Journal Configurations

Create a new journal config in `manuscript_quarto/journal_configs/`:

```yaml
# new_journal.yml
journal:
  name: "New Journal"
  abbreviation: "NJ"
  publisher: "Publisher Name"

abstract:
  max_words: 200

text_limits:
  word_limit: 8000

artwork:
  formats:
    raster_acceptable: ["PNG", "TIFF"]
  resolution:
    min_dpi: 300
```

Create matching Quarto profile in `manuscript_quarto/_quarto-newjournal.yml`.

### 5. Adding Estimation Specifications

Extend `src/stages/s03_estimation.py`:

```python
SPECIFICATIONS = {
    # ... existing specs ...
    'new_spec': {
        'name': 'New Specification',
        'outcome': 'outcome',
        'treatment': 'treatment',
        'controls': ['new_control1', 'new_control2'],
        'fe': ['unit_id', 'time_id'],
        'cluster': 'unit_id',
        'description': 'Description of this specification'
    },
}
```

## Configuration

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `PROJECT_ROOT` | Override project root detection | Auto-detect |
| `DATA_DIR` | Override data directory | `{PROJECT_ROOT}/data_work` |

### Project Configuration

The `utils/helpers.py` module provides centralized path configuration:

```python
def get_project_root() -> Path:
    """Find project root by looking for marker files."""

def get_data_dir(subdir: str = 'work') -> Path:
    """Get data directory path."""

def get_figures_dir() -> Path:
    """Get figures output directory."""
```

### Journal Profiles

Quarto rendering uses journal-specific profiles:

```bash
# Render with JEEM profile
quarto render --profile jeem

# Render with custom journal
quarto render --profile custom_journal
```

Profile files (`_quarto-<journal>.yml`) override base settings in `_quarto.yml`.

## Testing Strategy

### Unit Tests

Located in `tests/test_utils/`:
- Test individual utility functions
- Mock file I/O where appropriate
- Use fixtures for sample data

### Stage Tests

Located in `tests/test_stages/`:
- Test stage main functions
- Use synthetic data from fixtures
- Verify output file creation

### Integration Tests

Located in `tests/test_pipeline.py`:
- Test CLI argument parsing
- Test command routing
- Verify stage dependencies

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_utils/test_helpers.py

# With coverage
pytest --cov=src tests/
```

## Performance Considerations

### Large Datasets

- Use parquet format for columnar storage
- Process in chunks when memory-constrained
- Leverage pandas query optimization

### Parallel Execution

The pipeline includes built-in parallel execution for CPU-intensive stages:

- `ProcessPoolExecutor` for estimation and robustness (CPU-bound)
- `ThreadPoolExecutor` for figure generation (I/O-bound)
- Configure via `PARALLEL_ENABLED` and `PARALLEL_MAX_WORKERS` in `src/config.py`
- Disable with `--sequential` flag or `--workers 1`

### Caching

Built-in result caching dramatically improves re-run performance:

- Cache stored in `data_work/.cache/<stage_name>/`
- Automatic invalidation on data or config changes
- Manage with `python src/pipeline.py cache stats` and `cache clear`
- Disable with `--no-cache` flag for fresh computation
- Configure via `CACHE_ENABLED` and `CACHE_MAX_AGE_HOURS` in `src/config.py`

### Spatial Operations

The `src/spatial/` module has specific performance considerations:

- **Distance matrices**: O(n²) memory - use `max_distance` filter for large datasets
- **Spatial indexes**: GeoPandas automatically uses R-tree for `sjoin()` operations
- **CRS projection**: Project to UTM for accurate distance/area calculations
- **File formats**: GeoPackage (.gpkg) is faster than Shapefile for large datasets

See [GEOSPATIAL_ANALYSIS.md](GEOSPATIAL_ANALYSIS.md#performance-guidelines) for detailed guidance.

## Error Handling

### Validation Errors

The validation framework distinguishes severity levels:
- **error**: Processing cannot continue
- **warning**: Review recommended, processing continues
- **info**: Informational, no action needed

### Stage Failures

Each stage should:
1. Check for required inputs before processing
2. Provide clear error messages with file paths
3. Suggest remediation steps (e.g., "Run previous stage first")

### Recovery

- Stages are idempotent (can be re-run safely)
- Partial outputs are overwritten on re-run
- Use diagnostics to identify failure points
