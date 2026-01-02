# CENTAUR - Claude Code Instructions

**C**omputational **E**nvironment for **N**avigating **T**asks in **A**utomated **U**niversity **R**esearch

## Quick Start

```bash
source .venv/bin/activate  # REQUIRED for all scripts
```

**Full documentation:** See `doc/README.md` for complete index.

**Starting a new project?** See `doc/GETTING_STARTED.md` for scenario-based guidance.

### Common Commands

```bash
# Pipeline stages (use --demo if data_raw/ is empty)
python src/pipeline.py ingest_data
python src/pipeline.py run_estimation --specification baseline
python src/pipeline.py make_figures

# Stage versioning
python src/pipeline.py list_stages                # List all stages
python src/pipeline.py run_stage s00_ingest       # Run specific stage

# Render manuscript
cd manuscript_quarto && ./render_all.sh

# Synthetic review (multi-manuscript support)
python src/pipeline.py review_new -f methods              # Start review
python src/pipeline.py review_new -m main -f economics    # Specify manuscript
python src/pipeline.py review_verify                      # Run verification
```

## Configuration Module

Central configuration in `src/config.py`:

```python
from config import (
    PROJECT_ROOT, DATA_WORK_DIR, DATA_RAW_DIR,  # Paths
    ENABLE_QA_REPORTS, QA_REPORTS_DIR,           # QA settings
    SIGNIFICANCE_LEVEL, BOOTSTRAP_ITERATIONS,   # Methodological params
    MANUSCRIPTS, DEFAULT_MANUSCRIPT,             # Multi-manuscript
)
```

## Versioned Stage Pattern

Stages can evolve with version suffixes: `s00_ingest` → `s00b_standardize` → `s00c_enhanced`

```bash
python src/pipeline.py list_stages -p s00    # List all s00 versions
python src/pipeline.py run_stage s00b_standardize  # Run specific version
```

## QA Reports

Each stage generates quality reports in `data_work/quality/`:

```text
data_work/quality/s00_ingest_quality_20251227_143022.csv
```

Controlled by `ENABLE_QA_REPORTS` in `src/config.py`.

## Caching and Parallel Execution

Pipeline stages support caching and parallel execution for faster re-runs:

```bash
# Default: caching and parallel execution enabled
python src/pipeline.py run_estimation --all

# Disable caching (force recomputation)
python src/pipeline.py run_estimation --all --no-cache

# Disable parallel execution
python src/pipeline.py run_estimation --all --sequential

# Set specific worker count
python src/pipeline.py run_estimation --all --workers 4

# Cache management
python src/pipeline.py cache stats    # Show cache size
python src/pipeline.py cache clear    # Clear all caches
python src/pipeline.py cache clear -s s03_estimation  # Clear specific stage
```

**Cache location:** `data_work/.cache/<stage_name>/`

**Configuration in `src/config.py`:**

```python
CACHE_ENABLED = True          # Enable/disable caching
PARALLEL_ENABLED = True       # Enable/disable parallel execution
PARALLEL_MAX_WORKERS = None   # Max workers (None = CPU count)
```

## Extended Scripts Directory

Exploratory analyses separate from core pipeline in `scripts/`:

```text
scripts/
├── README.md              # Documentation
├── run_example.py         # Template script
└── run_<analysis_name>.py # Custom analyses
```

Use scripts for one-off analyses that shouldn't be part of the reproducible pipeline.

## Critical Constraints

### DO NOT

- Modify raw data in `data_raw/`
- Modify source projects during migration (copy only)
- Execute migrations without testing with `--dry-run` first

### ALWAYS

- Activate `.venv` before running scripts
- Run diagnostics after estimation changes
- Re-render Quarto after modifying `.qmd` files

## Manuscript Writing Standards

**DO NOT include in manuscript prose:**

- References to Python scripts or file paths (e.g., `script.py`, `src/...`)
- Internal documentation references
- Metacommentary about the writing process
- TODO/FIXME placeholders

**All manuscript text should be:**

- Self-contained academic prose
- Supported by formal citations where needed
- Free of implementation details visible only to developers

## Peer Review Management

Supports both **synthetic** (AI-generated) and **actual** (journal) peer reviews.

### Workflow

1. **Generate**: `python src/pipeline.py review_new --focus methods`
2. **Triage**: Classify comments in `manuscript_quarto/REVISION_TRACKER.md`
3. **Track**: Update checklist as changes are made
4. **Verify**: `python src/pipeline.py review_verify` (includes compliance checks)
5. **Archive**: `python src/pipeline.py review_archive`

### Synthetic vs Actual Reviews

```bash
# Synthetic review (default)
python src/pipeline.py review_new -m main -f economics

# Actual journal review
python src/pipeline.py review_new -m main --actual \
    --journal "JEEM" --round "R&R1" --reviewers R1 R2
```

### Focus Options (Synthetic)

**Discipline-based:** economics, engineering, social_sciences, general
**Aspect-based:** methods, policy, clarity

### New Commands

```bash
# Generate diff between review cycles
python src/pipeline.py review_diff -m main
python src/pipeline.py review_diff -m main --commit abc123

# Generate response letter
python src/pipeline.py review_response -m main

# Archive with git tag (default)
python src/pipeline.py review_archive -m main
python src/pipeline.py review_archive -m main --no-tag
```

### Multi-Manuscript Support

```bash
python src/pipeline.py review_status -m main     # Check specific manuscript
python src/pipeline.py review_new -m main -f economics
```

Configure manuscripts in `src/config.py` via the `MANUSCRIPTS` dictionary.

See `doc/SYNTHETIC_REVIEW_PROCESS.md` for full methodology.

## GUI Dashboard

A lightweight web interface for monitoring and supervision. **CLI remains primary interface.**

```bash
# Start the GUI server
python src/pipeline.py gui                    # Default: http://127.0.0.1:8000
python src/pipeline.py gui --port 8001        # Custom port
python src/pipeline.py gui --no-reload        # Disable hot reload
```

**Features:**

| Page | Purpose |
|------|---------|
| Pipeline Dashboard | View stage status, run stages, see QA metrics, manage cache |
| Review Tracker | Kanban board for peer review comments, drag-drop status updates |
| Supervision Controls | Set AI autonomy levels (0-4), approve pending actions, audit log |

**Tech stack:** FastAPI + HTMX + Alpine.js + Tailwind CSS (CDN-served, no build step)

**Supervision levels:**

- Level 0: Human only - no AI assistance
- Level 1: AI suggests - human approves all actions
- Level 2: AI proposes - human reviews before execution
- Level 3: AI executes - human monitors results
- Level 4: Fully autonomous - human handles exceptions only

Settings are stored in `.centaur/supervision.json`.

## AI-Assisted Drafting

Generate draft manuscript sections from pipeline outputs (requires LLM API key).

```bash
# Draft results section from estimation table
python src/pipeline.py draft_results --table main_results
python src/pipeline.py draft_results --table main_results --dry-run  # Preview prompt

# Generate figure captions
python src/pipeline.py draft_captions --figure "fig_*.png"

# Synthesize abstract from manuscript
python src/pipeline.py draft_abstract --max-words 200

# Use alternative provider
python src/pipeline.py draft_results --table main_results --provider openai
```

**Output:** `manuscript_quarto/drafts/` - All drafts require human review before integration.

**Configuration in `src/config.py`:**

```python
LLM_PROVIDER = 'anthropic'  # or 'openai'
LLM_TEMPERATURE = 0.3
```

**Environment variables:** `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

See `doc/skills.md` for full command reference.

## Geospatial Analysis

The `src/spatial/` module provides utilities for geographic data analysis.

```python
from spatial import load_spatial, haversine_distance, ensure_crs

# Load and prepare spatial data
gdf = load_spatial('data.gpkg')
gdf = ensure_crs(gdf, 'EPSG:4326')

# Calculate distances
dist = haversine_distance(40.7, -74.0, 34.1, -118.2)  # NYC to LA in meters
```

**Key functions:**
- `load_spatial()`, `save_spatial()` - Spatial I/O (GeoPackage, Shapefile, GeoJSON)
- `haversine_distance()`, `haversine_matrix()` - Distance calculations
- `nearest_neighbor()`, `distance_to_nearest()` - Proximity analysis
- `ensure_crs()`, `to_projected()` - CRS handling

See `doc/GEOSPATIAL_ANALYSIS.md` for full documentation.

## Web Dashboard (GUI)

Launch a lightweight web dashboard for human supervision of AI-operated pipelines.

```bash
# Start the dashboard (default: http://127.0.0.1:8000)
python src/pipeline.py gui

# Custom host/port
python src/pipeline.py gui --host 0.0.0.0 --port 8080

# Disable auto-reload (production)
python src/pipeline.py gui --no-reload
```

**Features:**

- **Pipeline Dashboard**: View stage status, run stages, see QA metrics
- **Review Tracker**: Kanban view of peer review comments
- **Supervision Controls**: Set AI autonomy level (0-4), approve pending actions

**Tech stack:** FastAPI + HTMX + Alpine.js (no build step, CDN-served)

**Note:** CLI remains the primary interface. GUI is for human monitoring and oversight.

## Multilanguage Analysis

Run estimations using Python or R engines.

```bash
# Check available engines
python src/pipeline.py engines list
python src/pipeline.py engines check

# Run estimation with specific engine
python src/pipeline.py run_estimation --engine r --specification baseline
```

**Supported engines:**

- `python` (default): Native NumPy implementation
- `r`: Uses fixest package for fast fixed effects

**Configuration in `src/config.py`:**

```python
ANALYSIS_ENGINE = 'python'  # Default engine
R_EXECUTABLE = 'Rscript'
```

See `doc/MULTILANGUAGE_SETUP.md` for installation and setup.

## Key References

| Topic | Document |
|-------|----------|
| Pipeline stages | `doc/PIPELINE.md` |
| Stage pattern and keywords | `doc/PIPELINE.md` |
| Project migration | `doc/AGENT_TOOLS.md` |
| Skills and actions | `doc/skills.md` |
| Statistical methods | `doc/METHODOLOGY.md` |
| Troubleshooting | `doc/agents.md` |
| Geospatial analysis | `doc/GEOSPATIAL_ANALYSIS.md` |
| AI agent orchestration | `doc/design/AI_AGENT_ORCHESTRATION.md` |
| Multilanguage setup | `doc/MULTILANGUAGE_SETUP.md` |
| Multilanguage design | `doc/design/MULTILANGUAGE_ANALYSIS.md` |
