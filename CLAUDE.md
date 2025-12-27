# CENTAUR - Claude Code Instructions

**C**omputational **E**nvironment for **N**avigating **T**asks in **A**utomated **U**niversity **R**esearch

## Quick Start

```bash
source .venv/bin/activate  # REQUIRED for all scripts
```

**Full documentation:** See `doc/README.md` for complete index.

### Common Commands

```bash
# Pipeline stages
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

## Synthetic Peer Review

### Workflow

1. **Generate**: `python src/pipeline.py review_new --focus methods`
2. **Triage**: Classify comments in `manuscript_quarto/REVISION_TRACKER.md`
3. **Track**: Update checklist as changes are made
4. **Verify**: `python src/pipeline.py review_verify` (includes compliance checks)
5. **Archive**: `python src/pipeline.py review_archive`

### Focus Options

**Discipline-based:** economics, engineering, social_sciences, general
**Aspect-based:** methods, policy, clarity

### Multi-Manuscript Support

```bash
python src/pipeline.py review_status -m main     # Check specific manuscript
python src/pipeline.py review_new -m main -f economics
```

Configure manuscripts in `src/config.py` via the `MANUSCRIPTS` dictionary.

See `doc/SYNTHETIC_REVIEW_PROCESS.md` for full methodology.

## Key References

| Topic | Document |
|-------|----------|
| Pipeline stages | `doc/PIPELINE.md` |
| Stage pattern and keywords | `doc/PIPELINE.md` |
| Project migration | `doc/AGENT_TOOLS.md` |
| Skills and actions | `doc/skills.md` |
| Statistical methods | `doc/METHODOLOGY.md` |
| Troubleshooting | `doc/agents.md` |
