# [Project Name] - Claude Code Instructions

## Quick Start

```bash
source .venv/bin/activate  # REQUIRED for all scripts
```

### Common Commands

```bash
# Run pipeline stages
python src/pipeline.py ingest_data
python src/pipeline.py run_estimation --specification baseline
python src/pipeline.py make_figures

# Render manuscript
cd manuscript_quarto && ./render_all.sh
cd manuscript_quarto && ./render_all.sh --profile jeem  # Journal-specific

# Synthetic review management
python src/pipeline.py review_status              # Check status
python src/pipeline.py review_new -d economics    # Start new review
python src/pipeline.py review_verify              # Run verification
python src/pipeline.py review_archive             # Archive completed
```

## Key Concepts

### [Your Domain Concepts]

[Describe the key conceptual framework for your research. Example:]

1. **[Concept 1]** - [Description]
2. **[Concept 2]** - [Description]

### Data Conventions

- [Describe key variable naming conventions]
- [Describe signed distance or treatment conventions if applicable]
- [Describe event time conventions if applicable]

## Critical Constraints

### DO NOT

- Modify raw data in `data_raw/`
- [Add project-specific constraints]

### ALWAYS

- Activate `.venv` before running scripts
- Run diagnostics after estimation changes
- Re-render Quarto after modifying `.qmd` files

## Data Files

| File | Purpose |
|------|---------|
| `data_work/panel.parquet` | Main analysis panel |
| `data_work/diagnostics/*.csv` | Estimation diagnostics |
| [Add your key data files] | |

## Manuscript

Location: `manuscript_quarto/`

### Rendering All Formats

**Problem:** Quarto book projects overwrite `_output/` on each render.

**Solution:** Use `render_all.sh`:

```bash
cd manuscript_quarto
./render_all.sh                    # All formats (HTML, PDF, DOCX)
./render_all.sh --profile jeem     # JEEM submission format
./render_all.sh --profile aer      # AER submission format
```

Output files in `manuscript_quarto/_output/`:
- `index.html` (+ appendix HTMLs)
- `[ProjectName].pdf`
- `[ProjectName].docx`

### Manuscript Writing Standards

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

Use synthetic reviews to stress-test methodology before submission.

### Workflow

1. **Generate**: `python src/pipeline.py review_new --discipline economics`
2. **Triage**: Classify comments in `manuscript_quarto/REVISION_TRACKER.md`
3. **Track**: Update checklist as changes are made
4. **Verify**: `python src/pipeline.py review_verify`
5. **Archive**: `python src/pipeline.py review_archive`

### Status Classifications

- `VALID - ACTION NEEDED` - Requires changes
- `ALREADY ADDRESSED` - Already handled
- `BEYOND SCOPE` - Valid but deferred
- `INVALID` - Reviewer misunderstanding

### Discipline Templates

- `economics` - Identification, causal inference, econometrics
- `engineering` - Reproducibility, benchmarks, validation
- `social_sciences` - Theory, generalizability, ethics
- `general` - Structure, clarity, contribution

See `doc/SYNTHETIC_REVIEW_PROCESS.md` for full methodology.

## Documentation

See `doc/README.md` for complete index:

- `doc/PIPELINE.md` - Pipeline stages and commands
- `doc/METHODOLOGY.md` - Statistical methods
- `doc/DATA_DICTIONARY.md` - Variable definitions
- `doc/SYNTHETIC_REVIEW_PROCESS.md` - Review methodology
- `doc/agents.md` - AI agent guidelines
- `doc/skills.md` - Available skills/actions

## Troubleshooting

**Git hangs:** `rm -f .git/index.lock` (if no git operation running)

**Quarto errors:** Check that `.venv` is activated and all dependencies installed

**OneDrive issues:** See `doc/agents.md`
