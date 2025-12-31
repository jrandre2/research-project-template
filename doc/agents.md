# Agent Guidelines

**Related**: [AGENT_TOOLS.md](AGENT_TOOLS.md) | [skills.md](skills.md)
**Status**: Active
**Last Updated**: 2025-12-27

---

Guidance for AI agents working on this project.

## Project Structure

```
[Project Root]/
├── CLAUDE.md          # Quick reference (read first)
├── doc/               # Documentation
│   ├── README.md      # Doc index
│   ├── agents.md      # This file
│   ├── skills.md      # Available skills
│   └── AGENT_TOOLS.md # Migration tools reference
├── src/               # Pipeline code
│   ├── pipeline.py    # Main CLI
│   ├── stages/        # Pipeline stages (s00-s09)
│   └── agents/        # Project migration tools
├── manuscript_quarto/ # Quarto manuscript
└── data_work/         # Working data
```

## Common Tasks

### Running Pipeline

```bash
source .venv/bin/activate
python src/pipeline.py <command>
```

Available commands:

**Data Processing:**
- `ingest_data` - Load raw data
- `link_records` - Link data sources
- `build_panel` - Create analysis panel
- `run_estimation` - Run estimation
- `estimate_robustness` - Robustness checks
- `make_figures` - Generate figures
- `validate_submission` - Check journal requirements

**Project Migration (see [AGENT_TOOLS.md](AGENT_TOOLS.md)):**
- `analyze_project` - Analyze external project structure
- `map_project` - Map project to platform structure
- `plan_migration` - Generate migration plan
- `migrate_project` - Execute migration

### Rendering Manuscript

```bash
cd manuscript_quarto
./render_all.sh                    # All formats
./render_all.sh --profile jeem     # JEEM format
```

## OneDrive Sync Issues

If project is in a OneDrive-synced folder:

**Symptoms:** `git status` takes 30+ seconds, commands timeout.

**Solutions:**

1. **Pause OneDrive** via menu bar - click OneDrive icon → gear → Pause Syncing
2. **Work locally** - copy to `~/Projects/` for git-intensive work

**Agent protocol:**

- Use shorter timeouts (5-10s)
- If 3+ commands hang, advise user to pause OneDrive
- Batch file writes sequentially, not in parallel

## Git Operations

### Authentication

Uses `gh` CLI. If push fails with credential errors:

```bash
gh auth status          # Verify logged in
gh auth setup-git       # Configure git to use gh
```

### Lock Files

If git hangs due to interrupted operation:

```bash
rm -f .git/index.lock
rm -f .git/refs/heads/*.lock
```

Only remove locks if no git operation is in progress.

## File Modification Guidelines

### Safe to Modify

- `src/stages/*.py` - Pipeline implementation
- `src/agents/*.py` - Migration tools
- `manuscript_quarto/*.qmd` - Manuscript content
- `doc/*.md` - Documentation

### Modify with Caution

- `src/pipeline.py` - Affects all pipeline commands
- `manuscript_quarto/_quarto*.yml` - Affects rendering
- `src/agents/structure_mapper.py` - Affects module-to-stage mapping logic

### Do Not Modify

- `data_raw/` - Raw data (never modify)
- `data_work/*.parquet` - Generated data (re-run pipeline instead)
- `manuscript_quarto/csl/apa.csl` - Standard citation style
