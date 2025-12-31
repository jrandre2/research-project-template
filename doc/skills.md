# Skills Reference

**Related**: [SYNTHETIC_REVIEW_PROCESS.md](SYNTHETIC_REVIEW_PROCESS.md) | [PIPELINE.md](PIPELINE.md)
**Status**: Active
**Last Updated**: 2025-12-30

---

Available skills and actions for research project management.

> **Note:** For detailed stage documentation including data flow and outputs, see [PIPELINE.md](PIPELINE.md). This file provides quick command reference syntax.

## Pipeline Skills

### /ingest

Ingest raw data into the pipeline.

```bash
python src/pipeline.py ingest_data           # Requires data in data_raw/
python src/pipeline.py ingest_data --demo    # Generate synthetic demo data
```

**Options:**
- `--demo`: Generate synthetic demo data instead of reading from `data_raw/`

**Input:** `data_raw/` (or synthetic data with `--demo`)
**Output:** `data_work/data_raw.parquet`

### /link

Link records across data sources.

```bash
python src/pipeline.py link_records
```

**Input:** `data_work/data_raw.parquet`
**Output:** `data_work/data_linked.parquet`

### /panel

Build the analysis panel.

```bash
python src/pipeline.py build_panel
```

**Input:** `data_work/data_linked.parquet`
**Output:** `data_work/panel.parquet`

### /estimate

Run primary estimation.

```bash
python src/pipeline.py run_estimation --specification baseline
python src/pipeline.py run_estimation -s with_controls --sample restricted
```

**Options:**
- `--specification, -s`: Specification name (default: baseline)
- `--sample`: Sample restriction (default: full)

**Output:** `data_work/diagnostics/`

### /robustness

Run robustness checks.

```bash
python src/pipeline.py estimate_robustness
```

**Output:** `data_work/diagnostics/robustness_*.csv`

### /figures

Generate publication figures.

```bash
python src/pipeline.py make_figures
```

**Output:** `manuscript_quarto/figures/*.png`

### /validate

Validate manuscript against journal requirements.

```bash
python src/pipeline.py validate_submission --journal jeem
python src/pipeline.py validate_submission -j aer --report
```

**Options:**
- `--journal, -j`: Target journal (default: jeem)
- `--report`: Generate markdown report

---

## Review Management Skills

### /review-status

Check current review cycle status.

```bash
python src/pipeline.py review_status
python src/pipeline.py review_status -m main    # specific manuscript
```

Shows summary statistics, pending items, and verification progress.

### /review-new

Start a new review cycle with a focus-specific prompt.

```bash
python src/pipeline.py review_new --focus economics
python src/pipeline.py review_new -f engineering
python src/pipeline.py review_new -m main -f methods    # with manuscript
```

**Focus Options:**
- `economics` - Identification, causal inference, econometrics
- `engineering` - Reproducibility, benchmarks, validation
- `social_sciences` - Theory, generalizability, ethics
- `general` - Structure, clarity, contribution
- `methods` - Statistical rigor, methodology critique
- `policy` - Practitioner perspective, actionability
- `clarity` - Writing quality, accessibility

### /review-archive

Archive completed review cycle and reset for new one.

```bash
python src/pipeline.py review_archive
python src/pipeline.py review_archive -m main    # specific manuscript
```

Moves current tracker to `doc/reviews/archive/` and clears for next cycle.

### /review-verify

Run verification checklist for current cycle.

```bash
python src/pipeline.py review_verify
python src/pipeline.py review_verify -m main    # specific manuscript
```

Shows completed vs. pending verification items.

### /review-report

Generate summary report of all review cycles.

```bash
python src/pipeline.py review_report
```

Lists all archived and active reviews with statistics.

---

## Manuscript Skills

### /manuscript-guardrails

Behavior rules for manuscript edits. Apply when editing any `manuscript_quarto/**/*.qmd` (including appendices and variants).

- Manuscript must be standalone; do not reference internal prior work, drafts, or memos.
- Metacommentary is prohibited; present findings directly (avoid "in this paper", "the next section").
- External literature is appropriate; cite published research normally.

### /render

Render manuscript in all formats.

```bash
cd manuscript_quarto && ./render_all.sh
```

**Output:** `manuscript_quarto/_output/`
- HTML files
- PDF
- DOCX

### /render-journal

Render manuscript for specific journal.

```bash
cd manuscript_quarto && ./render_all.sh --profile jeem
cd manuscript_quarto && ./render_all.sh --profile aer
cd manuscript_quarto && ./render_all.sh --profile nhaz
```

**Available profiles:**
- `jeem` - Journal of Environmental Economics and Management
- `aer` - American Economic Review
- `nhaz` - Natural Hazards (Springer)

### /variant-new

Create a divergent manuscript variant and capture provenance.

```bash
cd manuscript_quarto && ./variant_new.sh <name>
```

### /variant-snapshot

Refresh variant provenance.

```bash
cd manuscript_quarto && python variant_tools.py snapshot --variant <name>
```

### /variant-compare

Compare two variants.

```bash
cd manuscript_quarto && python variant_tools.py compare --left <a> --right <b>
```

### /preview

Live preview manuscript.

```bash
cd manuscript_quarto && ../tools/bin/quarto preview
```

---

## Journal Configuration Skills

### /journal-list

List available journal configurations.

```bash
python src/pipeline.py journal_list
```

Shows all available journal configs and template files in `manuscript_quarto/journal_configs/`.

### /journal-validate

Validate a journal configuration against the comprehensive template.

```bash
python src/pipeline.py journal_validate --config natural_hazards
python src/pipeline.py journal_validate -c jeem
```

Reports missing sections and key fields.

### /journal-compare

Compare manuscript against journal requirements.

```bash
python src/pipeline.py journal_compare --journal natural_hazards
python src/pipeline.py journal_compare -j jeem --manuscript manuscript_quarto/
```

Checks:
- Required files present
- Abstract word limits
- Keyword count
- Figure resolution requirements
- Reference style
- Submission checklist

### /journal-parse

Parse raw journal guidelines into structured YAML configuration.

```bash
python src/pipeline.py journal_parse --input guidelines.txt --output new_journal.yml
python src/pipeline.py journal_parse -i guidelines.txt -o nature.yml --journal "Nature"
python src/pipeline.py journal_parse --url https://example.com/guidelines --output journal.yml --save-raw
```

**Options:**
- `--input, -i`: Input file with raw guidelines text
- `--url, -u`: URL to author guidelines (required if no input file)
- `--output, -o`: Output config filename (default: new_journal.yml)
- `--journal, -j`: Journal name (optional)
- `--template, -t`: Template name (default: template_comprehensive)
- `--save-raw`: Save fetched guidelines to `doc/journal_guidelines/`
- `--raw-dir`: Directory to save fetched guidelines
- `--overwrite`: Overwrite existing files

**Notes:**
- PDF guidelines must be converted to text or HTML before parsing.
- Parsing is heuristic; review the generated YAML for completeness.
- Downloads default to `doc/journal_guidelines/` when no output directory is provided.

**Workflow:**
1. Copy journal author guidelines to a text file, or use `journal_fetch`/`--url`
2. Run parser to create initial config
3. Review and fill in missing fields
4. Create Quarto profile if needed (`_quarto-{abbrev}.yml`)
5. Validate with `journal_validate`

### /journal-fetch

Download journal guidelines from a URL.

```bash
python src/pipeline.py journal_fetch --url https://example.com/guidelines --journal "Journal Name" --text
```

**Options:**
- `--url, -u`: URL to author guidelines (required)
- `--output, -o`: Output filename (default: slug + extension)
- `--journal, -j`: Journal name for default filename
- `--raw-dir`: Directory to save guidelines
- `--overwrite`: Overwrite existing files
- `--text`: Also save a text-only version

**Notes:**
- PDF guidelines must be converted to text or HTML before parsing.

---

## AI-Assisted Drafting Skills

Generate draft manuscript sections from pipeline outputs using LLMs (requires API key).

### /draft-results

Generate a results section draft from estimation tables.

```bash
python src/pipeline.py draft_results --table main_results
python src/pipeline.py draft_results --table main_results --section primary
python src/pipeline.py draft_results --table robustness_results --dry-run
python src/pipeline.py draft_results --table main_results --provider openai
```

**Options:**

- `--table, -t`: Diagnostic CSV name (without .csv extension) - required
- `--section, -s`: Section name for output file (default: main)
- `--manuscript, -m`: Target manuscript (default: main)
- `--dry-run`: Show prompt without making API call
- `--provider, -p`: LLM provider (anthropic/openai, default: from config)

**Output:** `manuscript_quarto/drafts/results_<section>_<timestamp>.md`

### /draft-captions

Generate figure captions from figure files.

```bash
python src/pipeline.py draft_captions --figure "fig_*.png"
python src/pipeline.py draft_captions --figure "*.png" --dry-run
```

**Options:**

- `--figure, -f`: Figure glob pattern (e.g., "fig_*.png") - required
- `--manuscript, -m`: Target manuscript (default: main)
- `--dry-run`: Show prompt without making API call
- `--provider, -p`: LLM provider (anthropic/openai)

**Output:** `manuscript_quarto/drafts/captions_<timestamp>.md`

### /draft-abstract

Synthesize an abstract from manuscript sections.

```bash
python src/pipeline.py draft_abstract
python src/pipeline.py draft_abstract --max-words 200
python src/pipeline.py draft_abstract --dry-run
```

**Options:**

- `--manuscript, -m`: Target manuscript (default: main)
- `--max-words`: Target word limit (default: 250)
- `--dry-run`: Show prompt without making API call
- `--provider, -p`: LLM provider (anthropic/openai)

**Output:** `manuscript_quarto/drafts/abstract_<timestamp>.md`

### Configuration

LLM settings in `src/config.py`:

```python
LLM_PROVIDER = 'anthropic'  # or 'openai'
LLM_MODELS = {
    'anthropic': 'claude-sonnet-4-20250514',
    'openai': 'gpt-4-turbo-preview',
}
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096
```

**Environment Variables:**

- `ANTHROPIC_API_KEY`: Required for Anthropic provider
- `OPENAI_API_KEY`: Required for OpenAI provider

**Notes:**

- All drafts include metadata headers and require human review
- Use `--dry-run` to preview prompts before making API calls
- Outputs are saved to `manuscript_quarto/drafts/` with timestamps

---

## Data Audit Skills

### /audit

Audit pipeline data files for quality and completeness.

```bash
python src/pipeline.py audit_data
python src/pipeline.py audit_data --full --report
python src/pipeline.py audit_data --output audit.json
```

**Options:**

- `--full, -f`: Run detailed column analysis
- `--report, -r`: Save markdown report to `data_work/diagnostics/`
- `--output, -o`: Custom output path for JSON report

**Output:** Console summary and optional JSON/markdown reports.

---

## Cache Management Skills

### /cache-stats

Show cache usage statistics for all pipeline stages.

```bash
python src/pipeline.py cache stats
```

**Output:** Cache file counts and sizes per stage.

### /cache-clear

Clear cached pipeline results to force recomputation.

```bash
python src/pipeline.py cache clear                    # Clear all caches
python src/pipeline.py cache clear -s s03_estimation  # Clear specific stage
```

**Options:**

- `--stage, -s`: Clear cache for specific stage only

**Notes:**

- Use when input data changes or you want fresh results
- Related CLI flags: `--no-cache`, `--sequential`, `--workers`

---

## Documentation Skills

### /update-docs

Update documentation after changes.

1. Update relevant doc file in `doc/`
2. Update `doc/CHANGELOG.md` with change summary
3. Update `doc/README.md` if adding/removing docs

### /update-changelog

Add entry to changelog.

Format:
```markdown
## [YYYY-MM-DD]

- [Description of change]
- Files modified: [list]
```

---

## Data Skills

### /sync-diagnostics

Copy diagnostics to manuscript data folder.

```bash
cp data_work/diagnostics/*.csv manuscript_quarto/data/
```

### /sync-figures

Copy figures to the manuscript figures folder (only needed if you export to `figures/`).

```bash
cp figures/*.png manuscript_quarto/figures/
```

---

## Git Skills

### /status

Check repository status.

```bash
git status
git log --oneline -5
```

### /commit

Create a commit (when requested by user).

```bash
git add <files>
git commit -m "Description"
```

### /push

Push to remote (when requested by user).

```bash
git push origin <branch>
```

---

## Troubleshooting Skills

### /fix-git-lock

Remove stale git lock files.

```bash
rm -f .git/index.lock
rm -f .git/refs/heads/*.lock
```

**Warning:** Only use if no git operation is running.

### /check-env

Verify environment setup.

```bash
source .venv/bin/activate
python --version
pip list | grep -E "(pandas|numpy|matplotlib)"
```

### /rebuild-panel

Rebuild panel from scratch.

```bash
python src/pipeline.py ingest_data
python src/pipeline.py link_records
python src/pipeline.py build_panel
```

---

## Project Migration Skills

AI-powered tools for analyzing and migrating external research projects to the standardized platform structure.

### /analyze-project

Analyze an external project's structure and codebase.

```bash
python src/pipeline.py analyze_project --path /path/to/project
python src/pipeline.py analyze_project --path /path/to/project --output analysis.json
```

**Options:**
- `--path, -p`: Path to project to analyze (required)
- `--output, -o`: Save JSON analysis to file (optional)

**Output includes:**
- Directory count and structure
- File count by type
- Python module analysis (imports, functions, classes, docstrings)
- Pattern detection (pipeline stages, tests, notebooks, manuscripts)

### /map-project

Map an analyzed project's structure to the platform structure.

```bash
python src/pipeline.py map_project --path /path/to/project
python src/pipeline.py map_project --path /path/to/project --output mapping.json
```

**Options:**
- `--path, -p`: Path to project to map (required)
- `--output, -o`: Save JSON mapping to file (optional)

**Mapping categories:**
- Data files (`data/`) -> `data_raw/`
- Output files (`output/`, `outputs/`, `figures/`) -> `manuscript_quarto/figures/` (primary)
- Documentation (`docs/`) -> `doc/`
- Tests (`tests/`) -> `tests/`
- Python modules -> `src/stages/s00-s06.py` (based on content keywords)
- Utility modules with `util` or `helper` in the path -> `src/utils/`

Note: `doc/` and `test/` are detected but not copied by the mapper; rename to `docs/` and `tests/` or copy manually.

### /plan-migration

Generate a detailed migration plan for moving a project to the platform structure.

```bash
python src/pipeline.py plan_migration --path /source --target /target
python src/pipeline.py plan_migration --path /source --target /target --output plan.md
```

**Options:**
- `--path, -p`: Path to source project (required)
- `--target, -t`: Path for migrated project (required)
- `--output, -o`: Save markdown plan to file (optional)

**Plan includes:**
- Setup steps (directory structure, git, venv)
- Copy operations (data, figures, docs, tests)
- Transform operations (merge modules into stages)
- Generate operations (create documentation)
- Verify operations (check imports, tests)
- Complexity estimate (low/medium/high)
- Warnings for items needing manual review

### /migrate-project

Execute a migration plan to transform a project.

```bash
# Dry run - see what would happen
python src/pipeline.py migrate_project --path /source --target /target --dry-run

# Actual execution
python src/pipeline.py migrate_project --path /source --target /target
```

**Options:**
- `--path, -p`: Path to source project (required)
- `--target, -t`: Path for migrated project (required)
- `--dry-run`: Show what would be done without making changes

**Outputs:**
- Created directory structure at target
- Copied data, figures, docs, tests
- Scaffold stage files (merge instructions for manual completion)
- Generated documentation templates
- `MIGRATION_REPORT.md` in target directory

**Workflow:**
1. Run with `--dry-run` first to preview changes
2. Execute without `--dry-run` to perform migration
3. Review `MIGRATION_REPORT.md` for verification status
4. Complete generated stage files by merging source code manually
5. Run verification steps to ensure completeness
