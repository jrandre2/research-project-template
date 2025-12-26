# Skills Reference

Available skills and actions for research project management.

## Pipeline Skills

### /ingest

Ingest raw data into the pipeline.

```bash
python src/pipeline.py ingest_data
```

**Input:** `data_raw/`
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
python src/pipeline.py run_estimation -s robustness --sample restricted
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

**Output:** `figures/*.png`

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
```

Shows summary statistics, pending items, and verification progress.

### /review-new

Start a new review cycle with discipline-specific template.

```bash
python src/pipeline.py review_new --discipline economics
python src/pipeline.py review_new --discipline engineering
python src/pipeline.py review_new --discipline social_sciences
python src/pipeline.py review_new --discipline general
```

**Disciplines:**
- `economics` - Identification, causal inference, econometrics
- `engineering` - Reproducibility, benchmarks, validation
- `social_sciences` - Theory, generalizability, ethics
- `general` - Structure, clarity, contribution

### /review-archive

Archive completed review cycle and reset for new one.

```bash
python src/pipeline.py review_archive
```

Moves current tracker to `doc/reviews/archive/` and clears for next cycle.

### /review-verify

Run verification checklist for current cycle.

```bash
python src/pipeline.py review_verify
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

Shows all available journal configs and templates in `manuscript_quarto/journal_configs/`.

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
```

**Options:**
- `--input, -i`: Input file with raw guidelines text (required)
- `--output, -o`: Output config filename (default: new_journal.yml)
- `--journal, -j`: Journal name (optional)

**Workflow:**
1. Copy journal author guidelines to a text file
2. Run parser to create initial config
3. Review and fill in missing fields
4. Create Quarto profile if needed (`_quarto-{abbrev}.yml`)
5. Validate with `journal_validate`

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

Copy figures to manuscript figures folder.

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
