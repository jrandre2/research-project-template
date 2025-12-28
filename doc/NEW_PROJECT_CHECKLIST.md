# New Project Checklist

Use this checklist when starting a new research project with CENTAUR.

## Before You Start

### Environment Prerequisites

- [ ] Python 3.10+ installed (`python --version`)
- [ ] pip or conda available for package management
- [ ] Git installed and configured
- [ ] Quarto installed (optional, for manuscript rendering)

### Data Assessment

- [ ] Data format identified (CSV, Excel, Parquet, other)
- [ ] Key columns identified:
  - [ ] Unit identifier (e.g., firm_id, county_fips, person_id)
  - [ ] Time period (e.g., year, month, quarter)
  - [ ] Treatment indicator
  - [ ] Outcome variable(s)
- [ ] Data quality understood:
  - [ ] Approximate row count
  - [ ] Missing value patterns
  - [ ] Potential duplicates

### Project Scope

- [ ] Research question defined
- [ ] Target journal identified (optional)
- [ ] Estimation approach decided (DiD, event study, RD, etc.)
- [ ] Key robustness checks planned

---

## Project Setup

### Initial Setup

- [ ] Clone or copy CENTAUR repository
- [ ] Remove `.git` directory if starting fresh
- [ ] Initialize new git repository
- [ ] Create virtual environment: `python -m venv .venv`
- [ ] Activate environment: `source .venv/bin/activate`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify installation: `python src/pipeline.py --help`

### Project Identity

- [ ] Update `README.md` with project description
- [ ] Update `CLAUDE.md` with project-specific instructions
- [ ] Update `manuscript_quarto/_quarto.yml` with paper title/authors

---

## Stage Customization

### Stage 00: Data Ingestion (`src/stages/s00_ingest.py`)

- [ ] Set `REQUIRED_COLUMNS` to your column names
- [ ] Set `COLUMN_TYPES` for type conversion
- [ ] Update `INPUT_PATTERNS` if using non-standard file extensions
- [ ] Test: `python src/pipeline.py ingest_data`
- [ ] Verify: Check `data_work/data_raw.parquet` has expected rows/columns

### Stage 01: Record Linkage (`src/stages/s01_link.py`)

- [ ] Decide if linking is needed (skip if single data source)
- [ ] Configure `LINK_KEYS` for exact matching
- [ ] Configure fuzzy matching threshold if needed
- [ ] Test: `python src/pipeline.py link_records`
- [ ] Verify: Check `data_work/diagnostics/linkage_summary.csv`

### Stage 02: Panel Construction (`src/stages/s02_panel.py`)

- [ ] Set `UNIT_ID` to your unit identifier column
- [ ] Set `TIME_ID` to your time period column
- [ ] Set `TREATMENT_VAR` and `OUTCOME_VAR`
- [ ] Configure event study parameters (treatment_period, etc.)
- [ ] Test: `python src/pipeline.py build_panel`
- [ ] Verify: Check `data_work/panel.parquet` and `panel_summary.csv`

### Stage 03: Estimation (`src/stages/s03_estimation.py`)

- [ ] Review default specifications in `SPECIFICATIONS`
- [ ] Add project-specific specifications if needed
- [ ] Set control variables
- [ ] Configure clustering variable
- [ ] Test: `python src/pipeline.py run_estimation`
- [ ] Verify: Check `estimation_results.csv` and `coefficients.csv`

### Stage 04: Robustness (`src/stages/s04_robustness.py`)

- [ ] Review default robustness checks
- [ ] Add project-specific placebo tests if needed
- [ ] Configure sample restrictions
- [ ] Test: `python src/pipeline.py estimate_robustness`
- [ ] Verify: Check `robustness_results.csv`

### Stage 05: Figures (`src/stages/s05_figures.py`)

- [ ] Review default figure specifications
- [ ] Customize figure styling in `src/utils/figure_style.py`
- [ ] Add project-specific figures if needed
- [ ] Test: `python src/pipeline.py make_figures`
- [ ] Verify: Check `manuscript_quarto/figures/`

---

## Manuscript Setup

### Basic Configuration

- [ ] Update `manuscript_quarto/_quarto.yml`:
  - [ ] Set `book.title`
  - [ ] Set `book.author`
  - [ ] Set output formats
- [ ] Update `manuscript_quarto/index.qmd` with your content
- [ ] Add references to `manuscript_quarto/references.bib`

### Journal Profile (Optional)

- [ ] Choose target journal profile or create new one
- [ ] Copy `_quarto-template.yml` to `_quarto-{journal}.yml`
- [ ] Configure journal-specific settings
- [ ] Test: `./render_all.sh --profile {journal}`

### Figures and Tables

- [ ] Verify figure references in manuscript match generated files
- [ ] Add figure captions and cross-references
- [ ] Configure table formatting

---

## Quality Assurance

### Before First Full Run

- [ ] Run pipeline end-to-end with `--demo` flag
- [ ] Check QA reports in `data_work/quality/`
- [ ] Review any threshold warnings
- [ ] Verify all expected outputs exist

### After Adding Real Data

- [ ] Run full pipeline without `--demo`
- [ ] Compare outputs to expected results
- [ ] Check estimation results for reasonableness
- [ ] Verify figure generation

### Before Submission

- [ ] Run `python src/pipeline.py validate_submission --journal {journal} --report`
- [ ] Address any validation warnings
- [ ] Verify word count meets requirements
- [ ] Check abstract length
- [ ] Run synthetic peer review: `python src/pipeline.py review_new --focus {focus}`

---

## Version Control

### Git Setup

- [ ] Initialize repository: `git init`
- [ ] Review `.gitignore` (data directories are excluded by default)
- [ ] Make initial commit
- [ ] Set up remote repository (GitHub, GitLab, etc.)
- [ ] Push initial version

### Ongoing

- [ ] Commit after each significant change
- [ ] Use descriptive commit messages
- [ ] Tag major milestones (e.g., "v1.0-submission")

---

## Common Issues

### Pipeline Fails at Ingestion

- Check that data files exist in `data_raw/`
- Verify column names match `REQUIRED_COLUMNS`
- Check for encoding issues in CSV files

### Panel Construction Errors

- Verify unit and time ID columns exist after linking
- Check for duplicate unit-time combinations
- Ensure treatment variable is properly coded

### Figure Generation Fails

- Verify estimation ran successfully first
- Check that required columns exist in results
- Review figure style configuration

### Manuscript Won't Render

- Verify Quarto is installed: `quarto --version`
- Check for syntax errors in .qmd files
- Verify all referenced figures exist

---

## Quick Reference

```bash
# Full pipeline run
source .venv/bin/activate
python src/pipeline.py ingest_data
python src/pipeline.py link_records
python src/pipeline.py build_panel
python src/pipeline.py run_estimation
python src/pipeline.py estimate_robustness
python src/pipeline.py make_figures

# Render manuscript
cd manuscript_quarto && ./render_all.sh

# Validate for submission
python src/pipeline.py validate_submission --journal jeem --report
```
