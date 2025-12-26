# Pipeline Documentation

**Related**: [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | [METHODOLOGY.md](METHODOLOGY.md)
**Status**: Active
**Last Updated**: [Date]

---

## TL;DR

```bash
source .venv/bin/activate

# Full pipeline
python src/pipeline.py ingest_data
python src/pipeline.py link_records
python src/pipeline.py build_panel
python src/pipeline.py run_estimation
python src/pipeline.py make_figures

# Render manuscript
cd manuscript_quarto && ./render_all.sh
```

---

## Pipeline Overview

```
data_raw/ ──► ingest_data ──► link_records ──► build_panel ──► panel.parquet
                                                                    │
                                                                    ▼
figures/ ◄── make_figures ◄── run_estimation ──► diagnostics/*.csv
    │
    ▼
manuscript_quarto/figures/ ──► render_all.sh ──► _output/
```

---

## Stage Details

### Stage 00: Data Ingestion

**Command:** `python src/pipeline.py ingest_data`

**Purpose:** Load and preprocess raw data files.

**Input:** `data_raw/`
**Output:** `data_work/data_raw.parquet`

**Implementation:** `src/stages/s00_ingest.py`

---

### Stage 01: Record Linkage

**Command:** `python src/pipeline.py link_records`

**Purpose:** Link records across multiple data sources.

**Input:** `data_work/data_raw.parquet`
**Output:** `data_work/data_linked.parquet`

**Implementation:** `src/stages/s01_link.py`

---

### Stage 02: Panel Construction

**Command:** `python src/pipeline.py build_panel`

**Purpose:** Create the analysis panel from linked data.

**Input:** `data_work/data_linked.parquet`
**Output:** `data_work/panel.parquet`

**Implementation:** `src/stages/s02_panel.py`

---

### Stage 03: Primary Estimation

**Command:** `python src/pipeline.py run_estimation [options]`

**Options:**
- `--specification, -s`: Specification name (default: baseline)
- `--sample`: Sample restriction (default: full)

**Purpose:** Run primary estimation specifications.

**Input:** `data_work/panel.parquet`
**Output:** `data_work/diagnostics/*.csv`

**Implementation:** `src/stages/s03_estimation.py`

---

### Stage 04: Robustness Checks

**Command:** `python src/pipeline.py estimate_robustness`

**Purpose:** Run robustness specifications and sensitivity analyses.

**Input:** `data_work/panel.parquet`
**Output:** `data_work/diagnostics/robustness_*.csv`

**Implementation:** `src/stages/s04_robustness.py`

---

### Stage 05: Figure Generation

**Command:** `python src/pipeline.py make_figures`

**Purpose:** Generate publication-quality figures.

**Input:** `data_work/panel.parquet`, `data_work/diagnostics/*.csv`
**Output:** `figures/*.png`

**Implementation:** `src/stages/s05_figures.py`

---

### Stage 06: Manuscript Validation

**Command:** `python src/pipeline.py validate_submission [options]`

**Options:**
- `--journal, -j`: Target journal (default: jeem)
- `--report`: Generate markdown report

**Purpose:** Validate manuscript against journal requirements.

**Implementation:** `src/stages/s06_manuscript.py`

---

## Manuscript Rendering

### Render All Formats

```bash
cd manuscript_quarto
./render_all.sh
```

Output in `manuscript_quarto/_output/`:
- HTML files
- PDF
- DOCX

### Journal-Specific Rendering

```bash
./render_all.sh --profile jeem   # JEEM format
./render_all.sh --profile aer    # AER format
```

### Live Preview

```bash
cd manuscript_quarto
../tools/bin/quarto preview
```

---

## Common Workflows

### Full Rebuild

```bash
source .venv/bin/activate
python src/pipeline.py ingest_data
python src/pipeline.py link_records
python src/pipeline.py build_panel
python src/pipeline.py run_estimation
python src/pipeline.py make_figures
cd manuscript_quarto && ./render_all.sh
```

### Update After Data Change

```bash
python src/pipeline.py ingest_data
python src/pipeline.py link_records
python src/pipeline.py build_panel
python src/pipeline.py run_estimation
python src/pipeline.py make_figures
```

### Update Figures Only

```bash
python src/pipeline.py make_figures
cp figures/*.png manuscript_quarto/figures/
cd manuscript_quarto && ./render_all.sh
```
