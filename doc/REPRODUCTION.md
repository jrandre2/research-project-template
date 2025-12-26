# Reproduction Guide

**Related**: [PIPELINE.md](PIPELINE.md) | [DATA_DICTIONARY.md](DATA_DICTIONARY.md)
**Status**: Active
**Last Updated**: [Date]

---

## Overview

This guide documents how to reproduce the analysis from scratch.

---

## Prerequisites

### Software Requirements

- Python 3.10+
- Quarto 1.3+
- Git

### Python Packages

See `requirements.txt` for full list. Key packages:
- pandas
- numpy
- matplotlib
- statsmodels (if using)

---

## Setup

### 1. Clone Repository

```bash
git clone [repository-url]
cd [project-name]
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Obtain Raw Data

[Document how to obtain raw data]

Place files in `data_raw/`:
- `[file1.csv]` - [Description]
- `[file2.csv]` - [Description]

---

## Run Pipeline

### Full Reproduction

```bash
source .venv/bin/activate

# Data processing
python src/pipeline.py ingest_data
python src/pipeline.py link_records
python src/pipeline.py build_panel

# Estimation
python src/pipeline.py run_estimation
python src/pipeline.py estimate_robustness

# Figures
python src/pipeline.py make_figures

# Manuscript
cd manuscript_quarto
./render_all.sh
```

### Expected Runtime

| Stage | Approximate Time |
|-------|------------------|
| ingest_data | [X minutes] |
| link_records | [X minutes] |
| build_panel | [X minutes] |
| run_estimation | [X minutes] |
| make_figures | [X minutes] |
| render_all | [X minutes] |
| **Total** | **[X minutes]** |

---

## Output Files

### Data Files

| File | Description |
|------|-------------|
| `data_work/panel.parquet` | Main analysis panel |
| `data_work/diagnostics/*.csv` | Estimation results |

### Figures

| File | Description |
|------|-------------|
| `figures/[fig1].png` | [Description] |
| `figures/[fig2].png` | [Description] |

### Manuscript

| File | Description |
|------|-------------|
| `manuscript_quarto/_output/index.html` | HTML manuscript |
| `manuscript_quarto/_output/[Name].pdf` | PDF manuscript |
| `manuscript_quarto/_output/[Name].docx` | Word manuscript |

---

## Verification

### Check Output Counts

[Document expected counts at each stage]

```bash
# Example verification commands
python -c "import pandas as pd; print(pd.read_parquet('data_work/panel.parquet').shape)"
```

### Expected Results

[Document key expected results for verification]

---

## Troubleshooting

### Common Issues

**Issue: Package not found**
```bash
pip install -r requirements.txt
```

**Issue: Quarto not found**
```bash
# Use project-local quarto
../tools/bin/quarto render
```

**Issue: Memory error**
[Suggest workarounds for memory-intensive steps]
