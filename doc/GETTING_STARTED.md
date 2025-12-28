# Getting Started with CENTAUR

This guide helps you find the right starting path based on your situation.

## Quick Assessment: Is CENTAUR Right for You?

CENTAUR is designed for:

- **Reproducible research pipelines** - Data processing through publication
- **Panel data / econometric analysis** - Fixed effects, event studies, robustness checks
- **Academic manuscript preparation** - Quarto-based with journal profiles
- **Multi-stage data workflows** - Ingest → Link → Panel → Estimate → Figures

CENTAUR may not be ideal if you need:

- Real-time data processing or streaming
- Non-Python analysis environments (R, Stata, Julia)
- Simple one-off scripts without reproducibility requirements

## What's Your Starting Point?

| I have... | Start here |
|-----------|------------|
| Nothing yet, want to explore | [Demo Walkthrough](#demo-walkthrough) |
| Data files (CSV/Excel/Parquet) | [Data-First Workflow](#data-first-workflow) |
| Manuscript draft/outline | [Manuscript-First Workflow](#manuscript-first-workflow) |
| Both data and manuscript | [Integration Workflow](#integration-workflow) |
| Existing analysis codebase | [Migration Workflow](#migration-workflow) |

---

## Demo Walkthrough

**Time:** 5 minutes

Try CENTAUR with synthetic data to see how it works:

```bash
# 1. Clone/copy the repository
git clone https://github.com/jrandre2/centaur-platform.git my-project
cd my-project

# 2. Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run pipeline with demo data
python src/pipeline.py ingest_data --demo
python src/pipeline.py link_records
python src/pipeline.py build_panel
python src/pipeline.py run_estimation
python src/pipeline.py make_figures

# 4. View outputs
ls data_work/diagnostics/
ls manuscript_quarto/figures/
```

**What you'll see:**

- `data_work/panel.parquet` - Analysis-ready panel data
- `data_work/diagnostics/` - Estimation results, robustness checks
- `manuscript_quarto/figures/` - Publication-ready figures

**Next:** Read [TUTORIAL.md](TUTORIAL.md) for detailed explanations of each stage.

---

## Data-First Workflow

**You have:** Data files ready for analysis
**You need:** A reproducible pipeline and manuscript

### Step 1: Set Up Your Project

```bash
# Clone and rename
git clone https://github.com/jrandre2/centaur-platform.git my-research-project
cd my-research-project

# Remove git history for fresh start (optional)
rm -rf .git
git init

# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Add Your Data

```bash
# Copy your data files
cp /path/to/your/data.csv data_raw/

# Or multiple files
cp /path/to/data/*.csv data_raw/
```

### Step 3: Customize Ingestion

Edit `src/stages/s00_ingest.py` to match your data:

```python
# Update required columns (line ~65)
REQUIRED_COLUMNS = ['your_id_column', 'your_time_column', 'outcome']

# Update column types (line ~68)
COLUMN_TYPES = {
    'your_id_column': 'int64',
    'your_time_column': 'int64',
    'outcome': 'float64',
}
```

### Step 4: Configure Panel Construction

Edit `src/stages/s02_panel.py` to specify your panel structure:

```python
# Set your ID and time columns
UNIT_ID = 'your_id_column'
TIME_ID = 'your_time_column'
TREATMENT_VAR = 'treatment'
OUTCOME_VAR = 'outcome'
```

### Step 5: Run the Pipeline

```bash
python src/pipeline.py ingest_data
python src/pipeline.py link_records
python src/pipeline.py build_panel
python src/pipeline.py run_estimation
python src/pipeline.py estimate_robustness
python src/pipeline.py make_figures
```

### Step 6: Write Your Manuscript

Edit `manuscript_quarto/index.qmd` with your content. Figures are automatically available in `manuscript_quarto/figures/`.

**Next:** See [CUSTOMIZATION.md](CUSTOMIZATION.md) for advanced stage customization.

---

## Manuscript-First Workflow

**You have:** A manuscript outline or draft
**You need:** To integrate data analysis later

The manuscript template is designed to work before pipeline data exists. Code blocks that load data will show a note about missing data but won't fail.

### Step 1: Set Up Your Project

```bash
git clone https://github.com/jrandre2/centaur-platform.git my-paper
cd my-paper
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Start Your Manuscript

Edit `manuscript_quarto/index.qmd`:

```markdown
---
title: "Your Paper Title"
author: "Your Name"
---

# Introduction

[Your introduction here]

# Literature Review

[Your literature review here]

# Methods

[Your methods here - can reference figures that will be generated later]

![Event Study Results](figures/fig_event_study.png){#fig-event-study}

# Results

[Placeholder - will be updated when data is available]

# Conclusion

[Your conclusion here]
```

### Step 3: Set Up Figure Placeholders (Optional)

If you reference figures in your manuscript, you can create placeholders so images render:

```bash
# Create simple placeholder images (requires ImageMagick)
convert -size 800x600 xc:lightgray -gravity center \
  -annotate +0+0 'Event Study\n(pending data)' \
  manuscript_quarto/figures/fig_event_study.png
```

Alternatively, use HTML comments as placeholders: `<!-- TODO: Figure from pipeline -->`

### Step 4: Render Your Draft

```bash
cd manuscript_quarto
./render_all.sh
```

### Step 5: When Data Arrives

Follow the [Data-First Workflow](#data-first-workflow) steps 2-5, then re-render your manuscript. The generated figures will replace your placeholders.

---

## Integration Workflow

**You have:** Both data files and an existing manuscript
**You need:** To connect them through the CENTAUR pipeline

### Step 1: Set Up Your Project

```bash
git clone https://github.com/jrandre2/centaur-platform.git my-project
cd my-project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Import Your Data

```bash
cp /path/to/your/data/*.csv data_raw/
```

### Step 3: Import Your Manuscript

```bash
# Copy your existing manuscript content
cp /path/to/your/manuscript.qmd manuscript_quarto/index.qmd

# Copy your bibliography
cp /path/to/your/references.bib manuscript_quarto/references.bib
```

### Step 4: Update Figure References

In your manuscript, update figure paths to point to the pipeline output directory:

```markdown
# Change from:
![Results](old_figures/my_figure.png)

# To:
![Results](figures/fig_results.png)
```

### Step 5: Customize and Run Pipeline

Follow [Data-First Workflow](#data-first-workflow) steps 3-5 to configure and run the pipeline.

### Step 6: Verify Integration

```bash
# Check that figures were generated
ls manuscript_quarto/figures/

# Render manuscript
cd manuscript_quarto && ./render_all.sh
```

---

## Migration Workflow

**You have:** An existing analysis codebase (Python, R, Stata, or mixed)
**You need:** To migrate it into CENTAUR's structure

CENTAUR includes migration tools that analyze your existing project and generate a migration plan.

### Step 1: Analyze Your Project

```bash
# From a CENTAUR installation
python src/pipeline.py analyze_project --path /path/to/existing/project
```

This produces a report showing:

- File types and counts
- Detected data files
- Code structure analysis
- Recommended stage mappings

### Step 2: Generate Migration Plan

```bash
python src/pipeline.py plan_migration \
  --path /path/to/existing/project \
  --target /path/to/new/centaur/project
```

### Step 3: Execute Migration

```bash
# First, dry run to see what would happen
python src/pipeline.py migrate_project \
  --path /path/to/existing/project \
  --target /path/to/new/centaur/project \
  --dry-run

# If satisfied, run for real
python src/pipeline.py migrate_project \
  --path /path/to/existing/project \
  --target /path/to/new/centaur/project
```

### Step 4: Manual Refinement

Migration is automated but imperfect. After migration:

1. Review `MIGRATION_REPORT.md` in the new project
2. Test each stage individually
3. Customize stage logic as needed

**Details:** See [AGENT_TOOLS.md](AGENT_TOOLS.md) for complete migration tool documentation.

---

## Incremental Adoption

You don't have to migrate everything at once. Consider:

### Start with Just Data Processing

Use stages 00-02 (ingest, link, panel) while keeping your existing analysis code.

### Add Estimation Later

Once data processing is stable, migrate your estimation code to stage 03.

### Manuscript Integration Last

After analysis is reproducible, integrate with Quarto manuscript.

---

## Next Steps

| Goal | Resource |
|------|----------|
| Detailed stage walkthrough | [TUTORIAL.md](TUTORIAL.md) |
| Customize stages for your data | [CUSTOMIZATION.md](CUSTOMIZATION.md) |
| Understand architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Migrate existing project | [AGENT_TOOLS.md](AGENT_TOOLS.md) |
| Configure for your journal | [Journal profiles in manuscript_quarto/](../manuscript_quarto/) |

## Getting Help

- Check [TUTORIAL.md](TUTORIAL.md) troubleshooting section
- Review stage-specific documentation in `src/stages/`
- Open an issue at the repository
