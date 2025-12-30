# Pipeline Documentation

**Related**: [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | [METHODOLOGY.md](METHODOLOGY.md)
**Status**: Active
**Last Updated**: 2025-12-27

---

## TL;DR

```bash
source .venv/bin/activate

# Full pipeline (use --demo if data_raw/ is empty)
python src/pipeline.py ingest_data
python src/pipeline.py link_records
python src/pipeline.py build_panel
python src/pipeline.py run_estimation --specification baseline
python src/pipeline.py estimate_robustness
python src/pipeline.py make_figures

# Stage discovery and versioning
python src/pipeline.py list_stages                # List all stages
python src/pipeline.py list_stages -p s00         # List s00 versions
python src/pipeline.py run_stage s00_ingest       # Run specific stage

# Render manuscript
cd manuscript_quarto && ./render_all.sh
```

Note: `ingest_data` requires data in `data_raw/`. Use `--demo` flag to generate synthetic demo data.

Journal validation is optional. If no journal config is set, skip `validate_submission` and render with Quarto only. When a journal config exists, `validate_submission` checks word count (when configured), abstract length, required sections, figure formats, and bibliography/citation presence.

Review cycles are handled through `review_new`, `review_verify`, and `review_archive`, with `manuscript_quarto/REVISION_TRACKER.md` and `doc/reviews/archive/` preserving responses. Divergent drafts use manuscript variants (see `doc/MANUSCRIPT_VARIANTS.md`).

---

## Pipeline Overview

```
data_raw/ ──► ingest_data ──► data_work/data_raw.parquet
                           │
                           ▼
                link_records ──► data_work/data_linked.parquet
                           │           └─► data_work/diagnostics/linkage_summary.csv
                           ▼
                build_panel ──► data_work/panel.parquet
                           │           └─► data_work/diagnostics/panel_summary.csv
                           ▼
             run_estimation ──► data_work/diagnostics/estimation_results.csv
                           │           └─► data_work/diagnostics/coefficients.csv
                           ▼
          estimate_robustness ──► data_work/diagnostics/robustness_results.csv
                           │           └─► data_work/diagnostics/placebo_results.csv
                           ▼
              make_figures ──► manuscript_quarto/figures/*.png ──► render_all.sh ──► manuscript_quarto/_output/
```

---

## Stage Details

### Stage 00: Data Ingestion

**Command:** `python src/pipeline.py ingest_data`

**Purpose:** Load and preprocess raw data files. Use `--demo` flag to generate synthetic data if `data_raw/` is empty.

**Input:** `data_raw/`
**Output:** `data_work/data_raw.parquet`

**Implementation:** `src/stages/s00_ingest.py`

---

### Stage 01: Record Linkage

**Command:** `python src/pipeline.py link_records`

**Purpose:** Link records across multiple data sources.

**Input:** `data_work/data_raw.parquet`
**Output:** `data_work/data_linked.parquet`
**Diagnostics:** `data_work/diagnostics/linkage_summary.csv`

**Implementation:** `src/stages/s01_link.py`

---

### Stage 02: Panel Construction

**Command:** `python src/pipeline.py build_panel`

**Purpose:** Create the analysis panel from linked data.

**Input:** `data_work/data_linked.parquet`
**Output:** `data_work/panel.parquet`
**Diagnostics:** `data_work/diagnostics/panel_summary.csv`

**Implementation:** `src/stages/s02_panel.py`

---

### Stage 03: Primary Estimation

**Command:** `python src/pipeline.py run_estimation [options]`

**Options:**
- `--specification, -s`: Specification name (default: baseline)
- `--sample`: Sample restriction (default: full)

**Purpose:** Run primary estimation specifications.

**Input:** `data_work/panel.parquet`
**Output:** `data_work/diagnostics/estimation_results.csv`, `data_work/diagnostics/coefficients.csv`

**Implementation:** `src/stages/s03_estimation.py`

---

### Stage 04: Robustness Checks

**Command:** `python src/pipeline.py estimate_robustness`

**Purpose:** Run robustness specifications and sensitivity analyses.

**Input:** `data_work/panel.parquet`
**Output:** `data_work/diagnostics/robustness_results.csv`, `data_work/diagnostics/placebo_results.csv`

**Implementation:** `src/stages/s04_robustness.py`

**Extended Robustness Tests (when applicable):**

The robustness stage includes additional tests for geographic and ML-based analyses:

| Test | Description | When to Use |
|------|-------------|-------------|
| Spatial vs Random CV | Compare spatial and random cross-validation to quantify geographic data leakage | Data with latitude/longitude coordinates |
| Feature Ablation | Test model performance with feature subsets | Multiple feature groups |
| Tuned Models | Nested CV with hyperparameter tuning for Ridge, ElasticNet, RF, GB | ML prediction tasks |
| Encoding Comparisons | Compare categorical vs ordinal treatment encoding | Categorical treatment variables |

**Usage with spatial CV:**

```python
from stages.s04_robustness import run_spatial_cv_comparison, run_feature_ablation

# Compare spatial vs random CV
results = run_spatial_cv_comparison(
    df, feature_cols=['feature_1', 'feature_2'],
    lat_col='latitude', lon_col='longitude'
)

# Feature ablation study
results = run_feature_ablation(df, feature_cols)
```

See [Spatial Cross-Validation](#spatial-cross-validation) section below for methodology details.

---

### Stage 05: Figure Generation

**Command:** `python src/pipeline.py make_figures`

**Purpose:** Generate publication-quality figures.

**Input:** `data_work/panel.parquet`, `data_work/diagnostics/*.csv`
**Output:** `manuscript_quarto/figures/*.png`

**Implementation:** `src/stages/s05_figures.py`

If you need a top-level export, copy from `manuscript_quarto/figures/` to `figures/`.

---

### Stage 06: Manuscript Validation

**Command:** `python src/pipeline.py validate_submission [options]`

**Options:**
- `--journal, -j`: Target journal (default: jeem)
- `--report`: Generate markdown report

**Purpose:** Validate manuscript against journal requirements.

**Output (with `--report`):** `data_work/diagnostics/submission_validation.md`

**Implementation:** `src/stages/s06_manuscript.py`

---

### Stage 07: Review Management

**Commands:**

- `python src/pipeline.py review_status [-m manuscript]`
- `python src/pipeline.py review_new [-m manuscript] [-f focus]`
- `python src/pipeline.py review_verify [-m manuscript]`
- `python src/pipeline.py review_archive [-m manuscript]`
- `python src/pipeline.py review_report`

**Options:**

- `--manuscript, -m`: Target manuscript (default: main)
- `--focus, -f`: Review focus area (economics, engineering, social_sciences, general, methods, policy, clarity)
- `--discipline, -d`: Deprecated alias for --focus

**Purpose:** Manage synthetic peer review cycles and track responses. Supports multi-manuscript projects.

**Outputs:**

- `manuscript_quarto/REVISION_TRACKER.md`
- `doc/reviews/archive/` (archived reviews)

**Implementation:** `src/stages/s07_reviews.py`

---

### Stage 08: Journal Configuration Tools

**Commands:**
- `python src/pipeline.py journal_list`
- `python src/pipeline.py journal_validate --config natural_hazards`
- `python src/pipeline.py journal_compare --journal natural_hazards`
- `python src/pipeline.py journal_parse --input guidelines.txt --output new_journal.yml`
- `python src/pipeline.py journal_parse --url https://example.com/guidelines --journal "Journal Name" --output journal.yml --save-raw`
- `python src/pipeline.py journal_fetch --url https://example.com/guidelines --journal "Journal Name" --text`

**Purpose:** List, validate, compare, and parse journal requirements.

**Outputs:**
- `manuscript_quarto/journal_configs/<name>.yml` (for `journal_parse`)
- `doc/journal_guidelines/*` (when using `journal_fetch` or `journal_parse --save-raw`)

**Notes:**
- PDF guidelines must be converted to text or HTML before parsing.
- Parsing uses heuristic extraction; manual review is required.

**Implementation:** `src/stages/s08_journal_parser.py`

---

## Stage 09: AI-Assisted Writing

**Purpose:** Generate draft manuscript sections from pipeline outputs using LLMs.

**Module:** `src/stages/s09_writing.py`

### Commands

| Command          | Description                                  |
| ---------------- | -------------------------------------------- |
| `draft_results`  | Draft results section from estimation tables |
| `draft_captions` | Generate figure captions                     |
| `draft_abstract` | Synthesize abstract from manuscript          |

### Usage

```bash
# Draft results section from estimation table
python src/pipeline.py draft_results --table main_results
python src/pipeline.py draft_results --table main_results --section primary

# Preview prompt without API call
python src/pipeline.py draft_results --table main_results --dry-run

# Generate figure captions
python src/pipeline.py draft_captions --figure "fig_*.png"

# Synthesize abstract with word limit
python src/pipeline.py draft_abstract --max-words 200

# Use alternative provider
python src/pipeline.py draft_results --table main_results --provider openai
```

### Options

| Option             | Description                                   |
| ------------------ | --------------------------------------------- |
| `--table, -t`      | Diagnostic CSV name (without .csv extension)  |
| `--section, -s`    | Section name for output file (default: main)  |
| `--figure, -f`     | Figure glob pattern (e.g., "fig_*.png")       |
| `--manuscript, -m` | Target manuscript (default: main)             |
| `--max-words`      | Target word limit for abstract (default: 250) |
| `--dry-run`        | Show prompt without making API call           |
| `--provider, -p`   | LLM provider: anthropic or openai             |

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

### Output

Drafts are saved to `manuscript_quarto/drafts/` with metadata headers:

```markdown
<!-- AI-Generated Draft
     Source: data_work/diagnostics/main_results.csv
     Provider: anthropic/claude-sonnet-4-20250514
     Generated: 2025-12-27 14:30:22
     Status: REQUIRES HUMAN REVIEW
-->
```

All drafts require human review before integration into the manuscript.

**Implementation:** `src/stages/s09_writing.py`, `src/llm/`

---

## Versioned Stages

Stages can evolve over time using version suffixes. This allows keeping alternative implementations while maintaining a clear evolution history.

**Naming Convention:** `s00_ingest` → `s00b_standardize` → `s00c_enhanced`

**Commands:**

```bash
# List all available stages
python src/pipeline.py list_stages

# List versions of a specific stage
python src/pipeline.py list_stages -p s00

# Run a specific stage version
python src/pipeline.py run_stage s00b_standardize
```

**Benefits:**

- Preserve alternative implementations for comparison
- Track methodological evolution
- Switch between versions for robustness checks

---

## Caching and Parallel Execution

The pipeline includes intelligent caching and parallel execution to dramatically improve performance during iterative development.

### Performance Impact

| Stage                      | Without Cache | With Cache | Improvement |
| -------------------------- | ------------- | ---------- | ----------- |
| s03_estimation (4 specs)   | ~3 sec        | <100ms     | ~30x        |
| s04_robustness (10 tests)  | ~8 sec        | <300ms     | ~25x        |
| s05_figures (5 plots)      | ~4 sec        | <500ms     | ~8x         |

### How Caching Works

The cache automatically tracks:

1. **Data hash** - MD5 hash of input DataFrames
2. **Configuration hash** - Hash of specification parameters
3. **File dependencies** - Content hash of any file inputs

When you re-run a stage, cached results are used if:

- Input data is unchanged
- Configuration is unchanged
- All file dependencies are unchanged

Cache files are stored in `data_work/.cache/<stage_name>/`.

### CLI Flags

```bash
# Disable caching (force recomputation)
python src/pipeline.py run_estimation --no-cache
python src/pipeline.py estimate_robustness --no-cache
python src/pipeline.py make_figures --no-cache

# Disable parallel execution
python src/pipeline.py run_estimation --sequential
python src/pipeline.py estimate_robustness --sequential
python src/pipeline.py make_figures --sequential

# Control parallel workers (default: CPU count)
python src/pipeline.py run_estimation --workers 4
python src/pipeline.py estimate_robustness -w 2
```

### Cache Management

```bash
# View cache statistics
python src/pipeline.py cache stats

# Clear all cached data
python src/pipeline.py cache clear

# Clear a specific stage's cache
python src/pipeline.py cache clear --stage s03_estimation
```

### When Cache Invalidates

Caches automatically invalidate when:

- **Data changes**: Any modification to input parquet files
- **Config changes**: Different specification parameters
- **Code changes**: Not automatically tracked (use `cache clear` after code changes)

### Parallel Execution

Parallel execution runs independent computations simultaneously:

- **s03_estimation**: Multiple specifications in parallel
- **s04_robustness**: Robustness tests in parallel by category
- **s05_figures**: Multiple figures in parallel

Uses `ProcessPoolExecutor` for CPU-bound work (estimation) and `ThreadPoolExecutor` for I/O-bound work (figures).

### Configuration

Global settings in `src/config.py`:

```python
CACHE_ENABLED = True           # Enable/disable caching globally
CACHE_MAX_AGE_HOURS = 168      # Cache TTL (1 week default)
PARALLEL_ENABLED = True        # Enable/disable parallel execution
PARALLEL_MAX_WORKERS = None    # None = CPU count
```

### Best Practices

1. **During development**: Leave caching enabled for fast iteration
2. **After code changes**: Clear cache with `python src/pipeline.py cache clear`
3. **For final runs**: Use `--no-cache` to ensure clean computation
4. **Low memory systems**: Use `--sequential` to limit memory usage

---

## QA Reports

Each pipeline stage automatically generates quality assurance reports.

**Output Location:** `data_work/quality/`

**File Pattern:** `{stage_name}_quality_{timestamp}.csv`

**Example:**

```text
data_work/quality/s00_ingest_quality_20251227_143022.csv
data_work/quality/s01_link_quality_20251227_143025.csv
data_work/quality/s02_panel_quality_20251227_143028.csv
```

**Metrics Tracked:**

- Row and column counts
- Missing value percentages
- Duplicate row counts
- Memory usage
- Stage-specific metrics (e.g., linkage rates, estimation sample sizes)

**Configuration:** QA reports are controlled by `ENABLE_QA_REPORTS` in `src/config.py`.

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

PDF/DOCX filenames follow the `book.title` in `manuscript_quarto/_quarto.yml`.

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

## Data Audit

```bash
python src/pipeline.py audit_data
python src/pipeline.py audit_data --full
python src/pipeline.py audit_data --full --report
```

`audit_data` prints a summary to the console and optionally writes a markdown report to `data_work/diagnostics/`.

---

## Common Workflows

### Full Rebuild

```bash
source .venv/bin/activate
# Add --demo to ingest_data if using synthetic data
python src/pipeline.py ingest_data
python src/pipeline.py link_records
python src/pipeline.py build_panel
python src/pipeline.py run_estimation
python src/pipeline.py estimate_robustness
python src/pipeline.py make_figures
cd manuscript_quarto && ./render_all.sh
```

### Update After Data Change

```bash
python src/pipeline.py ingest_data
python src/pipeline.py link_records
python src/pipeline.py build_panel
python src/pipeline.py run_estimation
python src/pipeline.py estimate_robustness
python src/pipeline.py make_figures
```

### Update Figures Only

```bash
python src/pipeline.py make_figures
cd manuscript_quarto && ./render_all.sh
```

---

## Spatial Cross-Validation

When working with geographic data, standard k-fold cross-validation can produce overly optimistic performance estimates due to spatial autocorrelation. The spatial CV module provides tools to address this.

### Why Spatial CV?

Geographic observations that are close to each other tend to be similar (spatial autocorrelation). Standard CV randomly assigns observations to folds, which can:

- Put nearby observations in both training and test sets
- Allow information to "leak" from training to test
- Produce inflated performance metrics

Spatial CV ensures geographic separation between training and test sets.

### Spatial CV Usage

```python
from src.utils.spatial_cv import SpatialCVManager, compare_spatial_vs_random_cv

# Create spatial groups
manager = SpatialCVManager(n_groups=5, method='kmeans')
groups = manager.create_groups_from_coordinates(df['latitude'], df['longitude'])

# Cross-validate with spatial groups
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
results = manager.cross_validate(model, X, y)
print(f"Spatial CV R2: {results['mean']:.3f} +/- {results['std']:.3f}")

# Quantify leakage
comparison = manager.compare_to_random_cv(model, X, y)
print(f"Random CV:  {comparison['random_cv']['mean']:.3f}")
print(f"Spatial CV: {comparison['spatial_cv']['mean']:.3f}")
print(f"Leakage:    {comparison['leakage']:.3f}")
```

### Grouping Methods

| Method | Description | Requirements |
| ------ | ----------- | ------------ |
| `kmeans` | K-means clustering on coordinates | lat/lon |
| `balanced_kmeans` | K-means with balanced group sizes | lat/lon |
| `geographic_bands` | Latitude-based horizontal bands | lat/lon |
| `longitude_bands` | Longitude-based vertical bands | lat/lon |
| `spatial_blocks` | Grid-based spatial blocks | lat/lon |
| `zip_digit` | ZIP code digit-based grouping | zip codes |
| `contiguity_queen` | Polygon contiguity (shared edges/vertices) | geopandas |
| `contiguity_rook` | Polygon contiguity (shared edges only) | geopandas |

### Spatial CV Configuration

Settings in `src/config.py`:

```python
SPATIAL_CV_N_GROUPS = 5          # Number of spatial folds
SPATIAL_GROUPING_METHOD = 'kmeans'  # Default method
SPATIAL_SENSITIVITY_METHODS = [   # Methods for sensitivity analysis
    'kmeans', 'balanced_kmeans', 'geographic_bands',
    'longitude_bands', 'spatial_blocks'
]
```

### Optional Dependencies

Contiguity-based methods require geopandas:

```bash
pip install -r requirements-spatial.txt
```

---

## Minimal Demo

See `demo/README.md` for a small sample dataset and expected outputs that exercise the full data → manuscript path.
