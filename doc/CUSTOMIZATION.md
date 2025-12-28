# Customization Guide

This guide explains how to adapt CENTAUR to your specific research project.

## Overview

The platform is designed with clear extension points. Most customizations involve:

1. Modifying configuration values
2. Adding project-specific variables
3. Extending stage implementations
4. Creating custom validators

## Quick Start Checklist

When starting a new project:

- [ ] Update `CLAUDE.md` with project-specific instructions
- [ ] Update `README.md` with project overview
- [ ] Modify variable names in stage configurations
- [ ] Create/update `doc/DATA_DICTIONARY.md`
- [ ] Configure journal settings in `manuscript_quarto/journal_configs/`
- [ ] Customize figure styling in `src/utils/figure_style.py`

## Adapting Stages to Your Data

### Stage 00: Data Ingestion

Edit `src/stages/s00_ingest.py`:

```python
# Change input file configuration
INPUT_PATTERNS = ['*.csv', '*.xlsx']  # Add your file types

# Add project-specific column renaming
COLUMN_RENAMES = {
    'old_column_name': 'new_column_name',
    'PropertyID': 'unit_id',
    'SaleDate': 'date',
    'SalePrice': 'price',
}

# Add required columns for validation
REQUIRED_COLUMNS = ['unit_id', 'date', 'price', 'treatment']
```

Add custom cleaning logic:

```python
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Project-specific data cleaning."""
    df = df.copy()

    # Your cleaning logic
    df['price'] = df['price'].clip(lower=1000)  # Remove unrealistic prices
    df['date'] = pd.to_datetime(df['date'])

    # Create derived variables
    df['log_price'] = np.log(df['price'])
    df['year'] = df['date'].dt.year

    return df
```

### Stage 01: Record Linkage

Edit `src/stages/s01_link.py`:

```python
# Configure matching keys
DEFAULT_KEY_COLUMNS = ['property_id', 'parcel_num']

# Set fuzzy matching threshold
FUZZY_THRESHOLD = 0.85

# Add blocking for large datasets
BLOCKING_COLUMNS = ['zip_code', 'year']
```

For complex linkage:

```python
def link_with_spatial(df_left, df_right, distance_threshold=100):
    """Link records using spatial proximity."""
    # Implement spatial matching
    pass

# In main():
if 'latitude' in df.columns:
    df, spatial_result = link_with_spatial(df, df_spatial)
    linkage_results.append(spatial_result)
```

### Stage 02: Panel Construction

Edit `src/stages/s02_panel.py`:

```python
# Configure panel structure
UNIT_ID = 'parcel_id'        # Your unit identifier
TIME_ID = 'sale_month'       # Your time period
TREATMENT_TIME = 'treat_date'  # Treatment timing column

# Event study window
PRE_PERIODS = 24
POST_PERIODS = 24

# Treatment definition
TREATMENT_VAR = 'in_hazard_zone'
```

Custom panel construction:

```python
def create_custom_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Project-specific panel construction."""
    # Create balanced monthly panel
    df['sale_month'] = df['date'].dt.to_period('M')

    # Define treatment groups
    df['ever_treated'] = df.groupby(UNIT_ID)[TREATMENT_VAR].transform('max')

    # Create cohort identifiers
    df['treatment_cohort'] = df.groupby(UNIT_ID)[TREATMENT_TIME].transform('first')

    return df
```

### Stage 03: Estimation

Edit `src/stages/s03_estimation.py`:

```python
# Define your specification registry
SPECIFICATIONS = {
    'baseline': {
        'name': 'Baseline OLS',
        'outcome': 'log_price',
        'treatment': 'post_treatment',
        'controls': [],
        'fe': [],
        'cluster': None,
        'description': 'Simple treatment effect'
    },
    'with_controls': {
        'name': 'With Property Controls',
        'outcome': 'log_price',
        'treatment': 'post_treatment',
        'controls': ['sqft', 'bedrooms', 'age'],
        'fe': [],
        'cluster': 'zip_code',
        'description': 'Treatment effect with property characteristics'
    },
    'twoway_fe': {
        'name': 'Two-Way Fixed Effects',
        'outcome': 'log_price',
        'treatment': 'post_treatment',
        'controls': [],
        'fe': ['parcel_id', 'sale_month'],
        'cluster': 'parcel_id',
        'description': 'DiD with unit and time FE'
    },
    'event_study': {
        'name': 'Event Study',
        'outcome': 'log_price',
        'treatment': None,  # Use event-time dummies
        'event_time_col': 'event_time',
        'fe': ['parcel_id', 'sale_month'],
        'cluster': 'parcel_id',
        'description': 'Dynamic treatment effects'
    },
}
```

### Stage 04: Robustness Checks

Edit `src/stages/s04_robustness.py`:

```python
# Configure placebo tests
PLACEBO_PERIODS = [6, 8, 10, 12]  # Periods to use as fake treatment

# Sample restriction tests
SAMPLE_TESTS = {
    'no_outliers': lambda df: df[df['price'].between(
        df['price'].quantile(0.01),
        df['price'].quantile(0.99)
    )],
    'post_2010': lambda df: df[df['year'] >= 2010],
    'single_family': lambda df: df[df['property_type'] == 'SFR'],
}
```

### Stage 05: Figure Generation

Edit `src/stages/s05_figures.py`:

```python
# Configure figure output
INPUT_FILE = 'panel.parquet'
OUTCOME_VAR = 'log_price'
TREATMENT_VAR = 'post_treatment'

# Add custom figures
def plot_price_trends(df, output_path):
    """Plot median price by treatment group over time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = get_color_palette('treatment')

    for treat, group in df.groupby('ever_treated'):
        trends = group.groupby('sale_month')['price'].median()
        label = 'Treated' if treat else 'Control'
        ax.plot(trends.index, trends.values, label=label, color=colors[treat])

    ax.set_xlabel('Sale Month')
    ax.set_ylabel('Median Sale Price ($)')
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    save_figure(fig, output_path)
    return output_path

# Register in all_figures
all_figures['price_trends'] = lambda: plot_price_trends(df, fig_dir / 'fig_price_trends')
```

## Adding Custom Validators

Edit `src/utils/validation.py`:

```python
# Add project-specific validation rules
def positive_prices(column: str = 'price') -> ValidationRule:
    """Ensure all prices are positive."""
    return ValidationRule(
        name='positive_prices',
        check=lambda df: (df[column] > 0).all(),
        severity='error',
        message=f'All values in {column} must be positive'
    )

def valid_coordinates() -> ValidationRule:
    """Check latitude/longitude are in valid ranges."""
    def check(df):
        lat_ok = df['latitude'].between(-90, 90).all()
        lon_ok = df['longitude'].between(-180, 180).all()
        return lat_ok and lon_ok

    return ValidationRule(
        name='valid_coordinates',
        check=check,
        severity='error',
        message='Coordinates must be in valid ranges'
    )

def no_future_dates(date_col: str = 'sale_date') -> ValidationRule:
    """Ensure no dates are in the future."""
    from datetime import datetime

    return ValidationRule(
        name='no_future_dates',
        check=lambda df: (df[date_col] <= datetime.now()).all(),
        severity='warning',
        message='Some dates are in the future'
    )
```

Use in ingestion stage:

```python
# In s00_ingest.py
validator = DataValidator()
validator.add_rule(no_missing_values(['unit_id', 'price']))
validator.add_rule(positive_prices('price'))
validator.add_rule(valid_coordinates())
validator.add_rule(no_future_dates('sale_date'))

report = validator.validate(df)
if not report.passed:
    print("Validation failed!")
    print(report.format())
```

## Creating Custom Figures

### Using Figure Style Utilities

```python
from utils.figure_style import (
    apply_style,
    get_color_palette,
    save_figure,
    annotate_panel,
    get_journal_preset
)

# Apply consistent styling
apply_style('publication')

# Use color palettes
colors = get_color_palette('sequential', n=5)
treatment_colors = get_color_palette('treatment')

# Create multi-panel figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, label in zip(axes.flat, ['A', 'B', 'C', 'D']):
    # Plot data
    ...
    # Add panel label
    annotate_panel(ax, label)

# Save with multiple formats
save_figure(fig, 'manuscript_quarto/figures/main_results', formats=['png', 'pdf', 'eps'])
```

### Journal-Specific Formatting

```python
# Get journal presets
preset = get_journal_preset('jeem')
# Returns: {'figwidth': 6.5, 'fontsize': 10, ...}

# Apply to figure
fig, ax = plt.subplots(figsize=(preset['figwidth'], 4))
plt.rcParams['font.size'] = preset['fontsize']
```

### Custom Color Palettes

Add to `src/utils/figure_style.py`:

```python
COLOR_PALETTES = {
    # ... existing palettes ...
    'project_custom': ['#1a5f7a', '#57837b', '#c38e70', '#f1d6ab'],
    'hazard_zones': ['#2ecc71', '#e74c3c'],  # Safe/Hazard
}
```

## Extending the Pipeline CLI

### Adding New Commands

Edit `src/pipeline.py`:

```python
# Add subparser for new command
p_custom = sub.add_parser('custom_analysis', help='Run custom analysis')
p_custom.add_argument('--option1', '-o', default='default', help='Option description')
p_custom.add_argument('--flag', '-f', action='store_true', help='Enable flag')

# Add handler in main()
elif args.cmd == 'custom_analysis':
    from stages import custom_stage
    custom_stage.main(option1=args.option1, flag=args.flag)
```

### Creating a New Stage Module

Create `src/stages/s08_custom.py`:

```python
#!/usr/bin/env python3
"""
Stage 08: Custom Analysis

Purpose: [Describe what this stage does]

Input Files
-----------
- data_work/panel.parquet

Output Files
------------
- data_work/custom_output.parquet
- data_work/diagnostics/custom_results.csv

Usage
-----
    python src/pipeline.py custom_analysis --option1 value
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from utils.helpers import (
    get_data_dir,
    load_data,
    save_data,
    save_diagnostic,
    ensure_dir,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = 'panel.parquet'
OUTPUT_FILE = 'custom_output.parquet'


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def custom_analysis(df: pd.DataFrame, option1: str) -> pd.DataFrame:
    """Perform custom analysis."""
    # Your analysis code
    results = df.groupby('treatment').agg({
        'outcome': ['mean', 'std', 'count']
    }).round(4)

    return results


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(
    option1: str = 'default',
    flag: bool = False,
    verbose: bool = True
):
    """
    Execute custom analysis pipeline.

    Parameters
    ----------
    option1 : str
        Description of option
    flag : bool
        Description of flag
    verbose : bool
        Print detailed output
    """
    print("=" * 60)
    print("Stage 08: Custom Analysis")
    print("=" * 60)

    # Setup paths
    work_dir = get_data_dir('work')
    diag_dir = get_data_dir('diagnostics')
    input_path = work_dir / INPUT_FILE

    ensure_dir(diag_dir)

    # Load data
    print(f"\n  Loading: {INPUT_FILE}")
    df = load_data(input_path)
    print(f"    -> {len(df):,} rows")

    # Run analysis
    print(f"\n  Running custom analysis (option1={option1})...")
    results = custom_analysis(df, option1)

    # Save results
    save_diagnostic(results.reset_index(), 'custom_results')
    print(f"\n  Results saved to: {diag_dir}")

    # Summary
    print("\n" + "-" * 60)
    print("CUSTOM ANALYSIS SUMMARY")
    print("-" * 60)
    print(results.to_string())

    print("\n" + "=" * 60)
    print("Stage 08 complete.")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()
```

## Configuring Journal Settings

### Creating a New Journal Configuration

Create `manuscript_quarto/journal_configs/your_journal.yml`:

```yaml
# Your Journal Configuration

journal:
  name: "Your Target Journal"
  abbreviation: "YTJ"
  publisher: "Publisher Name"
  url: "https://journal.example.com"
  submission_url: "https://submit.example.com"

submission:
  system: "ScholarOne"
  peer_review: "double-blind"

abstract:
  min_words: 150
  max_words: 300
  structured: false

keywords:
  min: 4
  max: 6

text_limits:
  word_limit: 10000

references:
  style: "author-year"
  csl_file: "apa.csl"

artwork:
  formats:
    vector_preferred: ["PDF", "EPS"]
    raster_acceptable: ["PNG", "TIFF"]
  resolution:
    line_art_dpi: 1200
    halftone_dpi: 300
  dimensions:
    max_width_mm: 170

declarations:
  funding:
    required: true
  competing_interests:
    required: true
  data_availability:
    required: true
```

### Creating a Quarto Profile

Create `manuscript_quarto/_quarto-yourjournal.yml`:

```yaml
# Quarto profile for Your Journal
project:
  output-dir: _output/yourjournal

format:
  pdf:
    documentclass: article
    papersize: letter
    fontsize: 12pt
    linestretch: 2.0
    geometry:
      - margin=1in
    include-in-header:
      text: |
        \usepackage{setspace}
        \doublespacing

  docx:
    reference-doc: reference-yourjournal.docx

bibliography: references.bib
csl: csl/your-journal-style.csl
```

Use with:

```bash
quarto render --profile yourjournal
```

For divergent drafts, use `manuscript_quarto/variant_new.sh` and see `doc/MANUSCRIPT_VARIANTS.md`.

## Modifying Synthetic Data

Edit `src/utils/synthetic_data.py`:

```python
class SyntheticDataGenerator:
    """Generate synthetic data matching your project structure."""

    def generate_panel(
        self,
        n_units: int = 1000,
        n_periods: int = 24,
        treatment_share: float = 0.3,
        treatment_effect: float = 0.1,
        # Add project-specific parameters
        include_spatial: bool = True,
        include_property_chars: bool = True
    ) -> pd.DataFrame:
        """Generate panel data matching your data structure."""

        df = pd.DataFrame({
            'parcel_id': np.repeat(range(n_units), n_periods),
            'sale_month': np.tile(range(n_periods), n_units),
        })

        # Add property characteristics
        if include_property_chars:
            np.random.seed(self.seed)
            df['sqft'] = np.random.normal(1500, 500, len(df)).clip(500, 5000)
            df['bedrooms'] = np.random.choice([2, 3, 4, 5], len(df), p=[0.2, 0.4, 0.3, 0.1])
            df['age'] = np.random.exponential(30, len(df)).clip(0, 100)

        # Add spatial data
        if include_spatial:
            df['latitude'] = np.random.uniform(25, 48, n_units)[df['parcel_id']]
            df['longitude'] = np.random.uniform(-125, -70, n_units)[df['parcel_id']]

        # Generate outcome with treatment effect
        # ... (project-specific logic)

        return df
```

## Adding Domain-Specific Functionality

### Economics/Causal Inference Add-on

Create `src/addons/economics/__init__.py`:

```python
"""Economics/causal inference utilities."""

from .estimators import (
    DiDEstimator,
    RDEstimator,
    EventStudyEstimator,
    IVEstimator,
)

from .diagnostics import (
    PreTrendTest,
    BalanceTest,
    McCraryTest,
)

from .formatters import (
    format_regression_table,
    format_stargazer_style,
)

__all__ = [
    'DiDEstimator',
    'RDEstimator',
    'EventStudyEstimator',
    'IVEstimator',
    'PreTrendTest',
    'BalanceTest',
    'McCraryTest',
    'format_regression_table',
    'format_stargazer_style',
]
```

### Spatial Analysis Add-on

Create `src/addons/spatial/__init__.py`:

```python
"""Spatial analysis utilities."""

from .distance import (
    haversine_distance,
    calculate_nearest_neighbor,
    create_distance_matrix,
)

from .clustering import (
    spatial_cluster,
    conley_standard_errors,
)

from .visualization import (
    plot_choropleth,
    plot_point_map,
    create_boundary_buffer,
)

__all__ = [
    'haversine_distance',
    'calculate_nearest_neighbor',
    'create_distance_matrix',
    'spatial_cluster',
    'conley_standard_errors',
    'plot_choropleth',
    'plot_point_map',
    'create_boundary_buffer',
]
```

## Testing Your Customizations

### Adding Project-Specific Tests

Create `tests/test_stages/test_custom_stage.py`:

```python
"""Tests for custom stage."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from stages.s08_custom import custom_analysis, main


class TestCustomAnalysis:
    """Tests for custom_analysis function."""

    def test_basic_analysis(self, panel_df):
        """Test basic analysis runs without error."""
        result = custom_analysis(panel_df, option1='default')
        assert result is not None

    def test_output_structure(self, panel_df):
        """Test output has expected structure."""
        result = custom_analysis(panel_df, option1='default')
        assert 'mean' in result.columns.get_level_values(1)
        assert 'count' in result.columns.get_level_values(1)


class TestCustomMain:
    """Tests for main function."""

    def test_main_runs(self, temp_data_dir, panel_df):
        """Test main function runs without error."""
        # Setup
        panel_df.to_parquet(temp_data_dir / 'panel.parquet')

        # Run
        result = main(option1='test', verbose=False)

        # Verify
        assert result is not None
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run only custom tests
pytest tests/test_stages/test_custom_stage.py

# With verbose output
pytest tests/ -v

# Stop on first failure
pytest tests/ -x
```

## Best Practices

### 1. Version Control Your Customizations

```bash
# Create a branch for customizations
git checkout -b project/my-research

# Commit changes
git add -A
git commit -m "Customize pipeline for property price analysis"
```

### 2. Document Your Changes

Update relevant documentation:
- `doc/DATA_DICTIONARY.md` - Variable definitions
- `doc/METHODOLOGY.md` - Statistical methods
- `CLAUDE.md` - AI assistant instructions

### 3. Keep Platform Updates Separate

```bash
# Add platform as remote
git remote add platform https://github.com/platform-repo.git

# Fetch updates
git fetch platform

# Merge platform updates (careful with conflicts)
git merge platform/main --allow-unrelated-histories
```

### 4. Use Configuration Files

Instead of hardcoding values:

```python
# config.py
PROJECT_CONFIG = {
    'unit_id': 'parcel_id',
    'time_id': 'sale_month',
    'outcome': 'log_price',
    'treatment': 'post_flood_zone',
    'treatment_effect_expected': -0.05,
}
```

Import in stages:

```python
from config import PROJECT_CONFIG

UNIT_ID = PROJECT_CONFIG['unit_id']
OUTCOME_VAR = PROJECT_CONFIG['outcome']
```
