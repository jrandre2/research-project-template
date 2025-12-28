# Stage Development Guide

**Related**: [PIPELINE.md](PIPELINE.md) | [ARCHITECTURE.md](ARCHITECTURE.md) | [TESTING.md](TESTING.md)
**Status**: Active
**Last Updated**: 2025-12-27

---

## Overview

Pipeline stages are modular Python scripts in `src/stages/` that process data sequentially. Each stage reads input, performs operations, and writes output.

---

## Stage Structure

### File Naming Convention

```
src/stages/
├── s00_ingest.py      # Stage 00: Data ingestion
├── s01_link.py        # Stage 01: Record linkage
├── s02_panel.py       # Stage 02: Panel construction
├── s03_estimation.py  # Stage 03: Estimation
├── ...
└── _qa_utils.py       # Shared QA utilities (underscore = not a stage)
```

### Version Suffixes

Stages can have version suffixes for evolution:
- `s00_ingest.py` → Original version
- `s00b_standardize.py` → Alternative approach
- `s00c_enhanced.py` → Enhanced version

---

## Stage Template

```python
#!/usr/bin/env python3
"""
Stage XX: Stage Name

Purpose: Brief description of what this stage does.

This stage handles:
- Task 1
- Task 2
- Task 3

Input Files
-----------
- data_work/input_file.parquet

Output Files
------------
- data_work/output_file.parquet
- data_work/diagnostics/stage_summary.csv

Usage
-----
    python src/pipeline.py <command>
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys

# Add parent directory for imports
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
from stages._qa_utils import qa_for_stage


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = 'input_file.parquet'
OUTPUT_FILE = 'output_file.parquet'

# Stage-specific configuration
REQUIRED_COLUMNS = ['id', 'period']
COLUMN_TYPES = {
    'id': 'int64',
    'period': 'int64',
}


# ============================================================
# PROCESSING FUNCTIONS
# ============================================================

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main processing logic.

    Parameters
    ----------
    df : pd.DataFrame
        Input data

    Returns
    -------
    pd.DataFrame
        Processed data
    """
    # Processing logic here
    return df


def validate_output(df: pd.DataFrame) -> bool:
    """
    Validate output data.

    Returns
    -------
    bool
        True if validation passes
    """
    # Validation logic here
    return True


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(verbose: bool = True):
    """
    Execute stage pipeline.

    Parameters
    ----------
    verbose : bool
        Print detailed output
    """
    print("=" * 60)
    print("Stage XX: Stage Name")
    print("=" * 60)

    # Setup paths
    work_dir = get_data_dir('work')
    diag_dir = get_data_dir('diagnostics')
    input_path = work_dir / INPUT_FILE
    output_path = work_dir / OUTPUT_FILE

    # Load input
    print(f"\n  Loading: {INPUT_FILE}")
    if not input_path.exists():
        print(f"  ERROR: Input file not found: {input_path}")
        sys.exit(1)

    df = load_data(input_path)
    print(f"    -> {len(df):,} rows, {len(df.columns)} columns")

    # Process
    print("\n  Processing data...")
    df = process_data(df)

    # Validate
    if not validate_output(df):
        print("\nERROR: Validation failed.")
        sys.exit(1)

    # Save output
    print(f"\n  Saving to: {output_path}")
    save_data(df, output_path)

    # Generate QA report
    qa_for_stage('sXX_name', df, output_file=str(output_path))

    print("\n" + "=" * 60)
    print("Stage XX complete.")
    print("=" * 60)

    return df


if __name__ == '__main__':
    main()
```

---

## Required Constants

Each stage should define these constants:

| Constant | Purpose | Example |
|----------|---------|---------|
| `INPUT_FILE` | Input file name | `'panel.parquet'` |
| `OUTPUT_FILE` | Output file name | `'estimation_results.csv'` |
| `REQUIRED_COLUMNS` | Columns that must exist | `['id', 'period']` |
| `COLUMN_TYPES` | Expected column types | `{'id': 'int64'}` |

---

## Connecting to Pipeline

### Add Command to pipeline.py

```python
# In src/pipeline.py

@cli.command()
@click.option('--verbose', '-v', is_flag=True)
def my_new_command(verbose):
    """Brief description of command."""
    from stages.sXX_name import main
    main(verbose=verbose)
```

### Register in Stage Discovery

Stages are auto-discovered by naming convention. Ensure your file:
1. Is named `sXX_*.py` (where XX is a number)
2. Has a `main()` function
3. Is in `src/stages/`

---

## QA Reports

Use the QA utilities to generate quality reports:

```python
from stages._qa_utils import qa_for_stage

# At end of main()
qa_for_stage(
    stage_name='s00_ingest',
    df=df,
    additional_metrics={'custom_metric': value},
    output_file=str(output_path)
)
```

Reports are saved to `data_work/quality/`.

---

## Testing Stages

Create tests in `tests/test_stages/test_sXX_name.py`:

```python
#!/usr/bin/env python3
"""Tests for src/stages/sXX_name.py"""
import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from stages.sXX_name import process_data, validate_output


class TestProcessData:
    """Tests for the process_data function."""

    def test_basic_processing(self, sample_df):
        """Test basic processing works."""
        result = process_data(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_required_columns(self, sample_df):
        """Test required columns are preserved."""
        result = process_data(sample_df)
        for col in ['id', 'period']:
            assert col in result.columns
```

---

## Best Practices

### 1. Idempotency

Stages should be safe to re-run:
```python
# Always overwrite output
save_data(df, output_path)  # Not append
```

### 2. Error Handling

Fail fast with clear messages:
```python
if not input_path.exists():
    print(f"ERROR: File not found: {input_path}")
    print("Run 'previous_stage' first.")
    sys.exit(1)
```

### 3. Progress Reporting

Print progress for long operations:
```python
print("  Processing data...")
for i, chunk in enumerate(chunks):
    print(f"    Chunk {i+1}/{len(chunks)}")
    process(chunk)
```

### 4. Diagnostics

Save diagnostic outputs:
```python
# Save summary statistics
summary_df = pd.DataFrame([stats])
save_diagnostic(summary_df, 'stage_summary')
```

### 5. Documentation

Document module, functions, and constants:
```python
"""
Stage docstring with:
- Purpose
- Input/output files
- Usage examples
"""

def function(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function docstring with:
    - Description
    - Parameters
    - Returns
    """
```

---

## Common Patterns

### Loading with Fallback

```python
def load_input():
    work_dir = get_data_dir('work')
    if (work_dir / 'preferred.parquet').exists():
        return load_data(work_dir / 'preferred.parquet')
    elif (work_dir / 'fallback.parquet').exists():
        return load_data(work_dir / 'fallback.parquet')
    else:
        raise FileNotFoundError("No input file found")
```

### Validation with DataValidator

```python
from utils.validation import DataValidator, no_missing_values, row_count

def validate(df):
    validator = DataValidator()
    validator.add_rule(row_count(min_rows=10))
    validator.add_rule(no_missing_values(['id', 'period']))

    report = validator.validate(df)
    if report.has_errors:
        print(report.format())
        return False
    return True
```

### Dataclass for Results

```python
from dataclasses import dataclass

@dataclass
class StageResult:
    n_processed: int
    n_errors: int
    duration_seconds: float

    def to_dict(self):
        return {
            'n_processed': self.n_processed,
            'n_errors': self.n_errors,
            'duration': self.duration_seconds
        }
```
