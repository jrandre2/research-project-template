# Extended Analysis Scripts

> **Purpose**: This directory contains standalone analysis scripts that are separate
> from the core pipeline. These scripts use pipeline outputs for exploratory research,
> robustness checks, or specialized analyses.

## Why Separate Scripts?

The core pipeline (`src/stages/s00-s08`) handles:

- Data ingestion and cleaning
- Record linkage and panel construction
- Main estimation and robustness checks
- Figure generation and manuscript validation

Extended scripts handle:

- Exploratory analyses that may not make it into the final paper
- One-off investigations triggered by reviewer comments
- Specialized statistical methods not needed for all projects
- Sensitivity analyses with many parameter combinations

Keeping these separate prevents pipeline bloat and makes it clear which analyses are core vs. exploratory.

## Naming Convention

Scripts should follow the pattern:

```
run_<analysis_name>.py
```

Examples:

- `run_heterogeneity_analysis.py`
- `run_trajectory_clustering.py`
- `run_bootstrap_sensitivity.py`

## Script Template

All scripts should follow this structure:

```python
#!/usr/bin/env python3
"""
[Script Name]: [Brief Description]

Purpose: [What this script does]
Input:   [What data it reads, e.g., data_work/panel.parquet]
Output:  [Where results go, e.g., data_work/exploratory/]

Usage:
    python scripts/run_example.py [options]

Notes:
    This is an extended analysis script, separate from the core pipeline.
    Results should be validated before incorporating into the main manuscript.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import from pipeline modules
from config import DATA_WORK_DIR, FIGURES_DIR
from stages._io_utils import safe_read_parquet


def main():
    """Main entry point."""
    print("=" * 80)
    print("[Script Name]")
    print("=" * 80)

    # Load data
    panel = safe_read_parquet(DATA_WORK_DIR / 'panel.parquet')
    print(f"Loaded {len(panel):,} observations")

    # Analysis here
    # ...

    print("\nDone.")


if __name__ == '__main__':
    main()
```

## Directory Structure

```
scripts/
├── README.md              # This file
├── run_example.py         # Template script
├── run_<analysis1>.py     # Your analyses
├── run_<analysis2>.py
└── ...
```

## Best Practices

### DO

- Include clear docstrings with purpose, inputs, and outputs
- Save results to `data_work/exploratory/` or a designated subdirectory
- Print progress and summary statistics
- Document results in `ANALYSIS_JOURNEY.md`
- Version control your scripts

### DON'T

- Modify data in `data_raw/`
- Overwrite core pipeline outputs without backup
- Leave undocumented magic numbers
- Commit large output files to git

## Output Locations

| Output Type | Location |
|-------------|----------|
| Intermediate data | `data_work/exploratory/` |
| Figures | `figures/exploratory/` |
| Reports | `data_work/reports/` |

## Integration with Pipeline

To promote an exploratory analysis to the core pipeline:

1. Move logic to appropriate stage file (e.g., `s04_robustness.py`)
2. Add command to `src/pipeline.py`
3. Update `doc/PIPELINE.md`
4. Add tests to `tests/test_stages/`
5. Archive the original script or delete if no longer needed

---

*See also: [PIPELINE.md](../doc/PIPELINE.md) for core pipeline documentation*
