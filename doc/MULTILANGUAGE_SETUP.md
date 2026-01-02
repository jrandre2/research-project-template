# Multilanguage Analysis Setup

CENTAUR supports multiple analysis engines for estimation:

- **Python** (default): Native NumPy/Pandas implementation
- **R**: Uses fixest package for high-performance fixed effects

## Quick Start

```bash
# Check available engines
python src/pipeline.py engines list

# Detailed validation
python src/pipeline.py engines check

# Run estimation with specific engine
python src/pipeline.py run_estimation --engine r --specification baseline
```

## Python Engine (Default)

The Python engine is always available and requires no additional setup.

**Features:**

- Native NumPy matrix operations
- Within-transformation for fixed effects
- Clustered standard errors
- Parallel execution support

## R Engine Setup

### Requirements

- R >= 4.0
- Packages: arrow, fixest, jsonlite

### Installation

**macOS:**

```bash
# Install R via Homebrew
brew install r

# Install R packages
Rscript -e "install.packages(c('arrow', 'fixest', 'jsonlite'))"
```

**Ubuntu/Debian:**

```bash
# Install R
sudo apt-get update
sudo apt-get install r-base

# Install R packages
Rscript -e "install.packages(c('arrow', 'fixest', 'jsonlite'))"
```

**Windows:**

1. Download and install R from https://cran.r-project.org/
2. Open R and run:
   ```r
   install.packages(c("arrow", "fixest", "jsonlite"))
   ```

### Verify Installation

```bash
python src/pipeline.py engines check
```

Expected output:

```
r:
  Status:  OK
  Version: R version 4.x.x
  Details: R engine ready (arrow:x.x.x, fixest:x.x.x, jsonlite:x.x.x)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `R_EXECUTABLE` | `Rscript` | Path to Rscript executable |
| `EXTERNAL_PROCESS_TIMEOUT` | `3600` | Timeout in seconds |

Example:

```bash
export R_EXECUTABLE=/usr/local/bin/Rscript
```

## Configuration

Settings in `src/config.py`:

```python
# Default engine
ANALYSIS_ENGINE = os.getenv('CENTAUR_ANALYSIS_ENGINE', 'python')

# External tool paths
R_EXECUTABLE = os.getenv('R_EXECUTABLE', 'Rscript')

# Process timeout
EXTERNAL_PROCESS_TIMEOUT = 3600  # seconds
```

## Usage

### CLI

```bash
# Use default engine (from config)
python src/pipeline.py run_estimation --specification baseline

# Explicitly use R
python src/pipeline.py run_estimation --engine r --specification baseline

# Run all specifications with R
python src/pipeline.py run_estimation --engine r --all
```

### Python API

```python
from analysis import get_engine, list_engines

# List available engines
engines = list_engines()
# {'python': True, 'r': True}

# Get an engine
engine = get_engine('r')

# Validate installation
available, message = engine.validate_installation()

# Run estimation
from pathlib import Path

result = engine.estimate(
    data_path=Path('data_work/panel.parquet'),
    specification={
        'name': 'baseline',
        'outcome': 'outcome',
        'treatment': 'treatment',
        'controls': [],
        'fixed_effects': ['unit_fe', 'time_fe'],
        'cluster': 'id',
    },
    output_dir=Path('data_work/diagnostics'),
)

print(f"Coefficient: {result.coefficients['treatment']}")
print(f"Std Error: {result.std_errors['treatment']}")
```

## Specifications

Specifications are defined in `specifications.yml` at the project root:

```yaml
baseline:
  outcome: outcome
  treatment: treatment
  controls: []
  fixed_effects:
    - unit_fe
    - time_fe
  cluster: id
  description: Baseline specification

with_controls:
  outcome: outcome
  treatment: treatment
  controls:
    - covariate_1
    - covariate_2
  fixed_effects:
    - unit_fe
    - time_fe
  cluster: id
  description: With additional controls
```

## Cross-Engine Validation

To verify engines produce consistent results:

```python
from analysis import get_engine
from pathlib import Path

python_engine = get_engine('python')
r_engine = get_engine('r')

spec = {
    'name': 'baseline',
    'outcome': 'outcome',
    'treatment': 'treatment',
    'fixed_effects': ['unit_fe', 'time_fe'],
    'cluster': 'id',
}

python_result = python_engine.estimate(
    Path('data_work/panel.parquet'), spec, Path('/tmp/python')
)
r_result = r_engine.estimate(
    Path('data_work/panel.parquet'), spec, Path('/tmp/r')
)

# Compare coefficients (should be within 1%)
python_coef = python_result.coefficients['treatment']
r_coef = r_result.coefficients['treatment']
diff_pct = abs(python_coef - r_coef) / abs(python_coef) * 100
print(f"Coefficient difference: {diff_pct:.2f}%")
```

## Troubleshooting

### R Engine Not Found

```
R not found at 'Rscript'. Set R_EXECUTABLE env var.
```

**Solution:** Set the R_EXECUTABLE environment variable:

```bash
export R_EXECUTABLE=/path/to/Rscript
```

### Missing R Packages

```
R packages check failed: MISSING: arrow, fixest
```

**Solution:** Install missing packages:

```bash
Rscript -e "install.packages(c('arrow', 'fixest', 'jsonlite'))"
```

### Timeout Errors

```
R estimation timed out after 3600s
```

**Solution:** Increase timeout:

```bash
export EXTERNAL_PROCESS_TIMEOUT=7200  # 2 hours
```

Or in config.py:

```python
EXTERNAL_PROCESS_TIMEOUT = 7200
```
