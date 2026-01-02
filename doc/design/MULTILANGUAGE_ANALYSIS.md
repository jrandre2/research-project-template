# Multilanguage Analysis Engine Support

**Related**: [ARCHITECTURE.md](../ARCHITECTURE.md) | [PIPELINE.md](../PIPELINE.md) | [MULTILANGUAGE_SETUP.md](../MULTILANGUAGE_SETUP.md)
**Status**: In Progress (R complete, Stata/Julia planned)
**Last Updated**: 2026-01-02

---

## Overview

This design document details how to add R, Stata, and Julia support to CENTAUR's estimation pipeline while maintaining the current Python implementation as the default.

**Goal:** Allow researchers to run estimation in their preferred language while keeping the pipeline orchestration in Python.

**Principle:** Design the abstraction for all languages, implement R first (highest ROI), add Stata/Julia incrementally.

---

## Motivation

CENTAUR currently requires Python for all analysis. Many academic researchers prefer or require:

- **R**: Dominant in statistics, has excellent `fixest` package for fast fixed effects
- **Stata**: Standard in economics, required by some journals for replication
- **Julia**: Emerging choice for performance-critical numerical work

Adding multilanguage support expands CENTAUR's audience without abandoning Python's strengths for orchestration.

---

## Architecture

### Design Pattern

The architecture follows the same Protocol-based pattern used for LLM providers (`src/llm/base.py`):

```
┌─────────────────────────────────────────────────────────┐
│                    src/pipeline.py                       │
│                  (CLI + Orchestration)                   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 src/analysis/factory.py                  │
│              get_engine() / list_engines()               │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  Python  │    │    R     │    │  Stata   │  ...
    │  Engine  │    │  Engine  │    │  Engine  │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         ▼               ▼               ▼
    NumPy/Pandas    Rscript +       stata -b +
                    fixest          reghdfe
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `src/analysis/base.py` | `AnalysisEngine` Protocol + `EstimationResult` dataclass |
| `src/analysis/factory.py` | Engine registry with `@register_engine` decorator |
| `src/analysis/specifications.py` | Load specifications from YAML |
| `specifications.yml` | Language-agnostic specification definitions |
| `src/analysis/engines/*.py` | Engine implementations |

---

## Phase 1: Analysis Engine Abstraction

### 1.1 Protocol Definition

**New file:** `src/analysis/base.py`

```python
from typing import Protocol, Optional, runtime_checkable
from pathlib import Path
from dataclasses import dataclass

@dataclass
class EstimationResult:
    """Language-agnostic estimation result container."""
    specification: str
    n_obs: int
    n_units: int
    n_periods: int
    coefficients: dict[str, float]      # var_name -> coefficient
    std_errors: dict[str, float]        # var_name -> SE
    p_values: dict[str, float]          # var_name -> p-value
    confidence_intervals: dict[str, tuple[float, float]]
    r_squared: float
    r_squared_within: Optional[float]
    fixed_effects: list[str]
    cluster_var: Optional[str]
    warnings: list[str]
    engine: str                          # 'python', 'r', 'stata', 'julia'
    engine_version: str
    execution_time_seconds: float

@runtime_checkable
class AnalysisEngine(Protocol):
    """Protocol for estimation engines across languages."""

    @property
    def name(self) -> str:
        """Engine identifier (e.g., 'python', 'r', 'stata', 'julia')."""
        ...

    @property
    def version(self) -> str:
        """Engine/package version string."""
        ...

    def validate_installation(self) -> tuple[bool, str]:
        """Check if engine is properly installed.

        Returns:
            (success, message) - message explains what's missing if failed
        """
        ...

    def estimate(
        self,
        data_path: Path,
        specification: dict,
        output_dir: Path,
    ) -> EstimationResult:
        """Run estimation and return results.

        Args:
            data_path: Path to input Parquet file
            specification: Dict with keys: name, outcome, treatment,
                          controls, fe, cluster, description
            output_dir: Directory for any auxiliary outputs

        Returns:
            EstimationResult with coefficients, SEs, diagnostics
        """
        ...

    def estimate_batch(
        self,
        data_path: Path,
        specifications: list[dict],
        output_dir: Path,
        parallel: bool = True,
        n_workers: Optional[int] = None,
    ) -> list[EstimationResult]:
        """Run multiple specifications (enables language-native parallelism).

        Default implementation calls estimate() in loop.
        Override for language-native batch processing.
        """
        ...
```

### 1.2 Engine Factory

**New file:** `src/analysis/factory.py`

```python
from typing import Optional
from .base import AnalysisEngine
from config import ANALYSIS_ENGINE

_engine_registry: dict[str, type[AnalysisEngine]] = {}

def register_engine(name: str):
    """Decorator to register an engine implementation."""
    def decorator(cls):
        _engine_registry[name] = cls
        return cls
    return decorator

def get_engine(name: Optional[str] = None) -> AnalysisEngine:
    """Get configured or specified analysis engine."""
    engine_name = name or ANALYSIS_ENGINE
    if engine_name not in _engine_registry:
        available = ', '.join(_engine_registry.keys())
        raise ValueError(f"Unknown engine '{engine_name}'. Available: {available}")
    return _engine_registry[engine_name]()

def list_engines() -> dict[str, bool]:
    """Return dict of engine_name -> is_available."""
    return {
        name: cls().validate_installation()[0]
        for name, cls in _engine_registry.items()
    }
```

### 1.3 Configuration Updates

**Modify:** `src/config.py`

```python
# === Analysis Engine Settings ===
ANALYSIS_ENGINE = os.getenv('CENTAUR_ANALYSIS_ENGINE', 'python')

# External tool paths (auto-detected if not set)
R_EXECUTABLE = os.getenv('R_EXECUTABLE', 'Rscript')
STATA_EXECUTABLE = os.getenv('STATA_EXECUTABLE', 'stata-mp')
JULIA_EXECUTABLE = os.getenv('JULIA_EXECUTABLE', 'julia')

# Timeout for external processes (seconds)
EXTERNAL_PROCESS_TIMEOUT = int(os.getenv('EXTERNAL_PROCESS_TIMEOUT', '3600'))
```

### 1.4 Specification Format

**New file:** `specifications.yml` (project root)

```yaml
# Estimation specifications in language-agnostic format
# Used by all analysis engines

baseline:
  name: Baseline
  description: Baseline specification with unit and time fixed effects
  outcome: outcome_var
  treatment: treatment_var
  controls: []
  fixed_effects:
    - unit_id
    - time_id
  cluster: unit_id

with_controls:
  name: With Controls
  description: Baseline plus demographic controls
  outcome: outcome_var
  treatment: treatment_var
  controls:
    - control_1
    - control_2
    - control_3
  fixed_effects:
    - unit_id
    - time_id
  cluster: unit_id

event_study:
  name: Event Study
  description: Dynamic treatment effects
  outcome: outcome_var
  treatment: treatment_var
  event_time_var: event_time
  reference_period: -1
  controls: []
  fixed_effects:
    - unit_id
    - time_id
  cluster: unit_id
```

---

## Phase 2: Python Engine (Refactor Existing)

### 2.1 Extract Current Implementation

**New file:** `src/analysis/engines/python_engine.py`

Refactor existing code from `src/stages/s03_estimation.py`:

```python
from ..base import AnalysisEngine, EstimationResult
from ..factory import register_engine
import pandas as pd
import numpy as np
from pathlib import Path
import time

@register_engine('python')
class PythonEngine:
    """Native Python/NumPy estimation engine."""

    @property
    def name(self) -> str:
        return 'python'

    @property
    def version(self) -> str:
        return f"numpy {np.__version__}"

    def validate_installation(self) -> tuple[bool, str]:
        try:
            import numpy
            import pandas
            return True, f"numpy {numpy.__version__}, pandas {pandas.__version__}"
        except ImportError as e:
            return False, str(e)

    def estimate(
        self,
        data_path: Path,
        specification: dict,
        output_dir: Path,
    ) -> EstimationResult:
        start = time.time()
        df = pd.read_parquet(data_path)

        # ... existing estimation logic from s03_estimation.py ...

        return EstimationResult(
            specification=specification['name'],
            n_obs=n_obs,
            # ... fill all fields ...
            engine='python',
            engine_version=self.version,
            execution_time_seconds=time.time() - start,
        )
```

### 2.2 Update s03_estimation.py

```python
from analysis.factory import get_engine
from analysis.specifications import get_specification, load_specifications

def main(
    specification: str = None,
    run_all: bool = False,
    engine: str = None,  # NEW: override default engine
    **kwargs
):
    """Run estimation using configured or specified engine."""

    analysis_engine = get_engine(engine)

    # Validate engine is available
    available, msg = analysis_engine.validate_installation()
    if not available:
        raise RuntimeError(f"Engine '{analysis_engine.name}' not available: {msg}")

    # Load specifications
    if run_all:
        specs = load_specifications()
    else:
        specs = {specification: get_specification(specification)}

    # Run estimation
    results = []
    for spec_name, spec in specs.items():
        result = analysis_engine.estimate(
            data_path=DATA_WORK_DIR / 'panel.parquet',
            specification=spec,
            output_dir=DATA_WORK_DIR / 'estimation',
        )
        results.append(result)
```

### 2.3 CLI Updates

**Modify:** `src/pipeline.py`

```python
# Add --engine flag to estimation commands
estimation_parser.add_argument(
    '--engine',
    choices=['python', 'r', 'stata', 'julia'],
    default=None,
    help='Analysis engine (default: from config)'
)

# Add engine management commands
engine_parser = subparsers.add_parser('engines', help='Manage analysis engines')
engine_subparsers = engine_parser.add_subparsers(dest='engine_cmd')

engine_subparsers.add_parser('list', help='List available engines')
engine_subparsers.add_parser('check', help='Validate engine installations')
```

---

## Phase 3: R Engine Implementation

### 3.1 R Wrapper Script

**New file:** `src/analysis/engines/r/estimate.R`

```r
#!/usr/bin/env Rscript
# CENTAUR R Estimation Engine
# Usage: Rscript estimate.R <data.parquet> <spec.json> <output_dir>

library(arrow)
library(fixest)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
spec_path <- args[2]
output_dir <- args[3]

# Load data and specification
df <- read_parquet(data_path)
spec <- fromJSON(spec_path)

# Build formula
build_formula <- function(spec) {
  lhs <- spec$outcome
  rhs <- spec$treatment

  if (length(spec$controls) > 0) {
    rhs <- paste(c(rhs, spec$controls), collapse = " + ")
  }

  if (length(spec$fixed_effects) > 0) {
    fe_part <- paste(spec$fixed_effects, collapse = " + ")
    rhs <- paste0(rhs, " | ", fe_part)
  }

  as.formula(paste(lhs, "~", rhs))
}

# Run estimation
start_time <- Sys.time()
formula <- build_formula(spec)

model <- feols(
  formula,
  data = df,
  cluster = if (!is.null(spec$cluster)) as.formula(paste0("~", spec$cluster)) else NULL
)

end_time <- Sys.time()

# Extract and write results
result <- list(
  specification = spec$name,
  n_obs = model$nobs,
  coefficients = as.list(coef(model)),
  std_errors = as.list(se(model)),
  p_values = as.list(pvalue(model)),
  r_squared = r2(model, type = "r2"),
  engine = "r",
  engine_version = paste0("fixest ", packageVersion("fixest")),
  execution_time_seconds = as.numeric(difftime(end_time, start_time, units = "secs"))
)

output_path <- file.path(output_dir, paste0("result_", spec$id, ".json"))
write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE)
cat(output_path)
```

### 3.2 Python R Engine Wrapper

**New file:** `src/analysis/engines/r_engine.py`

```python
import subprocess
import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from ..base import AnalysisEngine, EstimationResult
from ..factory import register_engine
from config import R_EXECUTABLE, EXTERNAL_PROCESS_TIMEOUT

@register_engine('r')
class REngine:
    """R/fixest estimation engine."""

    def __init__(self):
        self._r_script = Path(__file__).parent / 'r' / 'estimate.R'

    @property
    def name(self) -> str:
        return 'r'

    @property
    def version(self) -> str:
        try:
            result = subprocess.run(
                [R_EXECUTABLE, '--version'],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout.split('\n')[0]
        except Exception:
            return 'unknown'

    def validate_installation(self) -> tuple[bool, str]:
        if not shutil.which(R_EXECUTABLE):
            return False, f"R not found at '{R_EXECUTABLE}'"

        check_script = '''
        packages <- c("arrow", "fixest", "jsonlite")
        missing <- packages[!packages %in% installed.packages()[,"Package"]]
        if (length(missing) > 0) {
            cat(paste("Missing packages:", paste(missing, collapse=", ")))
            quit(status=1)
        }
        cat("OK")
        '''
        try:
            result = subprocess.run(
                [R_EXECUTABLE, '-e', check_script],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return False, result.stdout + result.stderr
            return True, "R with arrow, fixest, jsonlite"
        except Exception as e:
            return False, str(e)

    def estimate(self, data_path: Path, specification: dict, output_dir: Path) -> EstimationResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(specification, f)
            spec_path = f.name

        try:
            result = subprocess.run(
                [R_EXECUTABLE, str(self._r_script), str(data_path), spec_path, str(output_dir)],
                capture_output=True, text=True,
                timeout=EXTERNAL_PROCESS_TIMEOUT
            )

            if result.returncode != 0:
                raise RuntimeError(f"R estimation failed:\n{result.stderr}")

            output_path = Path(result.stdout.strip())
            with open(output_path) as f:
                data = json.load(f)

            return EstimationResult(**data)
        finally:
            Path(spec_path).unlink(missing_ok=True)
```

---

## Phase 4: Stata Engine Implementation

### 4.1 Stata Do-File Template

**New file:** `src/analysis/engines/stata/estimate.do`

```stata
* CENTAUR Stata Estimation Engine
* Called via: stata -b do estimate.do data_path spec_path output_dir

args data_path spec_path output_dir

* Load data
import parquet "`data_path'", clear

* Run estimation with reghdfe
reghdfe `outcome' `treatment' `controls', absorb(`fe') vce(cluster `cluster')

* Export results to JSON
* ... result extraction and JSON writing ...
```

### 4.2 Python Stata Engine Wrapper

**New file:** `src/analysis/engines/stata_engine.py`

```python
@register_engine('stata')
class StataEngine:
    """Stata/reghdfe estimation engine."""

    def validate_installation(self) -> tuple[bool, str]:
        if not shutil.which(STATA_EXECUTABLE):
            return False, f"Stata not found at '{STATA_EXECUTABLE}'"
        return True, "Stata found (reghdfe check skipped)"

    # ... similar structure to R engine ...
```

---

## Phase 5: Julia Engine Implementation

### 5.1 Julia Script

**New file:** `src/analysis/engines/julia/estimate.jl`

```julia
# CENTAUR Julia Estimation Engine
using Arrow, JSON3, FixedEffectModels, DataFrames

function main(data_path, spec_path, output_dir)
    df = Arrow.Table(data_path) |> DataFrame
    spec = JSON3.read(read(spec_path, String))

    # Build and run model using FixedEffectModels.jl
    # ... estimation logic ...

    output_path = joinpath(output_dir, "result_$(spec.id).json")
    open(output_path, "w") do f
        JSON3.write(f, result)
    end

    println(output_path)
end

main(ARGS...)
```

---

## Phase 6: Testing Strategy

### 6.1 Engine Conformance Tests

**New file:** `tests/test_analysis_engines.py`

```python
import pytest
from analysis.factory import get_engine, list_engines
from analysis.base import EstimationResult

@pytest.fixture
def sample_panel(tmp_path):
    """Create sample panel data for testing."""
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    n_units, n_periods = 100, 10

    df = pd.DataFrame({
        'unit_id': np.repeat(range(n_units), n_periods),
        'time_id': np.tile(range(n_periods), n_units),
        'treatment': np.random.binomial(1, 0.3, n_units * n_periods),
        'outcome': np.random.normal(0, 1, n_units * n_periods),
    })

    path = tmp_path / 'panel.parquet'
    df.to_parquet(path)
    return path

class TestEngineConformance:
    """All engines must pass these tests."""

    @pytest.mark.parametrize('engine_name', ['python', 'r', 'stata', 'julia'])
    def test_result_structure(self, engine_name, sample_panel, tmp_path):
        """Engine returns valid EstimationResult."""
        engines = list_engines()
        if not engines.get(engine_name, False):
            pytest.skip(f"{engine_name} not available")

        engine = get_engine(engine_name)
        result = engine.estimate(sample_panel, baseline_spec, tmp_path)

        assert isinstance(result, EstimationResult)
        assert result.n_obs > 0
        assert result.engine == engine_name

    @pytest.mark.parametrize('engine_name', ['python', 'r', 'stata', 'julia'])
    def test_coefficient_consistency(self, engine_name, sample_panel, tmp_path):
        """Coefficients should be similar across engines (within tolerance)."""
        python_engine = get_engine('python')
        python_result = python_engine.estimate(sample_panel, baseline_spec, tmp_path)

        engines = list_engines()
        if not engines.get(engine_name, False):
            pytest.skip(f"{engine_name} not available")

        engine = get_engine(engine_name)
        result = engine.estimate(sample_panel, baseline_spec, tmp_path)

        # Coefficients should match within 1%
        python_coef = python_result.coefficients['treatment']
        other_coef = result.coefficients['treatment']
        assert abs(python_coef - other_coef) / abs(python_coef) < 0.01
```

### 6.2 CI Configuration

**Modify:** `.github/workflows/test.yml`

```yaml
jobs:
  test-r-engine:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: r-lib/actions/setup-r@v2
      - name: Install R packages
        run: install.packages(c("arrow", "fixest", "jsonlite"))
        shell: Rscript {0}
      - name: Run R engine tests
        run: pytest tests/test_analysis_engines.py -k "r"
```

---

## Implementation Milestones

### Milestone 1: Abstraction Layer ✅

- [x] Create `src/analysis/base.py` with Protocol and EstimationResult
- [x] Create `src/analysis/factory.py` with registry
- [x] Create `src/analysis/specifications.py` for YAML loading
- [x] Create `specifications.yml` template
- [x] Add config options to `src/config.py`
- [x] Add `engines list` and `engines check` CLI commands

### Milestone 2: Python Engine Refactor ✅

- [x] Extract estimation logic from `s03_estimation.py`
- [x] Create `src/analysis/engines/python_engine.py`
- [x] Update `s03_estimation.py` to use factory
- [x] Add `--engine` flag to CLI
- [x] Migrate existing tests

### Milestone 3: R Engine ✅

- [x] Create `src/analysis/engines/r/estimate.R`
- [x] Create `src/analysis/engines/r_engine.py`
- [x] Add R engine tests
- [x] Test coefficient consistency with Python

### Milestone 4: Stata Engine (Deferred)

- [ ] Create `src/analysis/engines/stata/estimate.do`
- [ ] Create `src/analysis/engines/stata_engine.py`
- [ ] Add Stata engine tests (skip in CI if no license)

### Milestone 5: Julia Engine (Planned)

- [ ] Create `src/analysis/engines/julia/estimate.jl`
- [ ] Create `src/analysis/engines/julia_engine.py`
- [ ] Add Julia engine tests

### Milestone 6: Documentation & Polish ✅

- [x] Update all existing docs
- [ ] Add CI workflows for R/Julia
- [x] Update CHANGELOG

---

## File Structure After Implementation

```
src/
├── analysis/
│   ├── __init__.py
│   ├── base.py              # Protocol + EstimationResult
│   ├── factory.py           # Engine registry
│   ├── specifications.py    # YAML loader
│   └── engines/
│       ├── __init__.py
│       ├── python_engine.py
│       ├── r_engine.py
│       ├── stata_engine.py
│       ├── julia_engine.py
│       ├── r/
│       │   └── estimate.R
│       ├── stata/
│       │   └── estimate.do
│       └── julia/
│           └── estimate.jl
├── stages/
│   └── s03_estimation.py    # Uses factory, simplified
└── config.py                # New engine settings

specifications.yml           # Language-agnostic specs (project root)

tests/
└── test_analysis_engines.py # Conformance tests
```

---

## Success Criteria

1. **Python works unchanged** - Default behavior is identical to current
2. **R produces matching results** - Coefficients within 1% of Python
3. **Engine is selectable** - Via config or `--engine` flag
4. **Clear errors** - Helpful messages when engine unavailable
5. **Easy to extend** - Adding new engine requires only `*_engine.py` + script
6. **Documented** - Each engine has setup guide

---

## Future Extensions

Once the abstraction exists, additional capabilities become straightforward:

- **Engine-specific options**: Pass through to underlying tool (e.g., fixest's `lean=TRUE`)
- **Result caching**: Cache by engine+spec hash (already have infrastructure)
- **Benchmark mode**: Compare engines on same data
- **Auto-selection**: Pick fastest available engine
- **Remote execution**: Run Stata on licensed server via SSH
