# Testing Guide

**Related**: [PIPELINE.md](PIPELINE.md) | [ARCHITECTURE.md](ARCHITECTURE.md)
**Status**: Active
**Last Updated**: 2025-12-27

---

## Overview

CENTAUR uses pytest for testing. This guide covers test structure, conventions, and how to write tests for new functionality.

---

## Quick Start

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_config.py

# Run specific test
pytest tests/test_config.py::test_project_root_exists

# Run tests matching a pattern
pytest -k "ingest"

# Run with coverage
pytest --cov=src --cov-report=html
```

---

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── fixtures/
│   └── sample_data.py       # Sample data generators
├── test_config.py           # Configuration tests
├── test_pipeline.py         # CLI command tests
├── test_data_audit.py       # Data audit tests
├── test_journal_parser.py   # Journal parser tests
├── test_multi_manuscript.py # Multi-manuscript tests
├── test_stages/
│   ├── test_s00_ingest.py   # Ingestion stage tests
│   ├── test_s01_link.py     # Linkage stage tests
│   └── ...                  # Other stage tests
├── test_utils/
│   ├── test_helpers.py      # Helper function tests
│   ├── test_validation.py   # Validation tests
│   └── test_variant_tools.py
├── test_agents/             # Agent module tests
│   ├── test_project_analyzer.py
│   └── ...
└── test_integration/        # End-to-end tests
    └── test_pipeline_e2e.py
```

---

## Writing Tests

### Naming Conventions

- Test files: `test_<module>.py`
- Test functions: `test_<what_is_being_tested>`
- Test classes: `Test<ClassName>`

```python
# Good names
def test_loads_csv_with_valid_path():
def test_raises_error_for_missing_file():
def test_validates_required_columns():

# Bad names
def test_1():
def test_csv():
def testLoadCSV():  # Use snake_case
```

### Basic Test Structure

```python
"""Tests for src/stages/s00_ingest.py."""
import pytest
import pandas as pd
from pathlib import Path

from stages.s00_ingest import load_data_files, validate_columns


class TestLoadDataFiles:
    """Tests for the load_data_files function."""

    def test_loads_single_csv(self, temp_dir, sample_csv):
        """Test loading a single CSV file."""
        result = load_data_files(temp_dir)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_raises_for_empty_directory(self, temp_dir):
        """Test that empty directories raise an error."""
        with pytest.raises(ValueError, match="No data files found"):
            load_data_files(temp_dir)
```

### Using Fixtures

Fixtures provide reusable test data. See `tests/conftest.py` for available fixtures.

```python
def test_with_sample_data(sample_df):
    """Use the sample_df fixture."""
    assert len(sample_df) == 100
    assert 'id' in sample_df.columns

def test_with_temp_directory(temp_dir):
    """Use the temp_dir fixture."""
    test_file = temp_dir / 'test.txt'
    test_file.write_text('hello')
    assert test_file.exists()

def test_with_panel_data(panel_df):
    """Use the panel_df fixture for panel data tests."""
    assert 'unit_id' in panel_df.columns
    assert 'period' in panel_df.columns
```

### Available Fixtures

| Fixture | Description |
|---------|-------------|
| `project_root` | Path to project root directory |
| `temp_dir` | Temporary directory (cleaned up after test) |
| `temp_data_dir` | Temp dir with data_work, data_raw, diagnostics |
| `sample_df` | Simple DataFrame with 100 rows |
| `panel_df` | Panel DataFrame with unit_id and period |
| `df_with_missing` | DataFrame with NaN values |
| `df_with_duplicates` | DataFrame with duplicate rows |
| `sample_config` | Sample configuration dictionary |
| `journal_config` | Sample journal configuration |
| `schema_simple` | Simple validation schema |
| `schema_with_constraints` | Schema with value constraints |
| `sample_parquet` | Path to sample .parquet file |
| `sample_csv` | Path to sample .csv file |
| `mock_project_structure` | Temp dir with mock project layout |

---

## Test Markers

Use markers to categorize tests:

```python
import pytest

@pytest.mark.slow
def test_full_pipeline_run():
    """This test takes a long time."""
    ...

@pytest.mark.integration
def test_database_connection():
    """This test requires external services."""
    ...

@pytest.mark.e2e
def test_complete_workflow():
    """End-to-end test."""
    ...
```

Run tests by marker:

```bash
# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run e2e tests
pytest -m e2e
```

---

## Testing Pipeline Stages

Each pipeline stage should have tests covering:

1. **Happy path**: Normal execution with valid input
2. **Edge cases**: Empty data, single row, boundary values
3. **Error conditions**: Invalid input, missing files
4. **Output validation**: Correct columns, data types, values

### Example Stage Test

```python
"""Tests for src/stages/s00_ingest.py."""
import pytest
import pandas as pd
from pathlib import Path

from stages.s00_ingest import (
    find_input_files,
    load_data_files,
    validate_and_clean,
    run_stage,
)


class TestFindInputFiles:
    """Tests for find_input_files function."""

    def test_finds_csv_files(self, temp_dir):
        """Test finding CSV files in directory."""
        (temp_dir / 'data1.csv').write_text('a,b\n1,2')
        (temp_dir / 'data2.csv').write_text('a,b\n3,4')

        files = find_input_files(temp_dir)
        assert len(files) == 2
        assert all(f.suffix == '.csv' for f in files)

    def test_finds_parquet_files(self, temp_dir, sample_df):
        """Test finding Parquet files in directory."""
        sample_df.to_parquet(temp_dir / 'data.parquet')

        files = find_input_files(temp_dir)
        assert len(files) == 1

    def test_returns_empty_for_no_data(self, temp_dir):
        """Test empty result for directory without data files."""
        files = find_input_files(temp_dir)
        assert len(files) == 0


class TestValidateAndClean:
    """Tests for validate_and_clean function."""

    def test_removes_duplicates(self, df_with_duplicates):
        """Test duplicate removal."""
        result = validate_and_clean(df_with_duplicates)
        assert len(result) < len(df_with_duplicates)

    def test_handles_missing_values(self, df_with_missing):
        """Test missing value handling."""
        result = validate_and_clean(df_with_missing)
        # Verify expected behavior with missing values
        assert result is not None
```

---

## Testing Utilities

Test utility functions with various inputs:

```python
"""Tests for src/utils/helpers.py."""
import pytest
from utils.helpers import format_pvalue, format_coefficient


class TestFormatPValue:
    """Tests for p-value formatting."""

    @pytest.mark.parametrize("pvalue,expected", [
        (0.001, "0.001***"),
        (0.01, "0.010**"),
        (0.05, "0.050*"),
        (0.10, "0.100"),
        (0.50, "0.500"),
    ])
    def test_pvalue_formatting(self, pvalue, expected):
        """Test p-value formatting with significance stars."""
        result = format_pvalue(pvalue)
        assert result == expected

    def test_handles_zero(self):
        """Test handling of p-value = 0."""
        result = format_pvalue(0.0)
        assert "***" in result
```

---

## Testing Agent Modules

Agent modules require testing of analysis and migration logic:

```python
"""Tests for src/agents/project_analyzer.py."""
import pytest
from pathlib import Path

from agents.project_analyzer import (
    ProjectAnalyzer,
    analyze_directory_structure,
    identify_data_files,
)


class TestProjectAnalyzer:
    """Tests for ProjectAnalyzer class."""

    def test_analyzes_directory_structure(self, mock_project_structure):
        """Test directory structure analysis."""
        analyzer = ProjectAnalyzer(mock_project_structure)
        result = analyzer.analyze()

        assert 'directories' in result
        assert 'files' in result

    def test_identifies_python_files(self, mock_project_structure):
        """Test Python file identification."""
        analyzer = ProjectAnalyzer(mock_project_structure)
        python_files = analyzer.find_python_files()

        assert len(python_files) > 0
```

---

## Running Coverage

Generate coverage reports:

```bash
# Terminal report
pytest --cov=src

# HTML report (opens in browser)
pytest --cov=src --cov-report=html
open htmlcov/index.html

# XML report (for CI)
pytest --cov=src --cov-report=xml
```

Coverage targets:
- Utilities: 90%+
- Configuration: 90%+
- Pipeline stages: 80%+
- Agent modules: 70%+

---

## Continuous Integration

Tests run automatically on push via GitHub Actions. See `.github/workflows/test.yml`.

Local pre-commit check:

```bash
# Run tests before committing
pytest -x  # Stop on first failure
```

---

## Troubleshooting Tests

### Import Errors

Ensure you're in the project root and venv is activated:

```bash
cd /path/to/project
source .venv/bin/activate
pytest
```

### Fixture Not Found

Check that:
1. Fixture is defined in `conftest.py`
2. Fixture name matches exactly
3. No circular imports

### Slow Tests

Mark slow tests and skip them during development:

```python
@pytest.mark.slow
def test_expensive_operation():
    ...
```

```bash
pytest -m "not slow"
```

---

## Adding Tests for New Features

1. **Create test file** in appropriate directory
2. **Add fixtures** to conftest.py if needed
3. **Write tests** covering happy path, edge cases, errors
4. **Run tests** to verify they pass
5. **Check coverage** to ensure adequate coverage

Template for new test file:

```python
"""Tests for src/<module>.py."""
import pytest

from <module> import function_to_test


class TestFunctionToTest:
    """Tests for function_to_test."""

    def test_happy_path(self):
        """Test normal execution."""
        result = function_to_test(valid_input)
        assert result == expected_output

    def test_edge_case(self):
        """Test edge case behavior."""
        result = function_to_test(edge_input)
        assert result is not None

    def test_error_condition(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```
