# Troubleshooting Guide

**Related**: [TESTING.md](TESTING.md) | [PIPELINE.md](PIPELINE.md)
**Status**: Active
**Last Updated**: 2025-12-27

---

## Quick Diagnostics

```bash
# Check virtual environment
source .venv/bin/activate
which python  # Should point to .venv

# Run tests
pytest -v

# Check pipeline status
python src/pipeline.py list_stages
```

---

## Common Errors

### Import Errors

#### ModuleNotFoundError: No module named 'src'

**Cause**: Virtual environment not activated or running from wrong directory.

**Solution**:
```bash
cd /path/to/project
source .venv/bin/activate
python src/pipeline.py <command>
```

#### ImportError: cannot import name 'X' from 'stages.sXX'

**Cause**: Stage module has been modified but Python cache is stale.

**Solution**:
```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
# Or
python -B src/pipeline.py <command>  # Run without bytecode
```

---

### Data Errors

#### FileNotFoundError: Input file not found

**Cause**: Previous pipeline stage hasn't been run.

**Solution**:
```bash
# Run stages in order
python src/pipeline.py ingest_data --demo  # Or with real data
python src/pipeline.py link_records
python src/pipeline.py build_panel
python src/pipeline.py run_estimation
```

#### ValueError: No data files found in data_raw/

**Cause**: No raw data files present and `--demo` flag not used.

**Solution**:
```bash
# Use synthetic demo data
python src/pipeline.py ingest_data --demo

# Or add your data files to data_raw/
ls data_raw/  # Check for .csv, .parquet, .xlsx files
```

#### KeyError: 'column_name'

**Cause**: Expected column missing from data.

**Solution**:
1. Check column names in your data:
   ```python
   import pandas as pd
   df = pd.read_parquet('data_work/panel.parquet')
   print(df.columns.tolist())
   ```
2. Update stage configuration (e.g., `REQUIRED_COLUMNS` in `s00_ingest.py`)

---

### Validation Errors

#### Validation failed: Duplicate rows found

**Cause**: Data has duplicate unit-period combinations.

**Solution**:
```python
# Find duplicates
import pandas as pd
df = pd.read_parquet('data_work/data_linked.parquet')
dups = df[df.duplicated(['id', 'period'], keep=False)]
print(dups)
```

#### Validation failed: Missing values in required columns

**Cause**: Required columns contain NaN values.

**Solution**:
```python
# Check for missing values
import pandas as pd
df = pd.read_parquet('data_work/data_raw.parquet')
print(df.isnull().sum())
```

---

### Estimation Errors

#### LinAlgError: Singular matrix

**Cause**: Perfect multicollinearity in regression variables.

**Solution**:
1. Check for redundant variables
2. Check fixed effects - may be collinear with treatment
3. Use a different specification

#### ValueError: Insufficient observations

**Cause**: Too few observations after dropping missing values.

**Solution**:
1. Check data for excessive missing values
2. Reduce number of control variables
3. Use a less restrictive sample

---

### Manuscript Errors

#### Quarto render fails

**Cause**: Various Quarto or R/Python environment issues.

**Solution**:
```bash
# Check Quarto installation
quarto check

# Render with verbose output
cd manuscript_quarto
quarto render index.qmd --verbose

# Check for missing dependencies
quarto install tinytex  # For PDF output
```

#### Missing figures in manuscript

**Cause**: Figures not generated or path incorrect.

**Solution**:
```bash
# Regenerate figures
python src/pipeline.py make_figures

# Check figure directory
ls manuscript_quarto/figures/
```

---

### Test Errors

#### Tests fail with import errors

**Cause**: src not in Python path.

**Solution**:
```bash
# Run from project root with activated venv
cd /path/to/project
source .venv/bin/activate
pytest tests/
```

#### Fixtures not found

**Cause**: conftest.py not being loaded.

**Solution**:
- Ensure `tests/conftest.py` exists
- Check for syntax errors in conftest.py
- Run pytest from project root

---

### Performance Issues

#### Pipeline stage is very slow

**Cause**: Large dataset or inefficient operations.

**Solution**:
1. Use Parquet format (faster than CSV)
2. Add progress indicators to long loops
3. Consider sampling for development:
   ```python
   df = df.sample(n=10000, random_state=42)
   ```

#### Memory errors

**Cause**: Data too large for available RAM.

**Solution**:
1. Process in chunks:
   ```python
   for chunk in pd.read_csv('large_file.csv', chunksize=100000):
       process(chunk)
   ```
2. Use more memory-efficient dtypes
3. Clear unused variables with `del df`

---

## Environment Issues

### Virtual environment problems

```bash
# Recreate virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Conflicting package versions

```bash
# Check installed packages
pip list

# Install specific versions
pip install pandas==2.0.0

# Update requirements
pip freeze > requirements.txt
```

---

## Getting Help

1. **Check documentation**: `doc/README.md` for the full index
2. **Run tests**: `pytest -v` to check if environment is working
3. **Enable verbose output**: Add `-v` or `--verbose` to commands
4. **Check logs**: Look for error messages in terminal output
5. **Report issues**: https://github.com/anthropics/claude-code/issues

---

## Debugging Tips

### Enable verbose mode

```bash
python src/pipeline.py <command> --verbose
```

### Add debug prints

```python
# Temporary debug output
print(f"DEBUG: df shape = {df.shape}")
print(f"DEBUG: columns = {df.columns.tolist()}")
import pdb; pdb.set_trace()  # Interactive debugger
```

### Check intermediate files

```bash
# List data_work contents
ls -la data_work/

# Inspect parquet file
python -c "import pandas as pd; print(pd.read_parquet('data_work/panel.parquet').head())"
```
