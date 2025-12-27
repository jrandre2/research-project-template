# Agent Tools Reference

AI-powered project analysis and migration tools.

**Related**: [skills.md](skills.md) | [PIPELINE.md](PIPELINE.md)

---

## Overview

The `src/agents/` package provides tools for:

- Analyzing existing research project structures
- Mapping projects to the standardized platform structure
- Generating migration plans
- Executing automated migrations

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Analyze a project
python src/pipeline.py analyze_project --path /path/to/project

# Map to platform structure
python src/pipeline.py map_project --path /path/to/project

# Generate migration plan
python src/pipeline.py plan_migration --path /source --target /target

# Execute migration (always test with dry-run first)
python src/pipeline.py migrate_project --path /source --target /target --dry-run
python src/pipeline.py migrate_project --path /source --target /target
```

---

## Module Reference

### ProjectAnalyzer

**File:** `src/agents/project_analyzer.py`

Scans project directories and extracts structural metadata.

**Classes:**

| Class | Purpose |
|-------|---------|
| `FileInfo` | File metadata (path, size, extension) |
| `ModuleInfo` | Python module metadata (docstring, imports, functions, classes) |
| `DirectoryInfo` | Directory statistics |
| `ProjectAnalysis` | Complete project analysis result |

**Usage:**

```python
from pathlib import Path
from agents.project_analyzer import ProjectAnalyzer

analyzer = ProjectAnalyzer(Path("/path/to/project"))
analysis = analyzer.analyze()

# Get summary
print(analysis.summary())

# Export to JSON
json_output = analysis.to_json()
```

**Pattern Detection:**

- Pipeline stages (numbered prefixes like `s00_`, `s01_`)
- Test directories (`tests/`, `test/`)
- Documentation files (`docs/`, `doc/`, `*.md`)
- Jupyter notebooks (`*.ipynb`)
- Manuscript/Quarto files (`*.qmd`, `manuscript/`)
- Data directories (`data/`, `data_raw/`)

---

### StructureMapper

**File:** `src/agents/structure_mapper.py`

Maps analyzed projects to platform stages based on content analysis.

**Classes:**

| Class | Purpose |
|-------|---------|
| `MappingRule` | Single mapping rule (source → target) |
| `StructureMapping` | Complete mapping result with rules and warnings |

**Usage:**

```python
from agents.project_analyzer import ProjectAnalyzer
from agents.structure_mapper import StructureMapper

analyzer = ProjectAnalyzer(Path("/path/to/project"))
analysis = analyzer.analyze()

mapper = StructureMapper(analysis)
mapping = mapper.generate_mapping()

print(mapping.summary())
```

**Mapping Logic:**

| Source Content | Target Location |
|----------------|-----------------|
| `data/` | `data_raw/` |
| `output/`, `outputs/`, `figures/` | `manuscript_quarto/figures/` |
| `docs/` | `doc/` |
| `tests/` | `tests/` |

Note: The mapper only copies `docs/` and `tests/`. If your source uses `doc/` or `test/`, rename or copy manually. Directories like `results/` are not mapped by default.

**Python Module Mapping (by keywords):**

| Keywords | Target Stage |
|----------|--------------|
| `data`, `loader`, `ingest`, `load`, `import` | `src/stages/s00_ingest.py` |
| `link`, `merge`, `join`, `match` | `src/stages/s01_link.py` |
| `panel`, `construct`, `build`, `prepare` | `src/stages/s02_panel.py` |
| `model`, `estim`, `regress`, `fit`, `sem` | `src/stages/s03_estimation.py` |
| `robust`, `sensitiv`, `check`, `valid` | `src/stages/s04_robustness.py` |
| `visual`, `plot`, `figure`, `graph`, `chart` | `src/stages/s05_figures.py` |
| `manuscript`, `report`, `output` | `src/stages/s06_manuscript.py` |

Modules with `util` or `helper` in the path are copied to `src/utils/`.

---

### MigrationPlanner

**File:** `src/agents/migration_planner.py`

Generates actionable migration plans.

**Classes:**

| Class | Purpose |
|-------|---------|
| `MigrationStep` | Single migration step with category, action, source, target |
| `MigrationPlan` | Complete plan with steps, complexity estimate, notes |

**Usage:**

```python
from agents.project_analyzer import ProjectAnalyzer
from agents.structure_mapper import StructureMapper
from agents.migration_planner import MigrationPlanner

analyzer = ProjectAnalyzer(Path("/path/to/project"))
analysis = analyzer.analyze()

mapper = StructureMapper(analysis)
mapping = mapper.generate_mapping()

planner = MigrationPlanner(analysis, mapping)
plan = planner.generate_plan("/path/to/target")

# Output as markdown
print(plan.to_markdown())

# Output as JSON
print(plan.to_json())
```

**Step Categories:**

| Category | Purpose |
|----------|---------|
| `setup` | Create directories, git init, venv |
| `copy` | Transfer files to standard locations |
| `transform` | Merge modules into stage files |
| `generate` | Create documentation templates |
| `verify` | Validate imports, run tests |

**Complexity Estimation:**

- **Low**: < 10 modules, no warnings, no notebooks
- **Medium**: 10-20 modules, 2-5 warnings, or notebooks present
- **High**: > 20 modules, > 5 warnings

---

### MigrationExecutor

**File:** `src/agents/migration_executor.py`

Executes migration plans.

**Classes:**

| Class | Purpose |
|-------|---------|
| `ExecutionResult` | Single step result (success, message, error, duration) |
| `ExecutionReport` | Complete execution report with all results |
| `MigrationExecutor` | Execution engine |

**Usage:**

```python
from agents.migration_executor import MigrationExecutor

executor = MigrationExecutor(
    plan=plan,
    source_path=Path("/source"),
    template_path=Path("/template"),
    dry_run=True,  # Always test first!
    verbose=True
)

report = executor.execute()

# Check results
print(f"Success: {report.success_count}/{len(report.results)}")
print(report.to_markdown())
```

**Features:**

- **Dry-run mode**: Preview changes without executing
- **Step-by-step progress**: Verbose output shows each step
- **Scaffold generation**: Creates merge instructions for code
- **Documentation templates**: Generates DATA_DICTIONARY.md, etc.
- **Verification**: Checks imports, runs tests, validates docs

---

## Extension Points

### Adding Custom Mapping Rules

Edit `src/agents/structure_mapper.py`:

```python
# In StructureMapper._generate_python_mappings()
STAGE_KEYWORDS = {
    's00_ingest': ['load', 'ingest', 'read', 'your_keyword'],
    's03_estimation': ['model', 'estim', 'regress', 'your_keyword'],
    # Add new stage mappings
}
```

### Adding Custom Step Categories

Edit `src/agents/migration_planner.py`:

```python
def _add_custom_steps(self, plan: MigrationPlan, order: int) -> int:
    """Add custom migration steps."""
    order += 1
    plan.steps.append(MigrationStep(
        order=order,
        category='custom',
        action='Your custom action',
        details='Details here',
    ))
    return order
```

### Adding Custom Execution Handlers

Edit `src/agents/migration_executor.py`:

```python
def _execute_step(self, step: MigrationStep) -> ExecutionResult:
    # ... existing code ...
    elif step.category == 'custom':
        result = self._execute_custom(step)
    # ...

def _execute_custom(self, step: MigrationStep) -> ExecutionResult:
    """Execute custom step type."""
    # Your custom execution logic
    return ExecutionResult(
        step=step,
        success=True,
        message="Custom step completed"
    )
```

### Adding Custom Verification

Add new verification in `MigrationExecutor`:

```python
def _verify_custom(self) -> ExecutionResult:
    """Custom verification logic."""
    # Check something specific to your project
    if some_condition:
        return ExecutionResult(
            step=MigrationStep(0, 'verify', 'Custom check'),
            success=True,
            message="Custom verification passed"
        )
    else:
        return ExecutionResult(
            step=MigrationStep(0, 'verify', 'Custom check'),
            success=False,
            error="Custom verification failed: reason"
        )
```

---

## Best Practices

1. **Always run dry-run first**
   ```bash
   python src/pipeline.py migrate_project --path /source --target /target --dry-run
   ```

2. **Review scaffold files** - Generated stage files contain merge instructions, not actual code

3. **Check warnings** - Migration plans include warnings for edge cases (notebooks, unmapped modules)

4. **Run verification** - Use verification steps to validate migration completeness

5. **Keep source intact** - Migration copies files, never modifies the source project

6. **Complete manually** - After migration, manually merge code into scaffold files

---

## Troubleshooting

### "No modules found"

The project may not have Python files in expected locations. Check:
- Are Python files in subdirectories?
- Are there `__init__.py` files?

### "Unmapped modules" warning

Modules without recognized keywords go to unmapped. Options:
- Add keywords to `structure_mapper.py`
- Manually copy after migration

### Verification failures

Expected for scaffold files. After migration:
1. Complete scaffold files with actual code
2. Create missing documentation
3. Re-run verification manually

### Import errors after migration

Check:
- `__init__.py` files in `src/stages/`
- PYTHONPATH includes project root
- Virtual environment activated
