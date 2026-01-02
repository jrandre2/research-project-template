# Library Packaging Analysis for CENTAUR

## Executive Summary

CENTAUR contains several well-architected components with strong library extraction potential. The project's Protocol-based design and clean separation of concerns make certain modules immediately suitable for standalone packaging.

**Target:** General data scientists
**Approach:** Single monolithic Python library + companion R package
**Packages to create:** `centaur-tools` (Python) + `centaurR` (R)

---

## Components Suitable for Library Extraction

### Tier 1: High Priority (Production-Ready)

| Component | Location | LOC | Coupling | Library Name Suggestion |
|-----------|----------|-----|----------|------------------------|
| Analysis Engine | `src/analysis/` | ~1,043 | Zero | `estimation-engine` |
| Spatial Utils | `src/spatial/` | ~857 | Zero | `geospatial-research` |
| LLM Provider | `src/llm/` | ~839 | Minimal | `multi-llm-provider` |

### Tier 2: Medium Priority (Needs Refactoring)

| Component | Location | LOC | Issue |
|-----------|----------|-----|-------|
| Cache Manager | `src/utils/cache.py` | ~668 | Config dependency |
| Validation Framework | `src/utils/validation.py` | ~729 | Minor coupling |
| QA Metrics | `src/stages/_qa_utils.py` | ~200 | Stage coupling |

### Tier 3: Domain-Specific (Not Recommended)

- Pipeline stages (s00-s09) - research-specific
- Review management - academic workflow specific
- GUI dashboard - tightly coupled to pipeline

---

## Detailed Analysis

### 1. Analysis Engine Module (HIGHEST VALUE)

**What it provides:**

- `AnalysisEngine` Protocol - language-agnostic interface for estimation
- `EstimationResult` dataclass - standardized results across engines
- Factory pattern with `@register_engine` decorator
- YAML-based specification management
- Python (NumPy) and R (fixest) implementations

**Why it's valuable:**

- Solves real problem: running same analysis across languages
- Protocol-based design allows new engines (Stata, Julia) without modification
- Specification format decouples "what to estimate" from "how"

**Benefits of packaging:**

- Reproducibility: Same specs work in Python and R
- Collaboration: Teams using different tools can share code
- Validation: Cross-engine comparison catches implementation bugs
- Teaching: Students can learn one spec format, multiple languages

### 2. Spatial Module

**What it provides:**

- Spatial I/O (GeoPackage, Shapefile, GeoJSON)
- Distance calculations (Haversine, matrices, nearest neighbor)
- CRS handling (UTM estimation, reprojection)

**Benefits of packaging:**

- Fills gap: simpler than geopandas for common research tasks
- Optional dependencies handled gracefully
- Well-documented with examples

### 3. LLM Provider Module

**What it provides:**

- Unified interface for Anthropic Claude and OpenAI GPT
- Factory pattern for provider selection
- Academic writing prompt templates

**Benefits of packaging:**

- Provider-agnostic code
- Easy to add new providers
- Reduces boilerplate in research projects

---

## Recommended Package Structure

### Python: `centaur-tools`

Single monolithic package with optional dependencies:

```text
centaur-tools/
├── pyproject.toml
├── src/centaur/
│   ├── __init__.py           # Public API exports
│   ├── estimation/           # Multi-language estimation engine
│   │   ├── engine.py         # AnalysisEngine protocol
│   │   ├── result.py         # EstimationResult dataclass
│   │   ├── specs.py          # YAML specification handling
│   │   ├── factory.py        # Engine registry
│   │   └── engines/
│   │       ├── python.py     # NumPy implementation
│   │       └── r.py          # R/fixest bridge
│   ├── spatial/              # Geospatial utilities
│   │   ├── io.py             # Load/save spatial data
│   │   ├── distance.py       # Haversine, matrices
│   │   └── crs.py            # CRS handling
│   ├── llm/                  # LLM provider abstraction
│   │   ├── provider.py       # LLMProvider protocol
│   │   ├── anthropic.py      # Claude implementation
│   │   └── openai.py         # GPT implementation
│   ├── cache/                # Hash-based caching
│   │   └── manager.py        # CacheManager
│   └── validation/           # Data validation framework
│       └── rules.py          # Rule-based validation
├── tests/
└── docs/
```

**Installation options:**

```bash
pip install centaur-tools                    # Core only
pip install centaur-tools[spatial]           # + geopandas, shapely
pip install centaur-tools[llm]               # + anthropic, openai SDKs
pip install centaur-tools[r-engine]          # + rpy2 for R integration
pip install centaur-tools[all]               # Everything
```

### R: `centaurR`

Companion R package for cross-language workflows:

```text
centaurR/
├── DESCRIPTION
├── NAMESPACE
├── R/
│   ├── specifications.R      # Load/validate YAML specs
│   ├── estimation.R          # fixest wrapper with standardized output
│   ├── result.R              # EstimationResult S3 class
│   ├── export.R              # Export to Python-compatible format
│   └── validate.R            # Cross-engine result comparison
├── inst/
│   └── python/               # Python interop utilities
├── man/
├── tests/
└── vignettes/
```

**Key R features:**

- Same YAML specification format as Python
- `EstimationResult` S3 class matching Python dataclass
- JSON/Parquet export for Python interop
- `compare_engines()` function for validation

---

## Benefits for General Data Scientists

### Multi-Language Estimation Engine

- **Problem solved:** Run same analysis in Python and R, compare results
- **Use case:** Validate implementations, collaborate across language preferences
- **Unique value:** Standardized `EstimationResult` format enables meta-analysis

### Spatial Utilities

- **Problem solved:** Common geospatial operations without full GIS complexity
- **Use case:** Distance calculations, CRS handling, spatial I/O
- **Unique value:** Simpler API than raw geopandas for research workflows

### LLM Provider Abstraction

- **Problem solved:** Switch between Claude/GPT without code changes
- **Use case:** Provider-agnostic AI integration
- **Unique value:** Unified interface, easy to add new providers

### Hash-Based Caching

- **Problem solved:** Intelligent caching with automatic invalidation
- **Use case:** Long-running computations, reproducible pipelines
- **Unique value:** Dependency tracking - cache invalidates when inputs change

---

## Implementation Approach

### Phase 1: Python Package Setup

1. Create new `centaur-tools/` repository
2. Set up `pyproject.toml` with modern Python packaging (hatchling or setuptools)
3. Define optional dependency groups: `[spatial]`, `[llm]`, `[r-engine]`, `[all]`
4. Extract modules from CENTAUR with minimal changes:
   - `src/analysis/` → `centaur/estimation/`
   - `src/spatial/` → `centaur/spatial/`
   - `src/llm/` → `centaur/llm/`
   - `src/utils/cache.py` → `centaur/cache/`
5. Remove CENTAUR-specific imports (config.py references)
6. Create clean public API in `__init__.py`

### Phase 2: Testing & Documentation

1. Port existing tests from CENTAUR
2. Add cross-engine validation tests
3. Set up CI/CD (GitHub Actions)
4. Create documentation with mkdocs-material
5. Write usage examples for each module

### Phase 3: R Package Development

1. Create `centaurR/` with standard R package structure
2. Implement `EstimationResult` as S3 class
3. Create `fixest` wrapper matching Python interface
4. Add YAML specification loader
5. Write vignettes showing Python-R interop
6. Submit to CRAN

### Phase 4: Distribution

1. Publish Python package to PyPI
2. Publish R package to CRAN
3. Create migration guide for CENTAUR users
4. Set up versioning strategy (keep in sync or independent)

---

## Key Files to Extract from CENTAUR

| Source | Destination | Changes Needed |
|--------|-------------|----------------|
| `src/analysis/base.py` | `centaur/estimation/engine.py` | None |
| `src/analysis/factory.py` | `centaur/estimation/factory.py` | None |
| `src/analysis/specifications.py` | `centaur/estimation/specs.py` | None |
| `src/analysis/engines/*.py` | `centaur/estimation/engines/` | None |
| `src/spatial/core/*.py` | `centaur/spatial/` | None |
| `src/llm/*.py` | `centaur/llm/` | Remove config import |
| `src/utils/cache.py` | `centaur/cache/manager.py` | Remove config import |
| `src/utils/validation.py` | `centaur/validation/rules.py` | Minor cleanup |

---

## Recommended Next Steps

1. **Create repository:** Set up `centaur-tools` repo with package structure
2. **Extract estimation module first:** Highest value, zero coupling
3. **Add spatial and LLM modules:** Both ready for extraction
4. **Develop R package in parallel:** Share specification format
5. **Publish alpha to TestPyPI:** Get early feedback
