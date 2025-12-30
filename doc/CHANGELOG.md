# Changelog

All notable changes to this project are documented in this file.

---

## [2025-12-30] - Documentation Review

### Updated

- **doc/METHODOLOGY.md**: Fixed placeholder date
- **doc/README.md**: Added missing utilities (`spatial_cv.py`, `cache.py`) to source code table

---

## [2025-12-29] - Spatial Cross-Validation and ML Model Expansion

### Added

- **Spatial Cross-Validation Module** (`src/utils/spatial_cv.py`)
  - `SpatialCVManager` class with 8 grouping methods
  - Methods: kmeans, balanced_kmeans, geographic_bands, longitude_bands, spatial_blocks, zip_digit, contiguity_queen, contiguity_rook
  - `cross_validate()` with GroupKFold integration
  - `compare_to_random_cv()` for leakage quantification
  - Convenience functions: `create_spatial_groups_simple()`, `compare_spatial_vs_random_cv()`
  - Optional geopandas dependency (graceful fallback for contiguity methods)

- **ML Model Registry** (`src/stages/s03_estimation.py`)
  - `ML_MODELS` registry with factory functions
  - Models: ridge, elasticnet, random_forest, extra_trees, gradient_boosting
  - `get_model()` function for model instantiation

- **Extended Robustness Tests** (`src/stages/s04_robustness.py`)
  - `run_spatial_cv_comparison()`: Compare spatial vs random CV
  - `run_feature_ablation()`: Systematic feature subset testing
  - `run_tuned_models()`: Nested CV with hyperparameter tuning
  - `run_encoding_comparisons()`: Categorical vs ordinal encoding tests

- **Configuration Updates** (`src/config.py`)
  - Spatial CV settings: `SPATIAL_CV_N_GROUPS`, `SPATIAL_GROUPING_METHOD`
  - ML hyperparameter grids for Ridge, ElasticNet, RF, ExtraTrees, GradientBoosting
  - Repeated CV settings: `REPEATED_CV_N_SPLITS`, `REPEATED_CV_N_REPEATS`

- **Optional Spatial Dependencies** (`requirements-spatial.txt`)
  - geopandas>=0.14.0, shapely>=2.0.0, pyproj>=3.5.0

- **Comprehensive Tests** (`tests/test_spatial_cv.py`)
  - 23 tests covering grouping methods, CV, leakage quantification, edge cases

### Changed

- `requirements.txt`: Made scikit-learn a required dependency (>=1.3.0)
- `doc/PIPELINE.md`: Added Stage 04 extended tests and Spatial CV section
- `doc/METHODOLOGY.md`: Added Spatial Cross-Validation methodology section
- `README.md`: Updated features list and project structure

### Source

Features backported from ML Vision Broadband research project.

---

## [2025-12-27] - AI-Assisted Drafting and Documentation Sync

### Added

- **AI-Assisted Manuscript Drafting (Stage 09)**
  - `draft_results`: Generate results section from estimation tables
  - `draft_captions`: Generate figure captions from figures
  - `draft_abstract`: Synthesize abstract from manuscript sections
  - Multi-provider support (Anthropic Claude, OpenAI GPT-4)
  - `--dry-run` flag to preview prompts without API calls
  - New `src/llm/` package with provider abstraction

- **Cache Management Commands**
  - `cache stats`: Show cache usage statistics
  - `cache clear`: Clear cached pipeline results
  - Stage-specific cache clearing with `--stage` option

- **Documentation Updates**
  - Added Data Audit Skills section to skills.md
  - Added Cache Management Skills section to skills.md
  - Updated all stage count references from s00-s08 to s00-s09
  - Added src/llm/ package to architecture documentation

### Changed

- `src/config.py`: Added LLM configuration settings (provider, models, temperature)
- `src/config.py`: Added `DRAFTS_DIR` for AI-generated content
- Updated project structure documentation to include new modules

### Files Modified

- src/stages/s09_writing.py (new)
- src/llm/__init__.py, base.py, anthropic.py, openai.py, prompts.py (new)
- src/config.py
- src/pipeline.py
- doc/PIPELINE.md
- doc/skills.md
- doc/ARCHITECTURE.md
- doc/README.md
- README.md
- CLAUDE.md
- tests/test_llm/ (new)
- tests/test_stages/test_s09_writing.py (new)

---

## [2025-12-26] - Journal Guidelines Tooling and Docs Sync

- Added guideline fetch + heuristic parsing for journal configs
- Documented parsing limitations, outputs, and defaults
- Clarified manuscript validation checks in docs
- Added tests for journal parsing helpers and CLI args
- Added manuscript guardrails skill for agent edits

### Files Modified
- src/stages/s08_journal_parser.py
- src/pipeline.py
- tests/test_pipeline.py
- tests/test_journal_parser.py
- README.md
- doc/PIPELINE.md
- doc/README.md
- doc/skills.md
- doc/ARCHITECTURE.md
- doc/TUTORIAL.md
- doc/RISBS_PROJECT_DESCRIPTION.md

---

## [2025-12-26] - Manuscript Variants and Provenance

- Added variant snapshots with provenance manifests and comparison reports
- Updated Quarto rendering to respect profile output directories
- Added tests and documentation for variant workflows

### Files Modified
- manuscript_quarto/variant_tools.py
- manuscript_quarto/variant_new.sh
- manuscript_quarto/render_all.sh
- manuscript_quarto/index.qmd
- manuscript_quarto/appendix-*.qmd
- manuscript_quarto/variants/README.md
- manuscript_quarto/variants/INDEX.md
- manuscript_quarto/variants/index.json
- doc/MANUSCRIPT_VARIANTS.md
- doc/skills.md
- doc/PIPELINE.md
- doc/ARCHITECTURE.md
- doc/CUSTOMIZATION.md
- README.md
- tests/test_utils/test_variant_tools.py

---

## [2025-12-25] - Platform Initialized

- Created project structure from platform scaffold
- Set up pipeline stages (s00-s06)
- Configured Quarto manuscript with journal profiles
- Added documentation framework

### Files Added

- `src/pipeline.py` - Main CLI
- `src/stages/*.py` - Pipeline stages
- `manuscript_quarto/*.qmd` - Manuscript files
- `doc/*.md` - Documentation

---

## Format Guide

Each entry should include:

```markdown
## [YYYY-MM-DD] - Brief Description

- Change 1
- Change 2

### Files Modified
- file1.py
- file2.qmd

### Related
- Link to issue or PR if applicable
```

### Keywords

Use these keywords to categorize changes:

- **Data**: Changes to data processing
- **Estimation**: Changes to estimation code
- **Figures**: Changes to figure generation
- **Manuscript**: Changes to manuscript content
- **Docs**: Documentation updates
- **Config**: Configuration changes
- **Fix**: Bug fixes
