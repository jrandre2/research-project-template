# Changelog

All notable changes to this project are documented in this file.

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
