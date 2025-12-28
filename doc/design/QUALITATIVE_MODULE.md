# Qualitative & Mixed Methods Module Design Plan

**Feature**: Add qualitative analysis capabilities to CENTAUR for mixed-methods research
**Status**: Design Document
**Created**: 2025-12-27

---

## Executive Summary

Extend CENTAUR to support qualitative and mixed-methods research by adding a parallel qualitative analysis track (stages s10-s14) that integrates with the existing quantitative pipeline at a synthesis stage (s15). The design prioritizes interoperability with existing qualitative tools (NVivo, Atlas.ti) while enabling AI-assisted analysis for researchers who prefer an integrated workflow.

---

## Architecture Overview

```text
QUANTITATIVE TRACK                 QUALITATIVE TRACK
══════════════════                 ══════════════════
s00_ingest ─────────┐              s10_qual_ingest ────────┐
s01_link            │              s11_coding              │
s02_panel           │              s12_themes              │
s03_estimation      │              s13_qual_synthesis      │
s04_robustness      │              s14_qual_figures        │
s05_figures ────────┤                     │                │
                    │                     │                │
                    └─────────┬───────────┘                │
                              ▼                            │
                       s15_integration ◄───────────────────┘
                              │
                              ▼
                    s06_manuscript (existing)
                    s07_reviews (existing)
                    s09_writing (existing, enhanced)
```

---

## New Stages

### Stage 10: Qualitative Data Ingestion (`s10_qual_ingest.py`)

**Purpose**: Import qualitative data from multiple sources and formats.

**Input Sources**:

- Interview transcripts (TXT, DOCX, PDF, RTF)
- Field notes (TXT, Markdown)
- NVivo exports (NVPX project files, Excel coded exports)
- Atlas.ti exports (ATLAS XML, Excel)
- MAXQDA exports (MEX, Excel)
- Audio/video metadata (timestamps, durations)
- Document collections (PDF folders)

**Output Files**:

- `data_work/qual/corpus.parquet` — Unified document corpus
- `data_work/qual/corpus_metadata.parquet` — Document metadata (source, date, participant, etc.)
- `data_work/qual/segments.parquet` — Text segments with positions
- `data_work/diagnostics/qual_ingest_summary.csv`

**Key Features**:

- Text extraction and cleaning (encoding normalization, whitespace)
- Paragraph/sentence segmentation with position tracking
- Participant/document ID assignment
- Deduplication detection
- Import existing codes from NVivo/Atlas.ti exports

**CLI Commands**:

```bash
python src/pipeline.py qual_ingest --source transcripts/
python src/pipeline.py qual_ingest --source project.nvpx --format nvivo
python src/pipeline.py qual_ingest --source export.xlsx --format atlas
```

---

### Stage 11: Qualitative Coding (`s11_coding.py`)

**Purpose**: Apply codes to text segments, supporting both deductive and inductive approaches.

**Coding Modes**:

1. **Deductive** — Apply predefined codebook to corpus
2. **Inductive** — AI-assisted code suggestion with human review
3. **Import** — Load codes from external tool exports
4. **Hybrid** — Combine predefined codes with emergent discovery

**Input Files**:

- `data_work/qual/corpus.parquet`
- `data_work/qual/segments.parquet`
- `data_work/qual/codebook.yaml` (optional, for deductive)
- External tool exports (for import mode)

**Output Files**:

- `data_work/qual/coded_segments.parquet` — Segments with applied codes
- `data_work/qual/codebook.yaml` — Codebook (created or updated)
- `data_work/qual/coding_log.parquet` — Audit trail of coding decisions
- `data_work/qual/code_frequencies.csv` — Code occurrence counts
- `data_work/diagnostics/coding_summary.csv`

**Codebook Schema** (`codebook.yaml`):

```yaml
codebook:
  version: 1
  created: 2025-12-27
  codes:
    - id: BARRIER_FINANCIAL
      label: Financial Barriers
      definition: References to cost, funding, or economic constraints
      parent: BARRIERS
      examples:
        - "We couldn't afford the equipment"
        - "Budget cuts forced us to cancel"
    - id: BARRIER_TECHNICAL
      label: Technical Barriers
      definition: References to technology limitations or failures
      parent: BARRIERS
```

**AI-Assisted Features**:

- Suggest codes for uncoded segments
- Identify potential new codes from patterns
- Flag inconsistent coding for review
- Generate code definitions from examples

**CLI Commands**:

```bash
python src/pipeline.py qual_code --mode deductive --codebook codebook.yaml
python src/pipeline.py qual_code --mode inductive --suggest-codes
python src/pipeline.py qual_code --mode import --source nvivo_export.xlsx
python src/pipeline.py qual_code --review  # Interactive review mode
python src/pipeline.py qual_code --dry-run --provider anthropic
```

---

### Stage 12: Thematic Analysis (`s12_themes.py`)

**Purpose**: Develop themes from coded data, supporting multiple analytic approaches.

**Supported Approaches**:

- **Thematic Analysis** (Braun & Clarke) — Pattern-based theme development
- **Framework Analysis** — Matrix-based charting for applied research
- **Grounded Theory** — Category development with theoretical sampling notes
- **Content Analysis** — Frequency and co-occurrence analysis

**Input Files**:

- `data_work/qual/coded_segments.parquet`
- `data_work/qual/codebook.yaml`
- `data_work/qual/corpus_metadata.parquet`

**Output Files**:

- `data_work/qual/themes.yaml` — Theme definitions and relationships
- `data_work/qual/theme_segments.parquet` — Segments organized by theme
- `data_work/qual/code_cooccurrence.parquet` — Code co-occurrence matrix
- `data_work/qual/framework_matrix.parquet` — Cases × themes matrix (if framework)
- `data_work/qual/memos.parquet` — Analytic memos and notes
- `data_work/diagnostics/theme_summary.csv`

**Theme Schema** (`themes.yaml`):

```yaml
themes:
  version: 1
  approach: thematic_analysis
  themes:
    - id: THEME_SYSTEMIC_BARRIERS
      label: Systemic Barriers to Adoption
      definition: Organizational and structural factors that impede implementation
      codes: [BARRIER_FINANCIAL, BARRIER_POLICY, BARRIER_INSTITUTIONAL]
      subthemes:
        - id: SUBTHEME_RESOURCE_CONSTRAINTS
          label: Resource Constraints
          codes: [BARRIER_FINANCIAL, BARRIER_STAFFING]
      exemplar_quotes:
        - segment_id: seg_0142
          text: "The policy was clear, but we had no budget to implement it"
      analytic_notes: >
        This theme captures structural impediments beyond individual control...
```

**AI-Assisted Features**:

- Suggest theme groupings from code clusters
- Generate theme definitions from constituent codes
- Identify potential subtheme structures
- Draft analytic memos from patterns
- Suggest exemplar quotes for each theme

**CLI Commands**:

```bash
python src/pipeline.py qual_themes --approach thematic
python src/pipeline.py qual_themes --approach framework --cases participant_id
python src/pipeline.py qual_themes --suggest --dry-run
python src/pipeline.py qual_themes --export-matrix themes_matrix.xlsx
```

---

### Stage 13: Qualitative Synthesis (`s13_qual_synthesis.py`)

**Purpose**: Generate synthesis outputs for manuscript integration.

**Output Types**:

- **Quote Extracts** — Organized quotes by theme for results section
- **Thematic Summaries** — Prose summaries of each theme
- **Participant Profiles** — Case summaries for case study designs
- **Saturation Analysis** — Evidence of thematic saturation
- **Codebook Report** — Formatted codebook for appendix

**Input Files**:

- `data_work/qual/themes.yaml`
- `data_work/qual/theme_segments.parquet`
- `data_work/qual/corpus_metadata.parquet`
- `data_work/qual/memos.parquet`

**Output Files**:

- `data_work/qual/quote_bank.parquet` — Quotes organized for manuscript use
- `data_work/qual/theme_summaries.md` — Prose summaries (AI-generated drafts)
- `data_work/qual/participant_profiles.md` — Case summaries
- `data_work/qual/saturation_analysis.csv` — New codes/themes by document
- `data_work/qual/codebook_report.md` — Formatted for appendix
- `manuscript_quarto/data/qual_quotes.yaml` — For manuscript integration

**AI-Assisted Features**:

- Draft theme summaries from coded data
- Generate participant vignettes
- Suggest representative vs. deviant cases
- Synthesize cross-cutting patterns

**CLI Commands**:

```bash
python src/pipeline.py qual_synthesize --output quotes
python src/pipeline.py qual_synthesize --output summaries --dry-run
python src/pipeline.py qual_synthesize --output profiles --cases 5
python src/pipeline.py qual_synthesize --saturation-report
```

---

### Stage 14: Qualitative Figures (`s14_qual_figures.py`)

**Purpose**: Generate publication-ready qualitative visualizations.

**Figure Types**:

- **Code Frequency Charts** — Bar/pie charts of code distributions
- **Co-occurrence Networks** — Node-link diagrams of code relationships
- **Theme Hierarchy Diagrams** — Tree/sunburst of theme structure
- **Framework Matrices** — Heatmaps of cases × themes
- **Quote Displays** — Formatted quote boxes for manuscripts
- **Timeline Diagrams** — Temporal patterns in longitudinal data
- **Concept Maps** — Visual theory diagrams

**Input Files**:

- `data_work/qual/themes.yaml`
- `data_work/qual/code_cooccurrence.parquet`
- `data_work/qual/framework_matrix.parquet`
- `data_work/qual/quote_bank.parquet`

**Output Files**:

- `figures/qual_code_frequency.png`
- `figures/qual_cooccurrence_network.png`
- `figures/qual_theme_hierarchy.png`
- `figures/qual_framework_matrix.png`
- `figures/qual_concept_map.png`
- `data_work/diagnostics/qual_figures_summary.csv`

**CLI Commands**:

```bash
python src/pipeline.py qual_figures --all
python src/pipeline.py qual_figures --type network --min-cooccurrence 3
python src/pipeline.py qual_figures --type matrix --export-data
python src/pipeline.py qual_figures --type quotes --theme THEME_SYSTEMIC_BARRIERS
```

---

### Stage 15: Mixed Methods Integration (`s15_integration.py`)

**Purpose**: Synthesize quantitative and qualitative findings.

**Integration Strategies** (Creswell & Plano Clark):

- **Convergent** — Compare/contrast quant and qual findings
- **Explanatory Sequential** — Use qual to explain quant results
- **Exploratory Sequential** — Use qual to develop quant measures
- **Embedded** — Nest one method within another

**Input Files**:

- Quantitative: `data_work/estimates.parquet`, `data_work/robustness.parquet`
- Qualitative: `data_work/qual/themes.yaml`, `data_work/qual/quote_bank.parquet`
- Both: Manuscript outline/structure

**Output Files**:

- `data_work/mixed/joint_display.parquet` — Side-by-side quant/qual findings
- `data_work/mixed/integration_matrix.xlsx` — Integration visualization
- `data_work/mixed/convergence_report.md` — Agreement/divergence analysis
- `manuscript_quarto/data/integration.yaml` — For manuscript integration
- `figures/mixed_joint_display.png` — Visual joint display

**Joint Display Schema**:

```yaml
joint_display:
  - finding_id: 1
    quantitative:
      result: "Treatment effect = 0.15 (p < 0.01)"
      source: estimates.parquet
      specification: baseline
    qualitative:
      theme: THEME_SYSTEMIC_BARRIERS
      summary: "Participants described financial constraints as primary barrier"
      exemplar_quote: "We couldn't afford the required equipment"
      quote_source: participant_07
    integration:
      convergence: partial
      interpretation: >
        Quantitative results show modest effect; qualitative data suggests
        barriers limit full implementation, explaining attenuated impact.
```

**AI-Assisted Features**:

- Identify potential quant/qual linkages
- Draft integration interpretations
- Suggest areas of convergence/divergence
- Generate joint display summaries

**CLI Commands**:

```bash
python src/pipeline.py integrate --strategy convergent
python src/pipeline.py integrate --strategy explanatory --quant-finding "treatment_effect"
python src/pipeline.py integrate --joint-display --format xlsx
python src/pipeline.py integrate --convergence-report
```

---

## Configuration Additions

### New Sections in `src/config.py`

```python
# ============================================================
# QUALITATIVE ANALYSIS SETTINGS
# ============================================================

# Stage enablement
QUAL_ENABLED = True
QUAL_STAGES = ['s10_qual_ingest', 's11_coding', 's12_themes',
               's13_qual_synthesis', 's14_qual_figures']

# Data directories
QUAL_DATA_DIR = DATA_WORK_DIR / 'qual'
QUAL_RAW_DIR = DATA_RAW_DIR / 'qualitative'  # transcripts, field notes

# Coding settings
QUAL_CODING_MODE = 'hybrid'  # 'deductive', 'inductive', 'import', 'hybrid'
QUAL_MIN_SEGMENT_LENGTH = 50  # Minimum characters for a codeable segment
QUAL_AI_CODING_ENABLED = True
QUAL_AI_CODING_CONFIDENCE_THRESHOLD = 0.7  # Require human review below this

# Theme analysis
QUAL_THEME_APPROACH = 'thematic_analysis'  # 'thematic_analysis', 'framework', 'grounded_theory'
QUAL_SATURATION_THRESHOLD = 3  # Documents with no new codes = saturation

# Integration settings
MIXED_METHODS_STRATEGY = 'convergent'  # 'convergent', 'explanatory', 'exploratory', 'embedded'
MIXED_INTEGRATION_ENABLED = True

# External tool imports
QUAL_IMPORT_FORMATS = ['nvivo', 'atlas', 'maxqda', 'dedoose', 'excel']
NVIVO_EXPORT_VERSION = '14'  # NVivo version for export parsing

# Figure settings
QUAL_FIGURE_DPI = 300
QUAL_FIGURE_FORMAT = 'png'
QUAL_NETWORK_MIN_EDGE_WEIGHT = 2
QUAL_NETWORK_LAYOUT = 'spring'  # 'spring', 'circular', 'hierarchical'

# ============================================================
# MIXED METHODS SETTINGS
# ============================================================

MIXED_DATA_DIR = DATA_WORK_DIR / 'mixed'
MIXED_JOINT_DISPLAY_FORMAT = 'xlsx'  # 'xlsx', 'csv', 'yaml'
```

---

## Pipeline Commands

### New Commands to Register in `pipeline.py`

```python
# Qualitative Data Commands
qual_ingest          # --source, --format, --recursive
qual_code            # --mode, --codebook, --suggest-codes, --review, --dry-run
qual_themes          # --approach, --cases, --suggest, --export-matrix
qual_synthesize      # --output, --cases, --saturation-report
qual_figures         # --all, --type, --theme, --min-cooccurrence

# Mixed Methods Commands
integrate            # --strategy, --quant-finding, --joint-display, --format

# Utility Commands
qual_status          # Show qualitative analysis progress
qual_export          # Export to NVivo/Atlas.ti format
codebook_edit        # Interactive codebook management
```

---

## Documentation Updates

### New Documentation Files

| File                        | Purpose                          |
| --------------------------- | -------------------------------- |
| `doc/QUALITATIVE_ANALYSIS.md` | Comprehensive qual methods guide |
| `doc/MIXED_METHODS.md`        | Integration strategies and workflows |
| `doc/CODEBOOK_GUIDE.md`       | Codebook schema and management   |
| `doc/QUAL_TOOL_INTEROP.md`    | NVivo/Atlas.ti import/export     |

### Updates to Existing Files

| File                     | Changes                              |
| ------------------------ | ------------------------------------ |
| `doc/README.md`            | Add qual/mixed methods section to index |
| `doc/PIPELINE.md`          | Add stages s10-s15 documentation     |
| `doc/ARCHITECTURE.md`      | Add qual track to data flow diagram  |
| `doc/STAGE_DEVELOPMENT.md` | Add qual stage examples              |
| `doc/skills.md`            | Add qual commands to reference       |
| `CLAUDE.md`                | Add qual quick reference section     |
| `README.md`                | Update feature list and stage count  |
| `doc/CHANGELOG.md`         | Document the addition                |

---

## File Structure

### New Directories

```text
data_raw/
└── qualitative/           # Raw qualitative data
    ├── transcripts/       # Interview transcripts
    ├── field_notes/       # Field notes
    └── imports/           # NVivo/Atlas.ti exports

data_work/
├── qual/                  # Qualitative working data
│   ├── corpus.parquet
│   ├── segments.parquet
│   ├── coded_segments.parquet
│   ├── codebook.yaml
│   ├── themes.yaml
│   ├── quote_bank.parquet
│   └── memos.parquet
└── mixed/                 # Mixed methods integration
    ├── joint_display.parquet
    └── integration_matrix.xlsx

src/stages/
├── s10_qual_ingest.py
├── s11_coding.py
├── s12_themes.py
├── s13_qual_synthesis.py
├── s14_qual_figures.py
├── s15_integration.py
└── _qual_utils.py         # Shared qualitative utilities

src/qual/                  # Qualitative-specific modules
├── __init__.py
├── parsers/               # Format-specific parsers
│   ├── nvivo.py
│   ├── atlas.py
│   ├── maxqda.py
│   └── transcript.py
├── coding/                # Coding logic
│   ├── deductive.py
│   ├── inductive.py
│   └── ai_coder.py
├── analysis/              # Analysis methods
│   ├── thematic.py
│   ├── framework.py
│   └── grounded.py
└── visualization/         # Qual figure generators
    ├── networks.py
    ├── matrices.py
    └── quote_displays.py

figures/
├── qual_*.png             # Qualitative figures
└── mixed_*.png            # Mixed methods figures

tests/
└── test_stages/
    ├── test_s10_qual_ingest.py
    ├── test_s11_coding.py
    ├── test_s12_themes.py
    ├── test_s13_qual_synthesis.py
    ├── test_s14_qual_figures.py
    └── test_s15_integration.py
```

---

## Implementation Phases

### Phase 1: Foundation (Core Infrastructure)

- [ ] Create `src/qual/` module structure
- [ ] Add configuration sections to `config.py`
- [ ] Implement `s10_qual_ingest.py` with basic transcript support
- [ ] Create `data_work/qual/` directory structure
- [ ] Add basic CLI commands to `pipeline.py`
- [ ] Write `doc/QUALITATIVE_ANALYSIS.md`

### Phase 2: Coding Infrastructure

- [ ] Implement codebook schema and YAML handling
- [ ] Implement `s11_coding.py` with deductive mode
- [ ] Add NVivo/Atlas.ti import parsers
- [ ] Implement coding QA reports
- [ ] Add `qual_code` CLI command with options

### Phase 3: Theme Analysis

- [ ] Implement `s12_themes.py` with thematic analysis
- [ ] Add co-occurrence matrix calculation
- [ ] Implement framework analysis option
- [ ] Add AI-assisted theme suggestion
- [ ] Implement saturation analysis

### Phase 4: Synthesis & Output

- [ ] Implement `s13_qual_synthesis.py`
- [ ] Create quote bank and manuscript integration
- [ ] Add AI-assisted summary generation
- [ ] Implement `s14_qual_figures.py`
- [ ] Create network and matrix visualizations

### Phase 5: Mixed Methods Integration

- [ ] Implement `s15_integration.py`
- [ ] Create joint display generation
- [ ] Add convergence/divergence analysis
- [ ] Implement integration strategies
- [ ] Update s09_writing.py for mixed methods drafting

### Phase 6: Polish & Documentation

- [ ] Complete all documentation updates
- [ ] Add comprehensive test coverage
- [ ] Create example/demo qualitative dataset
- [ ] Add export back to NVivo/Atlas.ti
- [ ] Performance optimization and caching

---

## Dependencies

### New Python Packages

```text
# Text processing
pdfplumber>=0.9.0          # PDF text extraction
python-docx>=0.8.11        # DOCX parsing
striprtf>=0.0.26           # RTF parsing
chardet>=5.0.0             # Encoding detection

# Visualization
networkx>=3.0              # Network graphs
pyvis>=0.3.0               # Interactive networks (optional)
matplotlib>=3.7.0          # Already present
seaborn>=0.12.0            # Already present

# NLP (optional, for AI-assisted features)
spacy>=3.5.0               # Sentence segmentation
sentence-transformers>=2.2 # Semantic similarity (optional)

# Export formats
openpyxl>=3.1.0            # Excel export (likely already present)
pyyaml>=6.0                # YAML handling (likely already present)
```

---

## AI Integration Points

### LLM-Assisted Features (using existing `src/llm/` infrastructure)

| Feature                  | Prompt Template                | Human Review Required        |
| ------------------------ | ------------------------------ | ---------------------------- |
| Code suggestion          | `prompts/qual_code_suggest.txt`  | Yes - all suggestions        |
| Theme grouping           | `prompts/qual_theme_suggest.txt` | Yes - before finalizing      |
| Quote selection          | `prompts/qual_quote_select.txt`  | Yes - verify representativeness |
| Theme summaries          | `prompts/qual_theme_summary.txt` | Yes - edit before manuscript |
| Integration interpretation | `prompts/mixed_interpret.txt`    | Yes - verify accuracy        |

### Safeguards

- All AI outputs marked with provenance metadata
- `--dry-run` flag shows prompts without execution
- Confidence scores on AI suggestions
- Audit log of AI-assisted decisions
- Human must explicitly approve before manuscript integration

---

## Critical Files to Modify

| File                          | Modification Type                   |
| ----------------------------- | ----------------------------------- |
| `src/config.py`                 | Add qual/mixed configuration sections |
| `src/pipeline.py`               | Register new commands and parsers   |
| `src/stages/_qa_utils.py`       | Add qual-specific QA metrics        |
| `src/llm/prompts.py`            | Add qual prompt templates           |
| `src/llm/__init__.py`           | Export new prompt functions         |
| `manuscript_quarto/_common.py`  | Add qual data loading helpers       |

---

## Risks & Mitigations

| Risk                         | Mitigation                                      |
| ---------------------------- | ----------------------------------------------- |
| Scope creep in qual features | Focus on integration-first; defer advanced features |
| AI coding quality concerns   | Require human review; track accuracy metrics    |
| NVivo/Atlas.ti format changes | Version-specific parsers; graceful degradation  |
| Performance with large corpora | Implement caching; segment-level processing     |
| Qual researcher adoption     | Prioritize import from existing tools           |

---

## Success Criteria

1. Researcher can import NVivo project and continue analysis in CENTAUR
2. Coded qualitative data integrates seamlessly with manuscript
3. Joint displays generated automatically from quant + qual outputs
4. AI-assisted features reduce coding time by 50%+ (with human verification)
5. Full audit trail maintained for reproducibility

---

## Related Documents

- [PIPELINE.md](../PIPELINE.md) — Stage reference
- [ARCHITECTURE.md](../ARCHITECTURE.md) — System design
- [STAGE_DEVELOPMENT.md](../STAGE_DEVELOPMENT.md) — Stage patterns
- [skills.md](../skills.md) — Command reference
