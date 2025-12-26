# [Project Title]

[One-paragraph description of the research project]

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python src/pipeline.py ingest_data
python src/pipeline.py link_records
python src/pipeline.py build_panel
python src/pipeline.py run_estimation

# Generate figures
python src/pipeline.py make_figures

# Render manuscript
cd manuscript_quarto && ./render_all.sh
```

## Project Structure

```
├── CLAUDE.md              # AI assistant instructions
├── README.md              # This file
├── requirements.txt       # Python dependencies
│
├── doc/                   # Documentation
│   ├── README.md          # Documentation index
│   ├── PIPELINE.md        # Pipeline stages
│   ├── METHODOLOGY.md     # Statistical methods
│   ├── DATA_DICTIONARY.md # Variable definitions
│   ├── agents.md          # AI agent guidelines
│   └── skills.md          # Available skills/actions
│
├── src/                   # Analysis pipeline
│   ├── pipeline.py        # CLI entry point
│   ├── stages/            # Numbered pipeline stages
│   │   ├── s00_ingest.py
│   │   ├── s01_link.py
│   │   ├── s02_panel.py
│   │   ├── s03_estimation.py
│   │   ├── s04_robustness.py
│   │   ├── s05_figures.py
│   │   └── s06_manuscript.py
│   └── utils/
│       ├── figure_style.py
│       └── helpers.py
│
├── manuscript_quarto/     # Quarto manuscript
│   ├── _quarto.yml        # Main configuration
│   ├── _quarto-jeem.yml   # JEEM profile
│   ├── _quarto-aer.yml    # AER profile
│   ├── render_all.sh      # Multi-format render script
│   ├── index.qmd          # Main manuscript
│   ├── appendix-*.qmd     # Appendices
│   ├── references.bib     # Bibliography
│   ├── journal_configs/   # Journal metadata
│   └── code/              # Rendering utilities
│
├── data_raw/              # Raw data (gitignored)
├── data_work/             # Working data (gitignored)
├── figures/               # Output figures
└── tools/
    └── bin/quarto         # Quarto wrapper
```

## Key Commands

| Command | Purpose |
|---------|---------|
| `python src/pipeline.py ingest_data` | Load raw data |
| `python src/pipeline.py link_records` | Link data sources |
| `python src/pipeline.py build_panel` | Create analysis panel |
| `python src/pipeline.py run_estimation` | Run estimation |
| `python src/pipeline.py estimate_robustness` | Robustness checks |
| `python src/pipeline.py make_figures` | Generate figures |
| `python src/pipeline.py validate_submission` | Check journal requirements |
| `./manuscript_quarto/render_all.sh` | Render manuscript (all formats) |
| `./manuscript_quarto/render_all.sh --profile jeem` | Render for JEEM |

## Journal Profiles

The manuscript can be rendered for different journals:

```bash
cd manuscript_quarto
./render_all.sh                    # Default (HTML + PDF + DOCX)
./render_all.sh --profile jeem     # JEEM submission format
./render_all.sh --profile aer      # AER submission format
```

See `manuscript_quarto/journal_configs/` for journal-specific requirements.

## Documentation

See `doc/README.md` for complete documentation index.

## Authors

- [Author 1] ([email])
- [Author 2] ([email])

## License

[License information]
