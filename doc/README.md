# Documentation Index

**Project**: [Your Project Name]
**Last Updated**: [Date]

---

## Quick Start

| Document | Location | Purpose |
|----------|----------|---------|
| **CLAUDE.md** | Root | AI agent project instructions |
| **README.md** | Root | Project overview, setup, key commands |

---

## Core References

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [PIPELINE.md](PIPELINE.md) | Pipeline stages and CLI | Running pipeline |
| [METHODOLOGY.md](METHODOLOGY.md) | Statistical methods | Methodology review |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | Variable definitions | Variable lookups |

---

## Guides

| Document | Purpose |
|----------|---------|
| [agents.md](agents.md) | AI agent guidelines |
| [skills.md](skills.md) | Available skills/actions |
| [REPRODUCTION.md](REPRODUCTION.md) | Running from scratch |

---

## Synthetic Review System

| Document | Purpose |
|----------|---------|
| [SYNTHETIC_REVIEW_PROCESS.md](SYNTHETIC_REVIEW_PROCESS.md) | Review methodology and prompts |
| [MANUSCRIPT_REVISION_CHECKLIST.md](MANUSCRIPT_REVISION_CHECKLIST.md) | High-level revision status |
| [reviews/README.md](reviews/README.md) | Review cycles index |

**CLI Commands:**
- `python src/pipeline.py review_status` - Check status
- `python src/pipeline.py review_new --discipline economics` - Start new review
- `python src/pipeline.py review_archive` - Archive completed cycle
- `python src/pipeline.py review_verify` - Run verification
- `python src/pipeline.py review_report` - Summary report

---

## Status Tracking

| Document | Purpose |
|----------|---------|
| [CHANGELOG.md](CHANGELOG.md) | Change history |

---

## Adding New Documentation

When adding a new document:

1. Create the file in `doc/`
2. Add an entry to this index
3. Add a related link header to the new file:

```markdown
**Related**: [File1](file1.md) | [File2](file2.md)
**Status**: Active
**Last Updated**: YYYY-MM-DD
```

## Document Status Legend

- **Active** - In use, regularly updated
- **Reference** - Stable, infrequently changed
- **Archive** - Historical, moved to `archive/`
