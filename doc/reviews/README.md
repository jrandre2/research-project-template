# Review Cycles Index

**Related**: [../SYNTHETIC_REVIEW_PROCESS.md](../SYNTHETIC_REVIEW_PROCESS.md) | [../MANUSCRIPT_REVISION_CHECKLIST.md](../MANUSCRIPT_REVISION_CHECKLIST.md)
**Status**: Active
**Last Updated**: 2025-12-30

---

## Overview

This directory tracks all peer review cycles for the manuscript, including both **synthetic** (AI-generated) and **actual** (journal) reviews.

---

## Review History

| Review | Date | Type | Source | Major | Minor | Addressed | Status |
|--------|------|------|--------|-------|-------|-----------|--------|
| | | synthetic/actual | Focus or Journal | | | | |

---

## Current Review

**Active Review**: None

**Tracker**: [../../manuscript_quarto/REVISION_TRACKER.md](../../manuscript_quarto/REVISION_TRACKER.md)

---

## Archive

Completed review cycles are stored in `archive/`:

| File | Review | Type | Date | Notes |
|------|--------|------|------|-------|
| | | | | |

---

## Starting a New Review

### Synthetic Review (pre-submission)

```bash
# Generate review using focus-specific prompt
python src/pipeline.py review_new --focus economics
python src/pipeline.py review_new -m main -f methods
```

### Actual Review (from journal)

```bash
# Track journal review with metadata
python src/pipeline.py review_new --actual --journal "JEEM" --round "R&R1"
python src/pipeline.py review_new --actual -j "AER" -r "initial" --decision major_revision
```

### Common Workflow

1. Start review cycle (synthetic or actual)
2. Triage comments in `REVISION_TRACKER.md`
3. Implement changes and track in checklist
4. Verify and archive when complete:
   ```bash
   python src/pipeline.py review_verify
   python src/pipeline.py review_archive
   ```

---

## Additional Commands

```bash
# Generate visual diff between cycles
python src/pipeline.py review_diff -m main

# Generate response letter
python src/pipeline.py review_response -m main
```

---

## Focus Templates (Synthetic Reviews)

- `economics` - Identification, causal inference, econometrics
- `engineering` - Reproducibility, benchmarks, technical accuracy
- `social_sciences` - Theory, generalizability, ethics
- `general` - Structure, clarity, contribution
- `methods` - Statistical rigor, methodology critique
- `policy` - Practitioner perspective, actionability
- `clarity` - Writing quality, accessibility

---

## File Organization

```
doc/reviews/
├── README.md           # This file
└── archive/
    ├── review_01.md    # Completed review #1 (with YAML frontmatter)
    ├── review_02.md    # Completed review #2
    └── ...
```

Review files include YAML frontmatter with metadata (source type, journal, dates, git commits).
