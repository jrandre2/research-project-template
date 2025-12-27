# Review Cycles Index

**Related**: [../SYNTHETIC_REVIEW_PROCESS.md](../SYNTHETIC_REVIEW_PROCESS.md) | [../MANUSCRIPT_REVISION_CHECKLIST.md](../MANUSCRIPT_REVISION_CHECKLIST.md)
**Status**: Active
**Last Updated**: [Date]

---

## Overview

This directory tracks all synthetic peer review cycles for the manuscript.

---

## Review History

| Review | Date | Discipline | Major | Minor | Addressed | Status |
|--------|------|------------|-------|-------|-----------|--------|
| | | | | | | |

---

## Current Review

**Active Review**: None

**Tracker**: [../../manuscript_quarto/REVISION_TRACKER.md](../../manuscript_quarto/REVISION_TRACKER.md)

---

## Archive

Completed review cycles are stored in `archive/`:

| File | Review | Date | Notes |
|------|--------|------|-------|
| | | | |

---

## Starting a New Review

1. Generate review using focus-specific prompt:
   ```bash
   python src/pipeline.py review_new --focus economics
   ```

2. Triage comments in `REVISION_TRACKER.md`

3. Implement changes and track in checklist

4. Verify and archive when complete:
   ```bash
   python src/pipeline.py review_verify
   python src/pipeline.py review_archive
   ```

---

## Discipline Templates Available

- `economics` - Identification, causal inference, econometrics
- `engineering` - Reproducibility, benchmarks, technical accuracy
- `social_sciences` - Theory, generalizability, ethics
- `general` - Structure, clarity, contribution

---

## File Organization

```
doc/reviews/
├── README.md           # This file
└── archive/
    ├── review_01.md    # Completed review #1
    ├── review_02.md    # Completed review #2
    └── ...
```
