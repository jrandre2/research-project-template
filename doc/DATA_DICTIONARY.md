# Data Dictionary

**Related**: [PIPELINE.md](PIPELINE.md) | [METHODOLOGY.md](METHODOLOGY.md)
**Status**: Active
**Last Updated**: [Date]

---

## Overview

This document defines all variables used in the analysis.

### Conventions

- [Describe naming conventions]
- [Describe missing value handling]
- [Describe any signed conventions (e.g., negative = inside boundary)]

---

## Identifier Variables

| Variable | Type | Description |
|----------|------|-------------|
| `id` | string | Unique observation identifier |
| `unit_id` | string | Unit identifier |
| `time_id` | string | Time period identifier |

---

## Outcome Variables

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `outcome1` | float | [unit] | [Description] |
| `outcome2` | float | [unit] | [Description] |

---

## Treatment Variables

| Variable | Type | Description |
|----------|------|-------------|
| `treatment` | boolean | Treatment indicator (True = treated) |
| `post` | boolean | Post-treatment indicator (True = after treatment) |
| `treat_post` | boolean | Treatment × Post interaction |

---

## Control Variables

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `control1` | float | [unit] | [Description] |
| `control2` | categorical | - | [Description with categories] |

---

## Fixed Effect Variables

| Variable | Type | Description |
|----------|------|-------------|
| `unit_fe` | string | Unit fixed effect grouping |
| `time_fe` | string | Time fixed effect grouping |

---

## Derived Variables

| Variable | Formula | Description |
|----------|---------|-------------|
| `log_outcome` | `ln(outcome1)` | Log of outcome |
| `outcome_change` | `outcome1_t - outcome1_{t-1}` | First difference |

---

## Data Files

### Panel Data

**File:** `data_work/panel.parquet`

| Variable | Type | Description |
|----------|------|-------------|
| [Include all variables in main panel] | | |

### Diagnostic Files

**Directory:** `data_work/diagnostics/`

| File | Description |
|------|-------------|
| `main_results.csv` | Primary estimation results |
| `robustness_*.csv` | Robustness check results |
| `pretrends_*.csv` | Pre-trend test results |

---

## Value Coding

### Categorical Variables

**control2:**
| Code | Label |
|------|-------|
| 0 | [Category 0] |
| 1 | [Category 1] |
| 2 | [Category 2] |

---

## Missing Values

| Variable | Missing Code | Handling |
|----------|--------------|----------|
| [var] | NaN | [Dropped / Imputed / Other] |
