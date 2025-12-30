# Methodology

**Related**: [PIPELINE.md](PIPELINE.md) | [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | [ARCHITECTURE.md](ARCHITECTURE.md)
**Status**: Template
**Last Updated**: [Date]

---

This document provides methodological guidance for common causal inference designs. Customize the sections relevant to your research.

## Research Design Overview

[Replace with your project-specific research design description]

### Common Design Patterns

This platform supports several common research designs:

| Design | Use Case | Key Requirement |
|--------|----------|-----------------|
| Difference-in-Differences | Policy change affecting treatment group | Parallel trends |
| Event Study | Dynamic effects with clear timing | Sharp treatment timing |
| Regression Discontinuity | Running variable determines treatment | Local randomization |
| Panel Fixed Effects | Repeated observations on same units | Time-invariant unobservables |

---

## Difference-in-Differences (DiD)

### Basic Setup

Compare treated and control groups, before and after treatment:

$$
Y_{it} = \alpha + \beta \cdot (Treated_i \times Post_t) + \gamma_i + \delta_t + \varepsilon_{it}
$$

where:
- $Y_{it}$ = Outcome for unit $i$ at time $t$
- $Treated_i$ = 1 if unit ever receives treatment
- $Post_t$ = 1 if period is after treatment
- $\gamma_i$ = Unit fixed effects
- $\delta_t$ = Time fixed effects
- $\beta$ = Average treatment effect on the treated (ATT)

### Implementation

```python
# In s03_estimation.py
SPECIFICATIONS = {
    'did_basic': {
        'name': 'Basic DiD',
        'outcome': 'outcome',
        'treatment': 'treat_x_post',  # Interaction term
        'fe': ['unit_id', 'period'],
        'cluster': 'unit_id',
    },
}
```

### Identifying Assumptions

1. **Parallel Trends**: In absence of treatment, treatment and control groups would have followed parallel outcome paths

2. **No Anticipation**: Units do not change behavior in anticipation of treatment

3. **SUTVA**: Treatment of one unit does not affect outcomes of other units

### Testing Parallel Trends

Visual test: Plot pre-treatment trends for treatment and control groups.

Statistical test: Regress outcome on leads of treatment:

$$
Y_{it} = \alpha + \sum_{k=-K}^{-1} \beta_k \cdot D_{it}^k + \gamma_i + \delta_t + \varepsilon_{it}
$$

where $D_{it}^k$ is an indicator for $k$ periods before treatment.

**Joint F-test**: Test $H_0: \beta_{-K} = \beta_{-K+1} = ... = \beta_{-1} = 0$

| Result | Interpretation |
|--------|----------------|
| F-stat small, p > 0.10 | Parallel trends assumption supported |
| F-stat large, p < 0.05 | Pre-trends present, caution needed |

---

## Event Study Design

### Dynamic Treatment Effects

Estimate treatment effects at each relative time period:

$$
Y_{it} = \alpha + \sum_{k \neq -1} \beta_k \cdot \mathbf{1}[t - E_i = k] + \gamma_i + \delta_t + \varepsilon_{it}
$$

where:
- $E_i$ = Treatment timing for unit $i$
- $\mathbf{1}[t - E_i = k]$ = Indicator for $k$ periods from treatment
- $k = -1$ is omitted as reference period
- $\beta_k$ = Treatment effect at relative time $k$

### Implementation

```python
# In s02_panel.py
def create_event_time(df, treatment_time_col='treat_date', time_col='period'):
    """Create event time indicators."""
    df['event_time'] = df[time_col] - df[treatment_time_col]

    # Create dummies, omitting k=-1
    for k in range(-12, 13):
        if k != -1:
            df[f'event_k{k}'] = (df['event_time'] == k).astype(int)

    return df
```

### Interpreting Event Study Plots

```
Pre-treatment effects (k < 0):
  - Should be near zero if parallel trends hold
  - Significant pre-trends indicate potential confounds

Treatment effect (k = 0):
  - Immediate effect at treatment time

Post-treatment effects (k > 0):
  - Dynamic treatment effects
  - Persistent vs. fading effects
```

### Staggered Adoption

When treatment timing varies across units, use heterogeneity-robust estimators:

| Method | Reference | Implementation |
|--------|-----------|----------------|
| Callaway & Sant'Anna | did package (R) | Cohort-specific ATT |
| Sun & Abraham | fixest (R) | Interaction-weighted |
| de Chaisemartin & D'Haultfoeuille | did_multiplegt (Stata) | Weighted averages |

---

## Regression Discontinuity (RD)

### Sharp RD Design

Treatment is deterministic function of running variable:

$$
D_i = \mathbf{1}[X_i \geq c]
$$

Estimate:

$$
Y_i = \alpha + \tau D_i + f(X_i - c) + \varepsilon_i
$$

where:
- $X_i$ = Running variable
- $c$ = Cutoff
- $D_i$ = Treatment indicator
- $f(\cdot)$ = Polynomial in normalized running variable
- $\tau$ = Local average treatment effect at cutoff

### Implementation

```python
# In s03_estimation.py
def run_rd_estimation(df, running_var, cutoff, bandwidth, outcome_var):
    """Run RD estimation with local linear regression."""
    # Center running variable
    df['running_centered'] = df[running_var] - cutoff
    df['treated'] = (df[running_var] >= cutoff).astype(int)

    # Restrict to bandwidth
    df_local = df[abs(df['running_centered']) <= bandwidth]

    # Allow different slopes on each side
    df_local['running_x_treat'] = df_local['running_centered'] * df_local['treated']

    # Estimate
    result = run_ols(
        df_local,
        y_var=outcome_var,
        x_vars=['treated', 'running_centered', 'running_x_treat']
    )

    return result
```

### Bandwidth Selection

| Method | Description |
|--------|-------------|
| Imbens-Kalyanaraman | MSE-optimal bandwidth |
| CCT (Calonico et al.) | Robust bias-corrected inference |
| Cross-validation | Data-driven selection |

### RD Diagnostics

1. **McCrary Density Test**: Check for manipulation at cutoff
2. **Covariate Balance**: Predetermined characteristics smooth at cutoff
3. **Placebo Cutoffs**: No effect at artificial cutoffs
4. **Bandwidth Sensitivity**: Results stable across bandwidth choices

---

## Standard Error Computation

### Options

| Method | Use Case | Implementation |
|--------|----------|----------------|
| Heteroskedasticity-robust (HC) | Cross-section, unknown heteroskedasticity | `se='robust'` |
| Clustered | Panel data, within-group correlation | `cluster='unit_id'` |
| Conley spatial HAC | Spatial correlation | Custom implementation |
| Bootstrap | Small samples, complex estimators | Resampling |

### Clustered Standard Errors

For panel data with clustering at unit level:

$$
\hat{V}_{cluster} = (X'X)^{-1} \left( \sum_{g=1}^{G} X_g' \hat{u}_g \hat{u}_g' X_g \right) (X'X)^{-1}
$$

where $g$ indexes clusters.

### Implementation

```python
# In s03_estimation.py
def clustered_se(residuals, X, cluster_ids):
    """Compute cluster-robust standard errors."""
    clusters = np.unique(cluster_ids)
    G = len(clusters)
    n, k = X.shape

    # Cluster adjustment factor
    adjustment = (G / (G - 1)) * ((n - 1) / (n - k))

    # Meat of sandwich
    meat = np.zeros((k, k))
    for g in clusters:
        mask = cluster_ids == g
        X_g = X[mask]
        u_g = residuals[mask]
        meat += X_g.T @ np.outer(u_g, u_g) @ X_g

    # Sandwich
    XtX_inv = np.linalg.inv(X.T @ X)
    V = adjustment * XtX_inv @ meat @ XtX_inv

    return np.sqrt(np.diag(V))
```

### Choosing Cluster Level

- Cluster at level of treatment variation
- More conservative with fewer, larger clusters
- Consider two-way clustering for panel data

---

## Spatial Cross-Validation

### The Problem: Spatial Data Leakage

When data has geographic structure, standard random cross-validation can leak information between training and test sets due to spatial autocorrelation. Nearby observations are often similar, so a model trained on neighbors of test points can appear to generalize better than it actually does.

### Solution: Spatial Grouping

Group observations by geographic proximity before cross-validation:

```python
from utils.spatial_cv import SpatialCVManager

# Create spatial groups
manager = SpatialCVManager(n_groups=5, method='kmeans')
groups = manager.create_groups_from_coordinates(
    df['latitude'].values,
    df['longitude'].values
)

# Cross-validate with spatial separation
results = manager.cross_validate(model, X, y)
```

### Available Grouping Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `kmeans` | K-means clustering on coordinates | General geographic clustering |
| `balanced_kmeans` | K-means with balanced group sizes | Even fold sizes |
| `geographic_bands` | Latitude-based bands | N-S spatial structure |
| `longitude_bands` | Longitude-based bands | E-W spatial structure |
| `spatial_blocks` | Grid-based blocks | Regular spatial grids |
| `zip_digit` | ZIP code digit grouping | Administrative boundaries |
| `contiguity_queen` | Queen contiguity (requires geopandas) | Polygon data |
| `contiguity_rook` | Rook contiguity (requires geopandas) | Polygon data |

### Quantifying Leakage

Compare spatial CV to random CV to measure potential leakage:

```python
comparison = manager.compare_to_random_cv(model, X, y)
print(f"Leakage estimate: {comparison['leakage_pct']:.1f}%")
```

A large gap between random CV performance and spatial CV performance indicates that the model may be exploiting spatial correlation rather than learning generalizable patterns.

### When to Use Spatial CV

- Studies with geographic data (county, zip code, coordinates)
- Environmental or climate research
- Urban economics and real estate
- Epidemiological studies
- Any analysis where spatial autocorrelation is expected

### Configuration

Configure in `src/config.py`:

```python
SPATIAL_CV_N_GROUPS = 5
SPATIAL_GROUPING_METHOD = 'kmeans'
SPATIAL_SENSITIVITY_METHODS = ['kmeans', 'balanced_kmeans', 'geographic_bands']
```

---

## Robustness Checks

### Specification Robustness

| Check | Purpose | Implementation |
|-------|---------|----------------|
| Add controls | Test for omitted variable bias | Extend specification |
| Alternative FE | Different fixed effect structure | Change `fe` parameter |
| Different outcome | Mechanism tests | Change outcome variable |
| Functional form | Log vs. level, polynomials | Transform variables |

### Sample Robustness

| Check | Purpose | Implementation |
|-------|---------|----------------|
| Trim outliers | Remove extreme observations | Filter 1-99 percentile |
| Subsample | Test heterogeneity | Restrict by covariate |
| Time window | Sensitivity to period | Restrict date range |
| Drop clusters | Influential observations | Leave-one-out |

### Placebo Tests

| Test | Purpose | Implementation |
|------|---------|----------------|
| Fake treatment time | Pre-trend detection | Shift treatment earlier |
| Fake treatment group | Spillover detection | Randomize treatment |
| Outcome placebo | Mechanism test | Use unaffected outcome |

### Implementation

```python
# In s04_robustness.py
ROBUSTNESS_TESTS = {
    'trim_outliers': {
        'type': 'sample',
        'filter': lambda df: df[df['outcome'].between(
            df['outcome'].quantile(0.01),
            df['outcome'].quantile(0.99)
        )],
    },
    'placebo_t6': {
        'type': 'placebo',
        'fake_treatment_period': 6,
    },
    'alt_se': {
        'type': 'specification',
        'cluster': 'state_id',  # Alternative cluster level
    },
}
```

---

## Key Parameters

### Default Values

| Parameter | Description | Default | Justification |
|-----------|-------------|---------|---------------|
| `bandwidth` | RD bandwidth | MSE-optimal | Imbens-Kalyanaraman |
| `cluster_level` | SE clustering | Unit | Treatment variation |
| `pre_periods` | Event study window | 12 | Standard practice |
| `post_periods` | Event study window | 12 | Standard practice |

### Project-Specific Parameters

| Parameter | Description | Value | Source |
|-----------|-------------|-------|--------|
| [param1] | [description] | [value] | [justification] |
| [param2] | [description] | [value] | [justification] |

---

## Estimation Pipeline

### Stage Sequence

```
s02_panel.py (Panel Construction)
    ├── Create unit and time identifiers
    ├── Create treatment indicators
    ├── Create event time variables
    └── Create fixed effects

s03_estimation.py (Main Estimation)
    ├── Run specification registry
    ├── Compute standard errors
    ├── Generate coefficient table
    └── Save diagnostics

s04_robustness.py (Robustness)
    ├── Placebo tests
    ├── Sample restrictions
    ├── Alternative specifications
    └── Save results

s05_figures.py (Visualization)
    ├── Event study plot
    ├── Coefficient comparison
    └── Robustness summary
```

### Diagnostic Outputs

| File | Contents |
|------|----------|
| `estimation_results.csv` | Coefficients, SEs, p-values |
| `robustness_results.csv` | All robustness checks |
| `pretrend_test.csv` | Joint F-test for pre-trends |
| `balance_test.csv` | Covariate balance |

---

## References

### General Causal Inference

- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press.

### Difference-in-Differences

- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.

### Regression Discontinuity

- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to Regression Discontinuity Designs*. Cambridge University Press.
- Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to practice. *Journal of Econometrics*, 142(2), 615-635.

### Standard Errors

- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
