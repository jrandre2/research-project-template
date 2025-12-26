# Methodology

**Related**: [PIPELINE.md](PIPELINE.md) | [DATA_DICTIONARY.md](DATA_DICTIONARY.md)
**Status**: Active
**Last Updated**: [Date]

---

## Research Design

[Describe your overall research design and approach]

### Key Features

- [Feature 1]
- [Feature 2]
- [Feature 3]

---

## Identification Strategy

[Describe your identification strategy]

### Identifying Assumptions

1. **[Assumption 1]**: [Description]

2. **[Assumption 2]**: [Description]

3. **[Assumption 3]**: [Description]

---

## Estimating Equations

### Main Specification

$$
Y_{it} = \alpha + \beta \cdot Treatment_{it} + \gamma \cdot X_{it} + \delta_i + \theta_t + \varepsilon_{it}
$$

where:
- $Y_{it}$ = Outcome for unit $i$ at time $t$
- $Treatment_{it}$ = Treatment indicator
- $X_{it}$ = Control variables
- $\delta_i$ = Unit fixed effects
- $\theta_t$ = Time fixed effects
- $\varepsilon_{it}$ = Error term

### Alternative Specifications

[Document alternative specifications]

---

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| [param1] | [description] | [value] |
| [param2] | [description] | [value] |

---

## Standard Errors

[Describe standard error computation]

Default: [HC3 / Clustered / Conley spatial]

---

## Robustness Checks

### Specification Robustness

- [Check 1]
- [Check 2]

### Sample Robustness

- [Check 1]
- [Check 2]

### Placebo Tests

- [Test 1]
- [Test 2]

---

## References

[Key methodological references]
