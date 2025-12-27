# Git Branching Strategy

> **Purpose**: Document branching patterns for managing analytical alternatives,
> manuscript revisions, and experimental work without losing reproducibility.

## Branch Naming Conventions

### Main Branches

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready code and manuscript |
| `develop` | Integration branch for features (optional) |

### Feature Branches

| Pattern | Purpose | Example |
|---------|---------|---------|
| `feature/<name>` | New functionality | `feature/add-robustness-checks` |
| `fix/<name>` | Bug fixes | `fix/duration-calculation` |

### Analysis Branches

| Pattern | Purpose | Example |
|---------|---------|---------|
| `analysis/<name>` | Exploratory analyses | `analysis/alternative-specifications` |
| `analysis/<name>-v2` | Revised exploration | `analysis/alternative-specifications-v2` |

Use analysis branches when:

- Testing alternative model specifications
- Exploring different operationalizations of variables
- Conducting sensitivity analyses that may not be included in final paper
- Responding to reviewer suggestions that require major changes

### Manuscript Branches

| Pattern | Purpose | Example |
|---------|---------|---------|
| `manuscript/<name>` | Major manuscript revisions | `manuscript/revision-r1` |
| `manuscript/<journal>` | Journal-specific version | `manuscript/jeem-submission` |

---

## Workflow Examples

### Starting an Analysis Branch

When exploring an alternative approach:

```bash
# Create and switch to new branch
git checkout -b analysis/alternative-capacity-measures

# Make changes and commit
git add .
git commit -m "Explore log-transformed capacity instead of ratios"

# Push to remote for backup
git push -u origin analysis/alternative-capacity-measures
```

### Incorporating Results into Main

If the analysis produces valid results to include:

```bash
# Switch to main
git checkout main

# Merge the analysis branch
git merge analysis/alternative-capacity-measures

# Push updated main
git push origin main
```

### Archiving Null or Rejected Analyses

If results are null or the approach is rejected, preserve for transparency:

```bash
# Tag the branch state before leaving it
git tag -a v0.2.0-null-findings -m "Null finding from alternative capacity measures"
git push origin v0.2.0-null-findings

# Switch back to main
git checkout main

# Document in ANALYSIS_JOURNEY.md why this approach was archived
```

### Handling Reviewer Requests

When reviewers request major changes:

```bash
# Create branch for revision
git checkout -b manuscript/revision-r1

# Make requested changes
# ... edit files ...

git add .
git commit -m "R1: Add sensitivity analysis per Reviewer 2"

# If satisfied, merge back to main
git checkout main
git merge manuscript/revision-r1
```

---

## Version Tagging

Use semantic versioning with descriptive suffixes:

| Tag Pattern | Meaning | Example |
|-------------|---------|---------|
| `v0.X.X` | Pre-submission versions | `v0.1.0`, `v0.3.0` |
| `v1.0.0` | Initial submission | `v1.0.0` |
| `v1.X.X` | Post-submission revisions | `v1.1.0` (R1), `v1.2.0` (R2) |
| `v*-<suffix>` | Analytical milestones | `v0.2.0-null-findings` |
| `v*-<journal>` | Journal submissions | `v1.0.0-jeem-submission` |

### Creating Tags

```bash
# Annotated tag with message
git tag -a v1.0.0 -m "Initial JEEM submission"

# Push tag to remote
git push origin v1.0.0

# List all tags
git tag -l
```

---

## Branch Lifecycle

### Active Branches

Keep branches that:

- Contain ongoing work
- Represent alternative approaches still under consideration
- Are needed for reproducibility of published results

### Archiving Branches

Archive (but don't delete) branches that:

- Contain completed exploratory work
- Document approaches that were tried but rejected
- Preserve reproducibility of intermediate results

To archive a branch:

```bash
# Create a tag to preserve the state
git tag -a archive/analysis-name -m "Archived: reason for archiving"
git push origin archive/analysis-name

# Optionally delete the branch (tag preserves the commits)
git branch -d analysis/analysis-name
git push origin --delete analysis/analysis-name
```

---

## Best Practices

### DO

- Create a new branch for any significant analytical experiment
- Tag important milestones (submissions, null findings, bug fixes)
- Document branch purposes in commit messages
- Keep `main` in a working, reproducible state
- Update `ANALYSIS_JOURNEY.md` when archiving analytical branches

### DON'T

- Force push to `main` (use `--force-with-lease` only if necessary)
- Delete branches without tagging if they contain important work
- Leave long-running branches without periodic merges from `main`
- Commit directly to `main` for significant changes

---

## Recovery Scenarios

### Recovering a Deleted Branch

If a branch was deleted but tagged:

```bash
# Recreate branch from tag
git checkout -b analysis/old-name archive/old-name
```

### Finding Old Commits

If commits seem lost:

```bash
# Show all commits, including orphaned ones
git reflog

# Recover a specific commit
git checkout <commit-hash>
git checkout -b recovery/branch-name
```

---

## Integration with CENTAUR Pipeline

### Branch-Specific Data

If branches require different data transformations:

1. Create versioned stage files (e.g., `s00b_ingest.py`)
2. Document which branch uses which stage version
3. Keep data outputs in branch-specific subdirectories if needed

### Manuscript Branches

When working on journal-specific submissions:

1. Create `manuscript/<journal>` branch
2. Use journal profile: `./render_all.sh --profile <journal>`
3. Keep journal-specific customizations on that branch
4. Merge common improvements back to `main`

---

*See also: [ANALYSIS_JOURNEY.md](ANALYSIS_JOURNEY.md) for documenting methodological evolution*
