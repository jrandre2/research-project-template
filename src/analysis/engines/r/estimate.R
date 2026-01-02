#!/usr/bin/env Rscript
# CENTAUR R Estimation Engine
#
# Fixed effects estimation using the fixest package.
#
# Usage:
#   Rscript estimate.R <data.parquet> <spec.json> <output.json>
#
# Arguments:
#   data.parquet  - Path to input data in Parquet format
#   spec.json     - Path to specification JSON file
#   output.json   - Path to write results JSON file
#
# Required packages: arrow, fixest, jsonlite

suppressPackageStartupMessages({
  library(arrow)
  library(fixest)
  library(jsonlite)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop("Usage: Rscript estimate.R <data.parquet> <spec.json> <output.json>")
}

data_path <- args[1]
spec_path <- args[2]
output_path <- args[3]

# Read data
cat("Loading data from:", data_path, "\n")
df <- read_parquet(data_path)
cat("  Loaded", nrow(df), "rows,", ncol(df), "columns\n")

# Read specification
cat("Loading specification from:", spec_path, "\n")
spec <- fromJSON(spec_path)

# Build formula
build_formula <- function(spec) {
  outcome <- spec$outcome
  treatment <- spec$treatment
  controls <- spec$controls

  # Build RHS
  rhs <- treatment
  if (!is.null(controls) && length(controls) > 0) {
    rhs <- paste(c(rhs, controls), collapse = " + ")
  }

  # Add fixed effects
  fe <- spec$fixed_effects
  if (!is.null(fe) && length(fe) > 0) {
    fe_part <- paste(fe, collapse = " + ")
    formula_str <- paste0(outcome, " ~ ", rhs, " | ", fe_part)
  } else {
    formula_str <- paste0(outcome, " ~ ", rhs)
  }

  return(as.formula(formula_str))
}

# Run estimation
cat("Running estimation...\n")
start_time <- Sys.time()

formula <- build_formula(spec)
cat("  Formula:", deparse(formula), "\n")

# Handle clustering
cluster_formula <- NULL
if (!is.null(spec$cluster) && spec$cluster != "" && !is.na(spec$cluster)) {
  cluster_formula <- as.formula(paste0("~", spec$cluster))
  cat("  Clustering by:", spec$cluster, "\n")
}

# Run fixest estimation
tryCatch({
  if (!is.null(cluster_formula)) {
    model <- feols(formula, data = df, cluster = cluster_formula)
  } else {
    model <- feols(formula, data = df)
  }
}, error = function(e) {
  stop(paste("Estimation failed:", e$message))
})

end_time <- Sys.time()
execution_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat("  Estimation completed in", round(execution_time, 2), "seconds\n")

# Extract results
coefs <- coef(model)
ses <- se(model)
tstats <- coefs / ses
pvals <- pvalue(model)
ci <- confint(model)

# Get R-squared values
r2_vals <- r2(model, type = "all")

# Count units and periods
n_units <- 0
n_periods <- 0
fe <- spec$fixed_effects
if (!is.null(fe) && length(fe) > 0) {
  if (fe[1] %in% names(df)) {
    n_units <- length(unique(df[[fe[1]]]))
  }
  if (length(fe) > 1 && fe[2] %in% names(df)) {
    n_periods <- length(unique(df[[fe[2]]]))
  }
}

# Build result object
# Convert named vectors to lists for JSON
coef_list <- as.list(coefs)
se_list <- as.list(ses)
tstat_list <- as.list(tstats)
pval_list <- as.list(pvals)

# Confidence intervals as named lists
ci_lower <- as.list(ci[, 1])
ci_upper <- as.list(ci[, 2])
names(ci_lower) <- rownames(ci)
names(ci_upper) <- rownames(ci)

result <- list(
  specification = ifelse(is.null(spec$name), "unnamed", spec$name),
  n_obs = as.integer(model$nobs),
  n_units = as.integer(n_units),
  n_periods = as.integer(n_periods),
  coefficients = coef_list,
  std_errors = se_list,
  t_stats = tstat_list,
  p_values = pval_list,
  ci_lower = ci_lower,
  ci_upper = ci_upper,
  r_squared = ifelse(is.null(r2_vals["r2"]), 0, as.numeric(r2_vals["r2"])),
  r_squared_within = ifelse(is.null(r2_vals["wr2"]), NA, as.numeric(r2_vals["wr2"])),
  fixed_effects = if (is.null(spec$fixed_effects)) list() else as.list(spec$fixed_effects),
  cluster_var = ifelse(is.null(spec$cluster) || is.na(spec$cluster), NA, spec$cluster),
  controls = if (is.null(spec$controls)) list() else as.list(spec$controls),
  warnings = character(0),
  engine = "r",
  engine_version = paste0("fixest ", packageVersion("fixest")),
  execution_time_seconds = execution_time
)

# Write results
cat("Writing results to:", output_path, "\n")
write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, na = "null")

cat("Done.\n")
