#!/usr/bin/env Rscript
# Check required packages for CENTAUR R engine
#
# Usage:
#   Rscript check_packages.R
#
# Returns:
#   Exit 0 and prints "OK" with versions if all packages installed
#   Exit 1 and prints "MISSING: <packages>" if any missing

required <- c("arrow", "fixest", "jsonlite")
installed <- installed.packages()[, "Package"]
missing <- setdiff(required, installed)

if (length(missing) > 0) {
  cat("MISSING:", paste(missing, collapse = ", "), "\n")
  quit(status = 1)
}

# Print OK and versions
cat("OK\n")
for (pkg in required) {
  ver <- tryCatch(
    as.character(packageVersion(pkg)),
    error = function(e) "unknown"
  )
  cat(pkg, ":", ver, "\n", sep = "")
}
