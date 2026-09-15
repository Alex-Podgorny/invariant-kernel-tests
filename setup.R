#!/usr/bin/env Rscript

# Restore exact R package versions
renv::restore(prompt = FALSE)

# Reference implementation for the paper: CPU
Sys.setenv(CUDA = "cpu")

# Install LibTorch + LibLantern corresponding to the
# torch R package version recorded in renv.lock
if (!torch::torch_is_installed(recheck = TRUE)) {
  torch::install_torch()
}

cat("\nEnvironment ready.\n")
cat("R version: ", R.version.string, "\n")
cat("torch R version: ", as.character(packageVersion("torch")), "\n")
cat("torch available: ", torch::torch_is_installed(recheck = TRUE), "\n")