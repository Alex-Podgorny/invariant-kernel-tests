#!/usr/bin/env Rscript

# Restore exact R package versions
renv::restore(prompt = FALSE)

# Reference implementation for the paper: CPU
Sys.setenv(CUDA = "cpu")

# Check whether the native Torch dependencies are present
if (!torch::torch_is_installed()) {
  torch::install_torch()
  stop(
    "Torch dependencies have been installed. Restart R and rerun source('setup.R').",
    call. = FALSE
  )
}

# Verify actual Torch usability.
torch_ok <- tryCatch(
  {
    x <- torch::torch_tensor(1)
    TRUE
  },
  error = function(e) {
    message("\nTorch is installed but could not be loaded:")
    message(conditionMessage(e))
    FALSE
  }
)

if (!torch_ok) {
  stop(
    "Torch native libraries cannot be loaded. Check the Windows application-control policy for lantern.dll.",
    call. = FALSE
  )
}

cat("\nEnvironment ready.\n")
cat("R version: ", R.version.string, "\n")
cat("torch R version: ", as.character(packageVersion("torch")), "\n")
cat("torch available: TRUE\n")