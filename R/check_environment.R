check_environment <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop(
      "Package 'torch' is not installed. ",
      "Run source('scripts/setup.R') first."
    )
  }

  if (!torch::torch_is_installed(recheck = TRUE)) {
    stop(
      "LibTorch/LibLantern are not installed. ",
      "Run source('scripts/setup.R') first."
    )
  }

  invisible(TRUE)
}