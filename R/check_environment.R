check_environment <- function() {

  if (!requireNamespace("torch", quietly = TRUE)) {
    stop(
      "Package 'torch' is not installed. ",
      "Run source('scripts/setup.R') first."
    )
  }

  ok <- tryCatch(
    {
      x <- torch::torch_tensor(1)
      invisible(x)
      TRUE
    },
    error = function(e) {
      message("Torch loading error: ", conditionMessage(e))
      FALSE
    }
  )

  if (!ok) {
    stop(
      "Torch is installed as an R package, but LibTorch/LibLantern ",
      "could not be loaded. Run source('scripts/setup.R') and restart R."
    )
  }

  invisible(TRUE)
}