#!/usr/bin/env Rscript

# Intrinsic medoid curves from each final model M1--M4.
#
# The script uses the model definitions from the simulation project directly.
# Curves are shown before the group nuisance action is applied.
# For each model/group/delta cell, the plotted curve is the medoid of a Monte Carlo sample.
#
# From the project root:
#   Rscript analysis/plot_model_realizations.R
#
# Optional environment variables:
#   MODEL_PLOT_SEED=20260916
#   MODEL_PLOT_N=100
#   MODEL_PLOT_P=512
#   MODEL_PLOT_WINDOW=4
#
# Output:
#   figures/fig_model_medoids.pdf

args0 <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args0[grep("^--file=", args0)])

if (length(file_arg)) {
  script_dir <- dirname(normalizePath(file_arg[[1L]], mustWork = FALSE))
  root <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
} else {
  root <- normalizePath(".", mustWork = FALSE)
}

source(file.path(root, "R", "config.R"))
source(file.path(root, "R", "models.R"))

seed <- env_int("MODEL_PLOT_SEED", 20260916L)
p_plot <- env_int("MODEL_PLOT_P", 512L)
window <- env_num("MODEL_PLOT_WINDOW", 4)
n_medoids <- env_int("MODEL_PLOT_N", 100L)

if (p_plot < 32L) {
  stop("MODEL_PLOT_P must be at least 32.")
}
if (!is.finite(window) || window <= 0) {
  stop("MODEL_PLOT_WINDOW must be positive.")
}
if (n_medoids < 2L) {
  stop("MODEL_PLOT_N must be at least 2.")
}

cfg <- make_shared_cfg()
validate_shared_cfg(cfg)

# The figure displays intrinsic model variability without measurement noise.
# Latent random amplitudes, coefficients, positions, and other model-specific
# random effects remain active.
cfg$noise_sd_mult <- 0
cfg$noise_sd_add <- 0

t <- seq(-window, window, length.out = p_plot)
model_names <- names(SCENARIOS)
groups <- c("X", "Y")
deltas <- c(1.0)

medoid_curve <- function(Z) {
  Z <- as.matrix(Z)
  D <- as.matrix(stats::dist(Z))
  Z[which.min(rowSums(D)), ]
}

curve_grid <- list()

for (j in seq_along(model_names)) {
  m <- model_names[j]
  tmp <- list()
  k <- 0L

  for (d in deltas) {
    for (g in groups) {
      k <- k + 1L
      set.seed(seed + 1000L * j + 100L * k)

      Z <- t(vapply(
        seq_len(n_medoids),
        function(i) {
          generate_intrinsic_line(
            t = t,
            scenario = SCENARIOS[[m]],
            delta = d,
            group = g,
            cfg = cfg
          )
        },
        numeric(length(t))
      ))

      tmp[[paste0(g, "_delta_", d)]] <- medoid_curve(Z)
    }
  }

  curve_grid[[m]] <- tmp
}

fig_dir <- file.path(root, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
pdf_file <- file.path(fig_dir, "fig_model_medoids.pdf")

draw_models <- function() {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)

  par(
    mfrow = c(2, 2),
    mar = c(3.4, 3.6, 2.3, 0.8),
    oma = c(0.5, 0.5, 0.3, 0.2),
    mgp = c(2.0, 0.65, 0),
    tcl = -0.25
  )

  curve_keys <- c("X_delta_1", "Y_delta_1")
  line_types <- c(1, 1)
  line_widths <- c(2.2, 2.2)
  line_cols <- c("#1F77B4", "#2CA02C")

  for (m in model_names) {
    yy <- unlist(curve_grid[[m]][curve_keys], use.names = FALSE)
    ylim <- range(yy, finite = TRUE)

    plot(
      t,
      curve_grid[[m]][[curve_keys[1]]],
      type = "l",
      lwd = line_widths[1],
      lty = line_types[1],
      col = line_cols[1],
      xlab = "t",
      ylab = "Signal",
      main = m,
      ylim = ylim
    )

    for (i in 2:length(curve_keys)) {
      lines(
        t,
        curve_grid[[m]][[curve_keys[i]]],
        lwd = line_widths[i],
        lty = line_types[i],
        col = line_cols[i]
      )
    }

    abline(h = 0, lty = 3, lwd = 0.6)

    legend(
      "topright",
      legend = c("X, delta = 1", "Y, delta = 1"),
      col = line_cols,
      lty = line_types,
      lwd = line_widths,
      cex = 0.72,
      bty = "n"
    )
  }
}

pdf(pdf_file, width = 8.2, height = 6.0)
draw_models()
dev.off()

if (interactive()) {
  draw_models()
}

message(
  "Colored noiseless model medoids written to: ",
  normalizePath(pdf_file, mustWork = FALSE),
  "\nCurves: X/Y at delta = 1",
  "\nMeasurement noise disabled; intrinsic latent variability retained.",
  "\nMedoid sample size per cell = ", n_medoids,
  "\nSeed = ", seed
)
