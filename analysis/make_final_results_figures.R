#!/usr/bin/env Rscript

# Final figures and numerical summaries for the clean M1--M4 simulations.
#
# From the project root, the default invocation is simply:
#   Rscript analysis/make_final_results_figures.R
#
# Default inputs:
#   results/affine/summary_results.csv
#   results/translation/summary_results.csv
#   results/circular/summary_results.csv
#
# Optional positional arguments (kept compatible with the historical script):
#   1. affine CSV
#   2. translation CSV
#   3. circular CSV
#   4. output directory
#
# Runtime figures use medians by default, because timing outliers from OS / BLAS /
# LibTorch scheduling can strongly distort means. Set RUNTIME_STAT=mean to recover
# mean runtime figures.

required <- c("ggplot2", "dplyr")
missing <- required[!vapply(required, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing) > 0L) {
  stop(
    "Missing required R packages: ",
    paste(missing, collapse = ", "),
    ". Install them before running the script."
  )
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

# Resolve the project root from this file when invoked with Rscript, and otherwise
# fall back to the current working directory (convenient for source() in RStudio).
args0 <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args0[grep("^--file=", args0)])
if (length(file_arg)) {
  script_dir <- dirname(normalizePath(file_arg[[1L]], mustWork = FALSE))
  root <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
} else {
  root <- normalizePath(".", mustWork = FALSE)
}

args <- commandArgs(trailingOnly = TRUE)
affine_file <- if (length(args) >= 1L) args[[1L]] else file.path(root, "results", "affine", "summary_results.csv")
translation_file <- if (length(args) >= 2L) args[[2L]] else file.path(root, "results", "translation", "summary_results.csv")
circle_file <- if (length(args) >= 3L) args[[3L]] else file.path(root, "results", "circular", "summary_results.csv")
fig_dir <- if (length(args) >= 4L) args[[4L]] else file.path(root, "figures")

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

runtime_stat <- tolower(Sys.getenv("RUNTIME_STAT", unset = "median"))
if (!runtime_stat %in% c("median", "mean")) {
  stop("RUNTIME_STAT must be either 'median' or 'mean'.")
}

read_group <- function(path, group_name) {
  if (!file.exists(path)) {
    stop(
      "File not found: ", path, "\n",
      "Run the corresponding simulation first, or pass explicit CSV paths."
    )
  }
  z <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  z$Group <- group_name
  z
}

aff <- read_group(affine_file, "Affine")
tra <- read_group(translation_file, "Translation")
cir <- read_group(circle_file, "Circle")

# Allow harmless additions of group-specific columns while enforcing the common
# columns needed for the final figures.
required_cols <- c(
  "scenario", "delta", "method", "S", "rejection_rate", "nrep",
  "mean_train_seconds", "median_train_seconds",
  "mean_eval_seconds", "median_eval_seconds",
  "mean_total_seconds", "median_total_seconds"
)
for (obj in list(Affine = aff, Translation = tra, Circle = cir)) {
  miss <- setdiff(required_cols, names(obj))
  if (length(miss)) stop("Missing required result columns: ", paste(miss, collapse = ", "))
}

all_cols <- unique(c(names(aff), names(tra), names(cir)))
add_missing <- function(z, cols) {
  for (nm in setdiff(cols, names(z))) z[[nm]] <- NA
  z[, cols, drop = FALSE]
}
aff <- add_missing(aff, all_cols)
tra <- add_missing(tra, all_cols)
cir <- add_missing(cir, all_cols)
dat <- bind_rows(aff, tra, cir)

# Only final-paper methods. These names are the ones written by the clean project.
final_methods <- c(
  "Base-RFF",
  "Align-energy-median-RFF",
  "Align-xcorr-RFF",
  "Energy-prob-RFF",
  "Learned-prob-CNN-RFF",
  "Align-energy-circular-median",
  "Align-xcorr-circular",
  "Circular-Energy-prob",
  "Circular-Haar-RFF",
  "Learned-Circular-GCNN-RFF",
  "Affine-MovingChart-Wavelet-energy-prob",
  "Align-wavelet-max-affine",
  "Align-xcorr-affine",
  "Learned-Affine-GCNN-RFF"
)

dat <- dat |>
  filter(method %in% final_methods)

# The clean simulation project writes the final model labels M1--M4 directly.
model_levels <- c("M1", "M2", "M3", "M4")
unknown_models <- setdiff(unique(dat$scenario), model_levels)
if (length(unknown_models)) {
  stop(
    "Unexpected scenario labels: ", paste(unknown_models, collapse = ", "),
    ". The final figure script expects only M1, M2, M3, M4."
  )
}
dat$Model <- factor(dat$scenario, levels = model_levels)
dat$Group <- factor(dat$Group, levels = c("Translation", "Circle", "Affine"))

short_method <- function(x) {
  out <- x
  out[x == "Base-RFF"] <- "Base"
  out[grepl("^Learned-", x) | grepl("^Learned-prob-", x)] <- "Learned"

  out[x == "Energy-prob-RFF"] <- "Energy sampler"
  out[x == "Circular-Energy-prob"] <- "Energy sampler"
  out[x == "Affine-MovingChart-Wavelet-energy-prob"] <- "Wavelet sampler"

  out[x == "Circular-Haar-RFF"] <- "Haar"

  out[x == "Align-energy-median-RFF"] <- "Energy median"
  out[x == "Align-energy-circular-median"] <- "Energy median"
  out[x == "Align-xcorr-RFF"] <- "Xcorr"
  out[x == "Align-xcorr-circular"] <- "Xcorr"
  out[x == "Align-xcorr-affine"] <- "Xcorr"
  out[x == "Align-wavelet-max-affine"] <- "Wavelet max"
  out
}

dat$Method <- short_method(dat$method)
method_levels <- c(
  "Base",
  "Energy median", "Xcorr", "Wavelet max",
  "Energy sampler", "Wavelet sampler", "Haar",
  "Learned"
)
dat$Method <- factor(dat$Method, levels = method_levels)

# A method is S-dependent if the clean result file contains at least one S > 0.
method_s_type <- dat |>
  group_by(Group, method) |>
  summarise(stochastic = any(!is.na(S) & S > 0), .groups = "drop")

dat <- dat |>
  left_join(method_s_type, by = c("Group", "method"))

# Main paper comparison: S=32 for S-dependent methods, native S=0 otherwise.
main <- dat |>
  filter(
    (stochastic & S == 32) |
      (!stochastic & (is.na(S) | S == 0))
  )

if (!nrow(main)) stop("No rows available for the main S=32 comparison.")

# ---------- Plot theme and semantic palette ----------
method_colours <- c(
  "Base" = "#7A7A7A",
  "Energy median" = "#D73027",
  "Xcorr" = "#B2182B",
  "Wavelet max" = "#F46D43",
  "Energy sampler" = "#3182BD",
  "Wavelet sampler" = "#08519C",
  "Haar" = "#9ECAE1",
  "Learned" = "#5B4BBA"
)

method_linetypes <- c(
  "Base" = "dotted",
  "Energy median" = "solid",
  "Xcorr" = "solid",
  "Wavelet max" = "solid",
  "Energy sampler" = "solid",
  "Wavelet sampler" = "solid",
  "Haar" = "solid",
  "Learned" = "solid"
)

method_shapes <- c(
  "Base" = 16,
  "Energy median" = 17,
  "Xcorr" = 15,
  "Wavelet max" = 8,
  "Energy sampler" = 1,
  "Wavelet sampler" = 2,
  "Haar" = 5,
  "Learned" = 19
)

delta_colours <- c(
  "0.4" = "#D7D4F0",
  "0.6" = "#AAA3DE",
  "0.8" = "#786BCB",
  "1"   = "#4B359F"
)

theme_final <- theme_bw(base_size = 11) +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey95"),
    plot.title.position = "plot"
  )

save_plot <- function(p, filename, width, height) {
  ggsave(
    filename = file.path(fig_dir, filename),
    plot = p,
    width = width,
    height = height,
    units = "in",
    device = "pdf"
  )
}

# ---------- Power curves, one figure per model ----------
for (m in model_levels) {
  pdat <- main |> filter(Model == m)
  if (!nrow(pdat)) next

  p <- ggplot(
    pdat,
    aes(
      x = delta,
      y = rejection_rate,
      group = Method,
      colour = Method,
      linetype = Method,
      shape = Method
    )
  ) +
    geom_hline(yintercept = 0.05, linewidth = 0.35, linetype = 3) +
    geom_line(linewidth = 0.75) +
    geom_point(size = 2.2, stroke = 0.8) +
    scale_colour_manual(values = method_colours, drop = FALSE) +
    scale_linetype_manual(values = method_linetypes, drop = FALSE) +
    scale_shape_manual(values = method_shapes, drop = FALSE) +
    facet_wrap(~Group, nrow = 1, scales = "fixed") +
    scale_x_continuous(breaks = seq(0, 1, by = 0.2)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
    labs(
      x = expression(delta),
      y = "Rejection probability",
      colour = NULL,
      linetype = NULL,
      shape = NULL
    ) +
    theme_final

  save_plot(p, paste0("fig_power_", m, ".pdf"), 10.5, 4.0)
}

# ---------- Empirical level ----------
level_dat <- main |> filter(delta == 0)

p_level <- ggplot(
  level_dat,
  aes(
    x = Model,
    y = rejection_rate,
    group = Method,
    colour = Method,
    linetype = Method,
    shape = Method
  )
) +
  geom_hline(yintercept = 0.05, linewidth = 0.35, linetype = 3) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2.2, stroke = 0.8) +
  scale_colour_manual(values = method_colours, drop = FALSE) +
  scale_linetype_manual(values = method_linetypes, drop = FALSE) +
  scale_shape_manual(values = method_shapes, drop = FALSE) +
  facet_wrap(~Group, nrow = 1) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
  labs(
    x = NULL,
    y = "Empirical rejection probability at delta = 0",
    colour = NULL,
    linetype = NULL,
    shape = NULL
  ) +
  theme_final

save_plot(p_level, "fig_level.pdf", 10.5, 4.0)

# ---------- Influence of S: learned sampler over several deltas ----------
s_learned <- dat |>
  filter(
    Method == "Learned",
    stochastic,
    delta %in% c(0.4, 0.6, 0.8, 1.0)
  ) |>
  mutate(delta_f = factor(delta, levels = c(0.4, 0.6, 0.8, 1.0)))

p_s_learned <- ggplot(
  s_learned,
  aes(
    x = S,
    y = rejection_rate,
    group = delta_f,
    colour = delta_f,
    shape = delta_f
  )
) +
  geom_line(linewidth = 0.75) +
  geom_point(size = 2.0, stroke = 0.8) +
  scale_colour_manual(values = delta_colours, drop = FALSE) +
  scale_shape_manual(values = c("0.4" = 16, "0.6" = 17, "0.8" = 15, "1" = 18)) +
  facet_grid(Model ~ Group) +
  scale_x_continuous(breaks = c(4, 8, 16, 32)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.25)) +
  labs(
    x = "Number of sampled transformations S",
    y = "Rejection probability",
    colour = expression(delta),
    shape = expression(delta)
  ) +
  theme_final +
  theme(legend.position = "bottom")

save_plot(p_s_learned, "fig_S_learned.pdf", 9.8, 8.5)

# ---------- Influence of S: all stochastic methods at delta = 0.8 ----------
s_prob <- dat |>
  filter(stochastic, delta == 0.8)

p_s_prob <- ggplot(
  s_prob,
  aes(
    x = S,
    y = rejection_rate,
    group = Method,
    colour = Method,
    linetype = Method,
    shape = Method
  )
) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.9, stroke = 0.8) +
  scale_colour_manual(values = method_colours, drop = FALSE) +
  scale_linetype_manual(values = method_linetypes, drop = FALSE) +
  scale_shape_manual(values = method_shapes, drop = FALSE) +
  facet_grid(Model ~ Group) +
  scale_x_continuous(breaks = c(4, 8, 16, 32)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.25)) +
  labs(
    x = "Number of sampled transformations S",
    y = "Rejection probability at delta = 0.8",
    colour = NULL,
    linetype = NULL,
    shape = NULL
  ) +
  theme_final

save_plot(p_s_prob, "fig_S_probabilistic.pdf", 9.8, 8.5)

# ---------- Runtime helpers ----------
# In the clean project, every summary row contains a self-contained timing:
#   total = train + eval.
# Learned rows repeat the same one-time train cost for each S, which is exactly
# what is needed for a cold end-to-end comparison at a chosen S.
train_col <- if (runtime_stat == "median") "median_train_seconds" else "mean_train_seconds"
eval_col  <- if (runtime_stat == "median") "median_eval_seconds"  else "mean_eval_seconds"
total_col <- if (runtime_stat == "median") "median_total_seconds" else "mean_total_seconds"

main_runtime <- main |>
  transmute(
    Group,
    Model,
    delta,
    Method,
    train_seconds = .data[[train_col]],
    eval_seconds = .data[[eval_col]],
    total_seconds = .data[[total_col]]
  )

# Average across model/delta cells after taking the requested per-cell timing
# statistic. This preserves the old manuscript-level aggregation while making
# each cell robust when RUNTIME_STAT=median.
runtime_end_to_end <- main_runtime |>
  group_by(Group, Method) |>
  summarise(
    train_seconds = mean(train_seconds, na.rm = TRUE),
    eval_seconds = mean(eval_seconds, na.rm = TRUE),
    total_seconds = mean(total_seconds, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    train_seconds = ifelse(is.finite(train_seconds), train_seconds, 0),
    eval_seconds = ifelse(is.finite(eval_seconds), eval_seconds, 0),
    total_seconds = ifelse(is.finite(total_seconds), total_seconds, train_seconds + eval_seconds)
  ) |>
  filter(is.finite(total_seconds), total_seconds > 0)

runtime_test_plot <- runtime_end_to_end |>
  transmute(
    Group, Method,
    Component = "Evaluation",
    seconds = eval_seconds,
    FillKey = ifelse(Method == "Learned", "Learned evaluation", as.character(Method))
  )

runtime_train_plot <- runtime_end_to_end |>
  filter(Method == "Learned", train_seconds > 0) |>
  transmute(
    Group, Method,
    Component = "Training",
    seconds = train_seconds,
    FillKey = "Learned training"
  )

runtime_plot <- bind_rows(runtime_test_plot, runtime_train_plot)

runtime_fill_colours <- c(
  method_colours[names(method_colours) != "Learned"],
  "Learned training" = "#B9B3E5",
  "Learned evaluation" = method_colours[["Learned"]]
)

p_runtime <- ggplot(
  runtime_plot,
  aes(x = Method, y = seconds, fill = FillKey)
) +
  geom_col(width = 0.72) +
  scale_fill_manual(
    values = runtime_fill_colours,
    breaks = c("Learned training", "Learned evaluation"),
    labels = c("Learned: training", "Learned: evaluation"),
    name = NULL
  ) +
  facet_wrap(~Group, nrow = 1, scales = "free_y") +
  labs(
    x = NULL,
    y = paste0(if (runtime_stat == "median") "Median" else "Mean", " end-to-end time (seconds)")
  ) +
  theme_final +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

save_plot(p_runtime, "fig_runtime_methods.pdf", 11.0, 4.8)

# ---------- Evaluation time as a function of S ----------
runtime_s <- dat |>
  filter(stochastic) |>
  transmute(
    Group, Model, delta, Method, S,
    seconds = .data[[eval_col]]
  ) |>
  group_by(Group, Method, S) |>
  summarise(
    seconds = mean(seconds, na.rm = TRUE),
    .groups = "drop"
  ) |>
  filter(is.finite(seconds), seconds > 0)

p_runtime_s <- ggplot(
  runtime_s,
  aes(
    x = S,
    y = seconds,
    group = Method,
    colour = Method,
    linetype = Method,
    shape = Method
  )
) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2.0, stroke = 0.8) +
  scale_colour_manual(values = method_colours, drop = FALSE) +
  scale_linetype_manual(values = method_linetypes, drop = FALSE) +
  scale_shape_manual(values = method_shapes, drop = FALSE) +
  facet_wrap(~Group, nrow = 1, scales = "free_y") +
  scale_x_continuous(breaks = c(4, 8, 16, 32)) +
  labs(
    x = "Number of sampled transformations S",
    y = paste0(if (runtime_stat == "median") "Median" else "Mean", " evaluation time (seconds)"),
    colour = NULL,
    linetype = NULL,
    shape = NULL
  ) +
  theme_final

save_plot(p_runtime_s, "fig_runtime_S.pdf", 10.5, 4.0)

# ---------- Numeric summaries used in the manuscript ----------
learned_level <- main |>
  filter(Method == "Learned", delta == 0) |>
  select(Group, Model, rejection_rate, nrep)

learned_power_d1 <- main |>
  filter(Method == "Learned", delta == 1) |>
  select(Group, Model, rejection_rate, nrep)

runtime_export <- runtime_end_to_end |>
  arrange(Group, total_seconds)

write.csv(
  learned_level,
  file.path(fig_dir, "table_learned_level.csv"),
  row.names = FALSE
)
write.csv(
  learned_power_d1,
  file.path(fig_dir, "table_learned_power_delta1.csv"),
  row.names = FALSE
)
write.csv(
  runtime_export,
  file.path(fig_dir, "table_runtime_end_to_end.csv"),
  row.names = FALSE
)

# A small manifest makes it explicit which files/statistic generated the figures.
manifest <- data.frame(
  item = c("affine_file", "translation_file", "circular_file", "runtime_stat"),
  value = c(
    normalizePath(affine_file, mustWork = FALSE),
    normalizePath(translation_file, mustWork = FALSE),
    normalizePath(circle_file, mustWork = FALSE),
    runtime_stat
  ),
  stringsAsFactors = FALSE
)
write.csv(manifest, file.path(fig_dir, "figure_manifest.csv"), row.names = FALSE)

message("Figures and summary CSV files written to: ", normalizePath(fig_dir, mustWork = FALSE))
message("Runtime statistic: ", runtime_stat)
