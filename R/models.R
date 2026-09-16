# Four final intrinsic models: M1, M2, M3, M4 ----------------------------------

.gauss <- function(u, mu, sd) exp(-((u - mu)^2) / (2 * sd^2))
.ref_grid <- seq(-12, 12, length.out = 30001L)
.ref_dt <- .ref_grid[2] - .ref_grid[1]
normalize_fun_l2 <- function(fun) {
  z <- sqrt(.ref_dt * sum(fun(.ref_grid)^2))
  function(u) fun(u) / z
}

make_aperiodic_templates <- function(cfg) {
  m1_1 <- normalize_fun_l2(function(u) .gauss(u, 0, 0.5))
  m1_2 <- normalize_fun_l2(function(u) .gauss(u, -1, 0.4) + .gauss(u, 1, 0.4))
  m1_3 <- normalize_fun_l2(function(u) .gauss(u, -1.5, 0.3) + 0.8 * .gauss(u, 0, 0.3) + .gauss(u, 1.5, 0.3))
  m2_1 <- normalize_fun_l2(function(u) .gauss(u, -1.5, cfg$m2_g1_sd))
  m2_2 <- normalize_fun_l2(function(u) .gauss(u,  1.5, cfg$m2_g2_sd))
  list(m1 = list(m1_1, m1_2, m1_3), m2 = list(m2_1, m2_2))
}

M3_POS <- c(-2, -1, -0.2, 1, 2)
M3_SD <- rep(0.2, 5)
M3_BASE_AMP <- c(1, 0.3, 1.5, 0.3, 1)

sample_m1_class <- function(n, group, delta, cfg) {
  # This is the normalized version of the historical c(.4,.3,.2) weights.
  # Writing it explicitly avoids relying on sample()'s implicit normalization.
  wx <- c(4, 3, 2) / 9
  w <- wx
  if (group == "Y") {
    move <- cfg$m1_mix_shift * delta
    w <- c(wx[1] - move, wx[2], wx[3] + move)
  }
  if (any(w < 0)) stop("M1 mixture probabilities became negative")
  sample.int(3L, n, replace = TRUE, prob = w)
}

draw_m2_coefficients <- function(n, group, delta, cfg) {
  rho <- cfg$m2_rho_max * delta * if (group == "X") 1 else -1
  rho <- max(-0.999, min(0.999, rho))
  e1 <- rnorm(n); e2 <- rnorm(n)
  cbind(
    1 + cfg$m2_coef_sd * e1,
    1 + cfg$m2_coef_sd * (rho * e1 + sqrt(1 - rho^2) * e2)
  )
}

draw_m3_amplitudes <- function(n, group, delta, cfg) {
  L <- matrix(
    rlnorm(n * length(M3_BASE_AMP), meanlog = -0.5 * cfg$m3_peak_sdlog^2, sdlog = cfg$m3_peak_sdlog),
    nrow = n, ncol = length(M3_BASE_AMP)
  )
  A <- sweep(L, 2, M3_BASE_AMP, "*")
  if (group == "Y") A[, 4] <- A[, 4] * (1 + cfg$m3_peak_effect * delta)
  A
}

m4_doublet_norm <- function(halfsep, sd) {
  sqrt(2 * sqrt(pi) * sd * (1 + exp(-(halfsep^2) / sd^2)))
}
m4_doublet <- function(u, halfsep, sd) {
  (.gauss(u, -halfsep, sd) + .gauss(u, halfsep, sd)) / m4_doublet_norm(halfsep, sd)
}

draw_m4_positions_line <- function(n, center, cfg) {
  K <- cfg$m4_nuisance_k; out <- matrix(NA_real_, n, K)
  for (i in seq_len(n)) {
    ok <- FALSE
    for (attempt in seq_len(5000L)) {
      z <- sort(runif(K, cfg$m4_nuisance_pos_min, cfg$m4_nuisance_pos_max))
      sep_ok <- K <= 1L || min(diff(z)) >= cfg$m4_nuisance_min_sep
      guard_ok <- all(abs(z - center[i]) >= cfg$m4_nuisance_guard)
      if (sep_ok && guard_ok) { out[i, ] <- z; ok <- TRUE; break }
    }
    if (!ok) stop("Could not place M4 nuisance peaks")
  }
  out
}

draw_m4_latents_line <- function(n, group, delta, cfg) {
  d <- if (group == "X") cfg$m4_halfsep0 - cfg$m4_halfsep_effect * delta else cfg$m4_halfsep0 + cfg$m4_halfsep_effect * delta
  if (d <= 0) stop("M4 half-separation became non-positive")
  center <- rnorm(n, 0, cfg$m4_motif_center_sd)
  motif_amp <- rlnorm(n, log(cfg$m4_motif_amp) - 0.5 * cfg$m4_motif_amp_sdlog^2, cfg$m4_motif_amp_sdlog)
  pos <- draw_m4_positions_line(n, center, cfg)
  amp <- matrix(rlnorm(n * cfg$m4_nuisance_k,
    log(cfg$m4_nuisance_amp) - 0.5 * cfg$m4_nuisance_amp_sdlog^2,
    cfg$m4_nuisance_amp_sdlog), nrow = n)
  sd <- matrix(runif(n * cfg$m4_nuisance_k, cfg$m4_nuisance_sd_min, cfg$m4_nuisance_sd_max), nrow = n)
  list(d = rep(d, n), center = center, motif_amp = motif_amp, pos = pos, amp = amp, sd = sd)
}

add_final_noise <- function(z, cfg) {
  z * rnorm(length(z), mean = cfg$noise_mean_mult, sd = cfg$noise_sd_mult)
}

generate_intrinsic_line <- function(t, scenario, delta, group, cfg, templates = NULL) {
  if (is.null(templates)) templates <- make_aperiodic_templates(cfg)
  if (scenario == SCENARIOS[["M1"]]) {
    cls <- sample_m1_class(1L, group, delta, cfg)
    gamma <- rlnorm(1L, -0.5 * cfg$sigma_gamma_common^2, cfg$sigma_gamma_common)
    z <- gamma * templates$m1[[cls]](t)
  } else if (scenario == SCENARIOS[["M2"]]) {
    ab <- draw_m2_coefficients(1L, group, delta, cfg)
    z <- ab[1, 1] * templates$m2[[1]](t) + ab[1, 2] * templates$m2[[2]](t)
  } else if (scenario == SCENARIOS[["M3"]]) {
    A <- draw_m3_amplitudes(1L, group, delta, cfg)
    z <- numeric(length(t))
    for (k in seq_along(M3_POS)) z <- z + A[1, k] * .gauss(t, M3_POS[k], M3_SD[k])
    gamma <- rlnorm(1L, -0.5 * cfg$sigma_gamma_common^2, cfg$sigma_gamma_common)
    z <- gamma * z
  } else if (scenario == SCENARIOS[["M4"]]) {
    q <- draw_m4_latents_line(1L, group, delta, cfg)
    z <- numeric(length(t))
    for (k in seq_len(cfg$m4_nuisance_k)) z <- z + q$amp[1, k] * .gauss(t, q$pos[1, k], q$sd[1, k])
    z <- z + q$motif_amp[1] * m4_doublet(t - q$center[1], q$d[1], cfg$m4_motif_peak_sd)
  } else stop("Unknown final scenario: ", scenario)
  add_final_noise(z, cfg)
}

# Circular versions -------------------------------------------------------------
wrap_angle <- function(x) ((x + pi) %% (2 * pi)) - pi
circ_dist <- function(a, b) abs(wrap_angle(a - b))
periodic_gauss <- function(u, mu, sd) exp(-(wrap_angle(u - mu)^2) / (2 * sd^2))
normalize_grid_l2 <- function(x, dt) { z <- sqrt(dt * sum(x^2)); if (z <= 1e-15) x else x / z }

make_circular_templates <- function(cfg) {
  u <- cfg$u
  list(
    m1 = rbind(
      normalize_grid_l2(periodic_gauss(u, 0, 0.5), cfg$dt),
      normalize_grid_l2(periodic_gauss(u, -1, 0.4) + periodic_gauss(u, 1, 0.4), cfg$dt),
      normalize_grid_l2(periodic_gauss(u, -1.5, 0.3) + 0.8 * periodic_gauss(u, 0, 0.3) + periodic_gauss(u, 1.5, 0.3), cfg$dt)
    ),
    m2 = rbind(
      normalize_grid_l2(periodic_gauss(u, -1.5, cfg$m2_g1_sd), cfg$dt),
      normalize_grid_l2(periodic_gauss(u,  1.5, cfg$m2_g2_sd), cfg$dt)
    )
  )
}

circ_shift <- function(x, k) {
  p <- length(x); k <- as.integer(round(k)) %% p
  x[((0:(p - 1L) - k) %% p) + 1L]
}
circ_shift_matrix <- function(Z, k) {
  p <- ncol(Z); k <- as.integer(round(k)) %% p
  Z[, ((0:(p - 1L) - k) %% p) + 1L, drop = FALSE]
}

draw_m4_positions_circle <- function(K, center, cfg) {
  for (attempt in seq_len(5000L)) {
    z <- runif(K, -pi, pi)
    guard_ok <- all(circ_dist(z, center) >= cfg$m4_nuisance_guard)
    sep_ok <- TRUE
    if (K > 1L) {
      D <- outer(z, z, function(a, b) circ_dist(a, b)); diag(D) <- Inf
      sep_ok <- min(D) >= cfg$m4_nuisance_min_sep
    }
    if (guard_ok && sep_ok) return(z)
  }
  stop("Could not place circular M4 nuisance peaks")
}

generate_intrinsic_circle <- function(scenario, delta, group, cfg, templates) {
  if (scenario == SCENARIOS[["M1"]]) {
    cls <- sample_m1_class(1L, group, delta, cfg)
    gamma <- rlnorm(1L, -0.5 * cfg$sigma_gamma_common^2, cfg$sigma_gamma_common)
    z <- gamma * templates$m1[cls, ]
  } else if (scenario == SCENARIOS[["M2"]]) {
    ab <- draw_m2_coefficients(1L, group, delta, cfg)
    z <- ab[1, 1] * templates$m2[1, ] + ab[1, 2] * templates$m2[2, ]
  } else if (scenario == SCENARIOS[["M3"]]) {
    A <- draw_m3_amplitudes(1L, group, delta, cfg)
    z <- numeric(cfg$p)
    for (k in seq_along(M3_POS)) z <- z + A[1, k] * periodic_gauss(cfg$u, M3_POS[k], M3_SD[k])
    z <- rlnorm(1L, -0.5 * cfg$sigma_gamma_common^2, cfg$sigma_gamma_common) * z
  } else if (scenario == SCENARIOS[["M4"]]) {
    d <- if (group == "X") cfg$m4_halfsep0 - cfg$m4_halfsep_effect * delta else cfg$m4_halfsep0 + cfg$m4_halfsep_effect * delta
    center <- wrap_angle(rnorm(1L, 0, cfg$m4_motif_center_sd))
    motif_amp <- rlnorm(1L, log(cfg$m4_motif_amp) - 0.5 * cfg$m4_motif_amp_sdlog^2, cfg$m4_motif_amp_sdlog)
    pos <- draw_m4_positions_circle(cfg$m4_nuisance_k, center, cfg)
    z <- numeric(cfg$p)
    amps <- rlnorm(length(pos), log(cfg$m4_nuisance_amp) - 0.5 * cfg$m4_nuisance_amp_sdlog^2, cfg$m4_nuisance_amp_sdlog)
    sds <- runif(length(pos), cfg$m4_nuisance_sd_min, cfg$m4_nuisance_sd_max)
    for (k in seq_along(pos)) z <- z + amps[k] * periodic_gauss(cfg$u, pos[k], sds[k])
    motif0 <- normalize_grid_l2(periodic_gauss(cfg$u, -d, cfg$m4_motif_peak_sd) + periodic_gauss(cfg$u, d, cfg$m4_motif_peak_sd), cfg$dt)
    z <- z + motif_amp * circ_shift(motif0, round(center / cfg$dt))
  } else stop("Unknown final scenario: ", scenario)
  add_final_noise(z, cfg)
}
