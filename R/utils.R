# Shared numerical utilities ----------------------------------------------------

require_torch <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) stop("Package 'torch' is required")
  if (!isTRUE(torch::torch_is_installed())) stop("LibTorch is not installed; run torch::install_torch()")
}

safe_seed <- function(x, modulus = 2000000000) {
  y <- as.numeric(x) %% as.numeric(modulus)
  if (!is.finite(y)) stop("Non-finite seed")
  as.integer(y + 1)
}
mix_seed <- function(base, mult = 1, add = 0, modulus = 2000000000) {
  safe_seed(as.numeric(base) * as.numeric(mult) + as.numeric(add), modulus)
}

l2_sq_rows <- function(X, Y = NULL, dt = 1) {
  X <- as.matrix(X)
  if (is.null(Y)) Y <- X else Y <- as.matrix(Y)
  xx <- rowSums(X^2); yy <- rowSums(Y^2)
  D2 <- dt * (outer(xx, yy, "+") - 2 * X %*% t(Y))
  D2[!is.finite(D2)] <- NA_real_
  D2[D2 < 0 & D2 > -1e-10] <- 0
  D2[D2 < 0] <- 0
  dim(D2) <- c(nrow(X), nrow(Y))
  D2
}
median_bandwidth <- function(Z, dt = 1) {
  D2 <- l2_sq_rows(Z, dt = dt)
  z <- sqrt(D2[upper.tri(D2)])
  z <- z[is.finite(z) & z > 1e-12]
  if (!length(z)) 1 else as.numeric(median(z))
}

make_rff_map <- function(p, sigma, dt, D, seed) {
  if (!is.finite(sigma) || sigma <= 0) stop("RFF requires sigma > 0")
  if (D %% 2L != 0L) stop("RFF dimension must be even")
  M <- D %/% 2L
  had <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had) old <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  on.exit({
    if (had) assign(".Random.seed", old, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv)
  }, add = TRUE)
  set.seed(safe_seed(seed))
  omega <- matrix(rnorm(p * M, sd = sqrt(dt) / sigma), nrow = p, ncol = M)
  list(omega = omega, M = M, D = D, scale = 1 / sqrt(M), sigma = sigma, dt = dt)
}
rff_features <- function(X, rff, chunk_size = 2048L) {
  X <- as.matrix(X); N <- nrow(X)
  if (ncol(X) != nrow(rff$omega)) stop("RFF input width mismatch")
  out <- matrix(0, N, rff$D)
  if (!N) return(out)
  for (a in seq.int(1L, N, by = max(1L, as.integer(chunk_size)))) {
    b <- min(N, a + chunk_size - 1L); ii <- a:b
    A <- X[ii, , drop = FALSE] %*% rff$omega
    out[ii, seq_len(rff$M)] <- cos(A) * rff$scale
    out[ii, rff$M + seq_len(rff$M)] <- sin(A) * rff$scale
  }
  out
}

mmd_u_features <- function(F, ix) {
  F <- as.matrix(F); N <- nrow(F); ix <- as.integer(ix); iy <- setdiff(seq_len(N), ix)
  nx <- length(ix); ny <- length(iy)
  if (nx < 2L || ny < 2L) stop("MMD needs at least two observations per group")
  FX <- F[ix, , drop = FALSE]; FY <- F[iy, , drop = FALSE]
  sx <- colSums(FX); sy <- colSums(FY)
  xx <- (sum(sx^2) - sum(FX^2)) / (nx * (nx - 1))
  yy <- (sum(sy^2) - sum(FY^2)) / (ny * (ny - 1))
  xy <- 2 * sum(sx * sy) / (nx * ny)
  xx + yy - xy
}
make_perm_plan <- function(N, nx, B, seed) {
  had <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had) old <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  on.exit({
    if (had) assign(".Random.seed", old, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv)
  }, add = TRUE)
  set.seed(safe_seed(seed))
  out <- matrix(0L, B, nx)
  for (b in seq_len(B)) out[b, ] <- sort(sample.int(N, nx, replace = FALSE))
  out
}
perm_test_features <- function(F, nx, alpha, plan) {
  stat <- mmd_u_features(F, seq_len(nx))
  perm <- apply(plan, 1L, function(ix) mmd_u_features(F, ix))
  pvalue <- (1 + sum(perm >= stat - 1e-15)) / (nrow(plan) + 1)
  list(stat = stat, pvalue = pvalue, reject = as.integer(pvalue <= alpha))
}

cnn_model_parameters <- function(model) {
  p <- model$parameters
  if (is.function(p)) p <- p()
  if (inherits(p, "torch_tensor")) p <- list(p)
  if (!is.list(p)) p <- as.list(p)
  p
}
torch_svdvals_compat <- function(x) {
  ns <- asNamespace("torch")
  if (exists("linalg_svdvals", envir = ns, inherits = FALSE)) return(get("linalg_svdvals", envir = ns)(x))
  if (exists("torch_svd", envir = ns, inherits = FALSE)) return(get("torch_svd", envir = ns)(x, some = TRUE, compute_uv = TRUE)[[2]])
  stop("No SVD implementation found in R torch")
}
flat_spectral_norm <- function(weight) {
  w <- weight$detach(); nr <- as.integer(w$size(1)); sv <- torch_svdvals_compat(w$reshape(c(nr, -1L)))
  if (as.integer(sv$numel()) < 1L) 0 else as.numeric(sv[1]$item())
}
project_spectral_norm_layers_ <- function(layers, caps) {
  torch::with_no_grad({
    for (j in seq_along(layers)) {
      w <- layers[[j]]$weight
      s <- flat_spectral_norm(w)
      if (is.finite(s) && s > caps[j] && s > 0) w$mul_(caps[j] / s)
    }
  })
  invisible(NULL)
}

balanced_batches <- function(n, batch_size) {
  b <- min(as.integer(batch_size), n)
  if (b >= n) return(list(c(seq_len(n), n + seq_len(n))))
  nb <- max(1L, floor(n / b)); px <- sample.int(n); py <- sample.int(n)
  lapply(seq_len(nb), function(j) {
    rr <- ((j - 1L) * b + 1L):(j * b)
    c(px[rr], n + py[rr])
  })
}

mmd_power_objective_torch <- function(Fbar, n, lambda) {
  FX <- Fbar$narrow(1, 1L, n); FY <- Fbar$narrow(1, n + 1L, n)
  dxy <- FX - FY
  svec <- dxy$sum(dim = 1)
  norm_s2 <- svec$pow(2)$sum(); sum_d2 <- dxy$pow(2)$sum()
  mmd2 <- (norm_s2 - sum_d2) / (n * (n - 1))
  proj <- (dxy * svec$unsqueeze(1))$sum(dim = 2)
  sig2 <- ((4 / n^3) * proj$pow(2)$sum() - (4 / n^4) * norm_s2$pow(2))$clamp(min = 0)
  J <- mmd2 / (sig2 + lambda)$sqrt()
  list(loss = -J, J = J, mmd2 = mmd2, sig2 = sig2)
}

write_results <- function(rows, cfg, outdir, config_extra = list()) {
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  raw <- do.call(rbind, rows)
  keys <- interaction(raw$scenario, raw$delta, raw$method, raw$S, drop = TRUE, lex.order = TRUE)
  spl <- split(raw, keys)
  summary <- do.call(rbind, lapply(spl, function(d) {
    data.frame(
      scenario = d$scenario[1], delta = d$delta[1], method = d$method[1], S = d$S[1],
      rejection_rate = mean(d$reject), mean_pvalue = mean(d$pvalue),
      mean_train_seconds = mean(d$train_seconds), median_train_seconds = median(d$train_seconds),
      mean_eval_seconds = mean(d$eval_seconds), median_eval_seconds = median(d$eval_seconds),
      mean_total_seconds = mean(d$total_seconds), median_total_seconds = median(d$total_seconds),
      q05_total_seconds = as.numeric(stats::quantile(d$total_seconds, .05, names = FALSE)),
      q95_total_seconds = as.numeric(stats::quantile(d$total_seconds, .95, names = FALSE)),
      nrep = nrow(d), stringsAsFactors = FALSE
    )
  }))
  rownames(summary) <- NULL
  write.csv(raw, file.path(outdir, "raw_results.csv"), row.names = FALSE)
  write.csv(summary, file.path(outdir, "summary_results.csv"), row.names = FALSE)
  saveRDS(list(config = c(cfg, config_extra), raw = raw, summary = summary), file.path(outdir, "results_and_config.rds"))
  invisible(list(raw = raw, summary = summary))
}
