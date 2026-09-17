# Methods for G = S^1 (numerical group C_p) ------------------------------------
# Learned sampler: deterministic moving chart followed by a 1D CNN on residual
# circular translations. Fixed Haar and energy competitors remain unchanged.

make_circular_cfg <- function(shared) {
  cfg <- shared
  cfg$p <- env_int("P_GRID", 128L)
  cfg$period <- 2 * pi
  cfg$dt <- cfg$period / cfg$p
  cfg$t <- (0:(cfg$p - 1L)) * cfg$dt
  cfg$u <- ((0:(cfg$p - 1L) + cfg$p %/% 2L) %% cfg$p - cfg$p %/% 2L) * cfg$dt
  cfg$phase_gap <- cfg$shift_gap
  cfg$sigma_phase <- cfg$sigma_shift
  cfg$cnn_channels <- env_int("CNN_CHANNELS", 8L)
  cfg$cnn_k <- c(11L, 7L, 5L)
  cfg$cnn_dilation <- c(1L, 4L, 8L)
  cfg$cnn_energy_eps_rel <- 1e-6
  cfg
}

# Moving chart -----------------------------------------------------------------

circular_anchor_index <- function(x, cfg) {
  w <- x^2
  if (!is.finite(sum(w)) || sum(w) <= 1e-15) return(0L)
  idx <- 0:(cfg$p - 1L)
  obj <- numeric(cfg$p)
  for (c in idx) {
    d <- abs(idx - c)
    d <- pmin(d, cfg$p - d)
    obj[c + 1L] <- sum(w * d)
  }
  which.min(obj) - 1L
}

canonicalize_circular <- function(Z, cfg) {
  Z <- as.matrix(Z)
  t(vapply(seq_len(nrow(Z)), function(i) {
    k0 <- circular_anchor_index(Z[i, ], cfg)
    circ_shift(Z[i, ], -k0)
  }, numeric(cfg$p)))
}

circular_candidate_rep <- function(xc, g, cfg) {
  circ_shift(xc, -as.integer(g))
}

circular_energy_probs <- function(Zc, cfg) {
  E <- as.matrix(Zc)^2
  E <- E + cfg$cnn_energy_eps_rel * rowMeans(E) + 1e-20
  E / rowSums(E)
}

# Learned CNN ------------------------------------------------------------------

make_circular_cnn <- function(cfg) {
  d <- cfg$cnn_dilation
  k <- cfg$cnn_k
  Net <- torch::nn_module(
    classname = "CircularMovingChartCNNFinal",
    initialize = function() {
      self$conv1 <- torch::nn_conv1d(1L, cfg$cnn_channels, kernel_size = k[1], dilation = d[1], padding = 0, bias = FALSE)
      self$conv2 <- torch::nn_conv1d(cfg$cnn_channels, cfg$cnn_channels, kernel_size = k[2], dilation = d[2], padding = 0, bias = FALSE)
      self$conv3 <- torch::nn_conv1d(cfg$cnn_channels, 1L, kernel_size = k[3], dilation = d[3], padding = 0, bias = FALSE)
    },
    circular_pad = function(x, pad) {
      pad <- as.integer(pad)
      if (pad <= 0L) return(x)
      P <- as.integer(x$size(3))
      torch::torch_cat(
        list(x$narrow(3, P - pad + 1L, pad), x, x$narrow(3, 1L, pad)),
        dim = 3
      )
    },
    forward = function(x) {
      p1 <- d[1] * (k[1] - 1L) %/% 2L
      p2 <- d[2] * (k[2] - 1L) %/% 2L
      p3 <- d[3] * (k[3] - 1L) %/% 2L
      z <- self$conv1(self$circular_pad(x, p1))$tanh()
      z <- self$conv2(self$circular_pad(z, p2))$tanh()
      self$conv3(self$circular_pad(z, p3))
    }
  )
  model <- Net()
  torch::nn_init_zeros_(model$conv3$weight)
  model
}

circular_logits_canonical <- function(model, Zc, cfg) {
  Zc <- as.matrix(Zc)
  rms <- sqrt(rowMeans(Zc^2))
  rms[!is.finite(rms) | rms < 1e-12] <- 1
  X <- torch::torch_tensor(Zc / rms, dtype = torch::torch_float32())$unsqueeze(2)
  residual <- model(X)$squeeze(2)
  if (as.integer(residual$size(2)) != cfg$p) stop("Circular CNN output width mismatch")
  base <- circular_energy_probs(Zc, cfg)
  torch::torch_tensor(log(base), dtype = torch::torch_float32()) + residual
}

circular_probs_canonical <- function(model, Zc, cfg) {
  model$eval()
  P <- torch::with_no_grad(circular_logits_canonical(model, Zc, cfg)$softmax(dim = 2))
  matrix(as.numeric(as.array(P)), nrow = nrow(Zc), ncol = cfg$p, byrow = FALSE)
}

circular_probs <- function(model, Z, cfg) {
  circular_probs_canonical(model, canonicalize_circular(Z, cfg), cfg)
}

build_circular_candidate_cache <- function(Xtr, Ytr, cfg, rff) {
  Z <- rbind(Xtr, Ytr)
  Zc <- canonicalize_circular(Z, cfg)
  N <- nrow(Zc)
  P <- cfg$p
  D <- rff$D

  reps <- array(0, dim = c(N, P, P))
  for (g in 0:(P - 1L)) {
    reps[, g + 1L, ] <- circ_shift_matrix(Zc, -g)
  }
  reps_flat <- matrix(reps, nrow = N * P, ncol = P)
  Fflat <- rff_features(reps_flat, rff, cfg$rff_chunk_size)
  A <- array(Fflat, dim = c(N, P, D))
  rm(reps, reps_flat, Fflat)
  list(Z = Z, Zc = Zc, candidate_rff = A)
}

circular_training_tensors <- function(cache, cfg) {
  Zc <- as.matrix(cache$Zc)
  rms <- sqrt(rowMeans(Zc^2))
  rms[!is.finite(rms) | rms < 1e-12] <- 1
  X <- torch::torch_tensor(Zc / rms, dtype = torch::torch_float32())$unsqueeze(2)
  B <- torch::torch_tensor(log(circular_energy_probs(Zc, cfg)), dtype = torch::torch_float32())
  F <- torch::torch_tensor(cache$candidate_rff, dtype = torch::torch_float32())
  list(input = X, base_log = B, candidate_rff = F)
}

circular_logits_precomputed <- function(model, input_tensor, base_log_tensor, cfg) {
  residual <- model(input_tensor)$squeeze(2)
  if (as.integer(residual$size(2)) != cfg$p) stop("Circular CNN output width mismatch")
  base_log_tensor + residual
}

fit_circular_cnn <- function(Xtr, Ytr, cfg, rff_train, torch_seed) {
  t0 <- proc.time()[3]
  torch::torch_manual_seed(safe_seed(torch_seed))
  try(torch::torch_set_num_threads(cfg$cnn_torch_threads), silent = TRUE)

  cache <- build_circular_candidate_cache(Xtr, Ytr, cfg, rff_train)
  tc <- circular_training_tensors(cache, cfg)
  n <- nrow(Xtr)
  model <- make_circular_cnn(cfg)
  project_spectral_norm_layers_(list(model$conv1, model$conv2, model$conv3), cfg$cnn_spectral_caps)
  opt <- torch::optim_adam(cnn_model_parameters(model), lr = cfg$cnn_lr, weight_decay = cfg$cnn_weight_decay)

  full_idx <- c(seq_len(n), n + seq_len(n))
  use_full <- min(as.integer(cfg$cnn_batch_size), n) >= n

  for (ep in seq_len(cfg$cnn_epochs)) {
    batches <- balanced_batches(n, cfg$cnn_batch_size)
    model$train()
    opt$zero_grad()
    nb <- length(batches)
    for (idx in batches) {
      b <- length(idx) %/% 2L
      if (use_full && b == n && identical(as.integer(idx), as.integer(full_idx))) {
        prob <- circular_logits_precomputed(model, tc$input, tc$base_log, cfg)$softmax(dim = 2)
        Fc <- tc$candidate_rff
      } else {
        logits <- circular_logits_canonical(model, cache$Zc[idx, , drop = FALSE], cfg)
        prob <- logits$softmax(dim = 2)
        Fc <- torch::torch_tensor(cache$candidate_rff[idx, , , drop = FALSE], dtype = torch::torch_float32())
      }
      Fbar <- torch::torch_einsum("ng,ngd->nd", list(prob, Fc))
      obj <- mmd_power_objective_torch(Fbar, b, cfg$cnn_lambda)
      (obj$loss / nb)$backward()
    }
    opt$step()
    project_spectral_norm_layers_(list(model$conv1, model$conv2, model$conv3), cfg$cnn_spectral_caps)
  }
  model$eval()
  list(model = model, train_seconds = proc.time()[3] - t0)
}

# Finite-S features -------------------------------------------------------------

circular_prob_features <- function(Z, S, cfg, rff, sampler = c("haar", "energy", "cnn"), model = NULL, seed = 1L) {
  sampler <- match.arg(sampler)
  set.seed(safe_seed(seed))
  Z <- as.matrix(Z)
  N <- nrow(Z)

  if (sampler == "cnn") {
    Zc <- canonicalize_circular(Z, cfg)
    P <- circular_probs_canonical(model, Zc, cfg)
    reps <- matrix(0, N * S, cfg$p)
    for (i in seq_len(N)) {
      g <- sample.int(cfg$p, S, replace = TRUE, prob = P[i, ]) - 1L
      rr <- ((i - 1L) * S + 1L):(i * S)
      for (s in seq_len(S)) reps[rr[s], ] <- circular_candidate_rep(Zc[i, ], g[s], cfg)
    }
  } else {
    P <- if (sampler == "haar") matrix(1 / cfg$p, N, cfg$p) else circular_energy_probs(Z, cfg)
    reps <- matrix(0, N * S, cfg$p)
    for (i in seq_len(N)) {
      g <- sample.int(cfg$p, S, replace = TRUE, prob = P[i, ]) - 1L
      rr <- ((i - 1L) * S + 1L):(i * S)
      for (s in seq_len(S)) reps[rr[s], ] <- circ_shift(Z[i, ], -g[s])
    }
  }

  FF <- rff_features(reps, rff, cfg$rff_chunk_size)
  F <- matrix(0, N, rff$D)
  for (i in seq_len(N)) {
    rr <- ((i - 1L) * S + 1L):(i * S)
    F[i, ] <- colMeans(FF[rr, , drop = FALSE])
  }
  F
}

# Deterministic competitors ----------------------------------------------------

energy_circular_median_index <- circular_anchor_index

align_energy_circular <- function(Z, cfg) {
  canonicalize_circular(Z, cfg)
}

pooled_medoid_circular <- function(Z, cfg) {
  D <- sqrt(l2_sq_rows(Z, dt = cfg$dt))
  Z[which.min(rowSums(D)), ]
}

align_xcorr_circular <- function(Z, template, cfg) {
  nt <- sqrt(sum(template^2))
  t(vapply(seq_len(nrow(Z)), function(i) {
    best <- -Inf
    best_k <- 0L
    for (k in 0:(cfg$p - 1L)) {
      z <- circ_shift(Z[i, ], k)
      nz <- sqrt(sum(z^2))
      sc <- if (nt <= 1e-15 || nz <= 1e-15) -Inf else sum(template * z) / (nt * nz)
      if (sc > best) {
        best <- sc
        best_k <- k
      }
    }
    circ_shift(Z[i, ], best_k)
  }, numeric(cfg$p)))
}
