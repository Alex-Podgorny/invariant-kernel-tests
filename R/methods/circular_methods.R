# Methods for G = S^1 (numerical group C_p) ------------------------------------

make_circular_cfg <- function(shared) {
  cfg <- shared
  cfg$p <- env_int("P_GRID", 128L)
  cfg$period <- 2 * pi
  cfg$dt <- cfg$period / cfg$p
  cfg$t <- (0:(cfg$p - 1L)) * cfg$dt
  cfg$u <- ((0:(cfg$p - 1L) + cfg$p %/% 2L) %% cfg$p - cfg$p %/% 2L) * cfg$dt
  cfg$phase_gap <- env_num("PHASE_GAP", 1.0)
  cfg$sigma_phase <- env_num("SIGMA_PHASE", 0.8)
  cfg$cnn_channels <- env_int("CNN_CHANNELS", 8L)
  cfg$cnn_k <- c(11L, 7L, 5L)
  cfg$cnn_dilation <- c(1L, 4L, 8L)
  cfg$cnn_energy_eps_rel <- 1e-6
  cfg
}

circular_energy_probs <- function(Z, cfg) {
  E <- as.matrix(Z)^2
  E <- E + cfg$cnn_energy_eps_rel * rowMeans(E) + 1e-20
  E / rowSums(E)
}

make_circular_cnn <- function(cfg) {
  d <- cfg$cnn_dilation; k <- cfg$cnn_k
  Net <- torch::nn_module(
    classname = "CircularSamplerCNNFinal",
    initialize = function() {
      self$conv1 <- torch::nn_conv1d(1L, cfg$cnn_channels, kernel_size = k[1], dilation = d[1], padding = 0, bias = FALSE)
      self$conv2 <- torch::nn_conv1d(cfg$cnn_channels, cfg$cnn_channels, kernel_size = k[2], dilation = d[2], padding = 0, bias = FALSE)
      self$conv3 <- torch::nn_conv1d(cfg$cnn_channels, 1L, kernel_size = k[3], dilation = d[3], padding = 0, bias = FALSE)
    },
    circular_pad = function(x, pad) {
      pad <- as.integer(pad); if (pad <= 0L) return(x)
      P <- as.integer(x$size(3))
      torch::torch_cat(list(x$narrow(3, P - pad + 1L, pad), x, x$narrow(3, 1L, pad)), dim = 3)
    },
    forward = function(x) {
      p1 <- d[1] * (k[1] - 1L) %/% 2L; p2 <- d[2] * (k[2] - 1L) %/% 2L; p3 <- d[3] * (k[3] - 1L) %/% 2L
      z <- self$conv1(self$circular_pad(x, p1))$tanh()
      z <- self$conv2(self$circular_pad(z, p2))$tanh()
      self$conv3(self$circular_pad(z, p3))
    }
  )
  model <- Net(); torch::nn_init_zeros_(model$conv3$weight); model
}

circular_logits <- function(model, Z, cfg) {
  Z <- as.matrix(Z); E <- Z^2; rms <- sqrt(rowMeans(E)); rms[rms < 1e-12] <- 1
  X <- torch::torch_tensor(Z / rms, dtype = torch::torch_float32())$unsqueeze(2)
  residual <- model(X)$squeeze(2)
  base <- circular_energy_probs(Z, cfg)
  torch::torch_tensor(log(base), dtype = torch::torch_float32()) + residual
}
circular_probs <- function(model, Z, cfg) {
  model$eval(); P <- torch::with_no_grad(circular_logits(model, Z, cfg)$softmax(dim = 2))
  matrix(as.numeric(as.array(P)), nrow = nrow(Z), ncol = cfg$p, byrow = FALSE)
}

build_circular_candidate_cache <- function(Xtr, Ytr, cfg, rff) {
  Z <- rbind(Xtr, Ytr); N <- nrow(Z); P <- cfg$p; D <- rff$D

  # One vectorized RFF call for all N*P cyclic representatives.
  reps <- array(0, dim = c(N, P, P))
  for (g in 0:(P - 1L)) reps[, g + 1L, ] <- circ_shift_matrix(Z, -g)
  reps_flat <- matrix(reps, nrow = N * P, ncol = P)
  Fflat <- rff_features(reps_flat, rff, cfg$rff_chunk_size)
  A <- array(Fflat, dim = c(N, P, D))
  rm(reps, reps_flat, Fflat)
  list(Z = Z, candidate_rff = A)
}

circular_training_tensors <- function(cache, cfg) {
  Z <- as.matrix(cache$Z)
  E <- Z^2
  rms <- sqrt(rowMeans(E)); rms[!is.finite(rms) | rms < 1e-12] <- 1
  X <- torch::torch_tensor(Z / rms, dtype = torch::torch_float32())$unsqueeze(2)
  B <- torch::torch_tensor(log(circular_energy_probs(Z, cfg)), dtype = torch::torch_float32())
  F <- torch::torch_tensor(cache$candidate_rff, dtype = torch::torch_float32())
  list(input = X, base_log = B, candidate_rff = F)
}

circular_logits_precomputed <- function(model, input_tensor, base_log_tensor) {
  base_log_tensor + model(input_tensor)$squeeze(2)
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
    model$train(); opt$zero_grad(); nb <- length(batches)
    for (idx in batches) {
      b <- length(idx) %/% 2L
      if (use_full && b == n && identical(as.integer(idx), as.integer(full_idx))) {
        prob <- circular_logits_precomputed(model, tc$input, tc$base_log)$softmax(dim = 2)
        Fc <- tc$candidate_rff
      } else {
        prob <- circular_logits(model, cache$Z[idx, , drop = FALSE], cfg)$softmax(dim = 2)
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

circular_prob_features <- function(Z, S, cfg, rff, sampler = c("haar", "energy", "cnn"), model = NULL, seed = 1L) {
  sampler <- match.arg(sampler); set.seed(safe_seed(seed)); Z <- as.matrix(Z); N <- nrow(Z)
  P <- switch(sampler,
    haar = matrix(1 / cfg$p, N, cfg$p),
    energy = circular_energy_probs(Z, cfg),
    cnn = circular_probs(model, Z, cfg))
  reps <- matrix(0, N * S, cfg$p)
  for (i in seq_len(N)) {
    g <- sample.int(cfg$p, S, replace = TRUE, prob = P[i, ]) - 1L
    rr <- ((i - 1L) * S + 1L):(i * S)
    for (s in seq_len(S)) reps[rr[s], ] <- circ_shift(Z[i, ], -g[s])
  }
  FF <- rff_features(reps, rff, cfg$rff_chunk_size); F <- matrix(0, N, rff$D)
  for (i in seq_len(N)) { rr <- ((i - 1L) * S + 1L):(i * S); F[i, ] <- colMeans(FF[rr, , drop = FALSE]) }
  F
}

energy_circular_median_index <- function(x, cfg) {
  w <- x^2; if (sum(w) <= 1e-15) return(0L)
  idx <- 0:(cfg$p - 1L); obj <- numeric(cfg$p)
  for (c in idx) { d <- abs(idx - c); d <- pmin(d, cfg$p - d); obj[c + 1L] <- sum(w * d) }
  which.min(obj) - 1L
}
align_energy_circular <- function(Z, cfg) {
  t(vapply(seq_len(nrow(Z)), function(i) circ_shift(Z[i, ], -energy_circular_median_index(Z[i, ], cfg)), numeric(cfg$p)))
}
pooled_medoid_circular <- function(Z, cfg) {
  D <- sqrt(l2_sq_rows(Z, dt = cfg$dt)); Z[which.min(rowSums(D)), ]
}
align_xcorr_circular <- function(Z, template, cfg) {
  nt <- sqrt(sum(template^2))
  t(vapply(seq_len(nrow(Z)), function(i) {
    best <- -Inf; best_k <- 0L
    for (k in 0:(cfg$p - 1L)) {
      z <- circ_shift(Z[i, ], k); nz <- sqrt(sum(z^2))
      sc <- if (nt <= 1e-15 || nz <= 1e-15) -Inf else sum(template * z) / (nt * nz)
      if (sc > best) { best <- sc; best_k <- k }
    }
    circ_shift(Z[i, ], best_k)
  }, numeric(cfg$p)))
}
