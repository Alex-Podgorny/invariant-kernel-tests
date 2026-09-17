# Methods for G = (R,+) ---------------------------------------------------------

make_translation_cfg <- function(shared) {
  cfg <- shared
  cfg$p <- env_int("P_GRID", 128L)
  cfg$window <- env_num("WINDOW", 5.0)
  cfg$t <- seq(-cfg$window, cfg$window, length.out = cfg$p)
  cfg$dt <- cfg$t[2] - cfg$t[1]

  # Generation support margin. This protects interpolation when nuisance
  # translations are applied; it is not a CNN halo.
  cfg$gen_halo <- env_num("GEN_HALO", 3.0)

  cfg$cnn_channels <- env_int("CNN_CHANNELS", 8L)
  cfg$cnn_k <- c(11L, 7L, 5L)
  cfg$cnn_dilation <- c(1L, 4L, 8L)
  cfg$cnn_energy_eps_rel <- 1e-6

  # Moving-chart score grid. The CNN input and output live on exactly this
  # same grid; all convolutions use same padding.
  cfg$t_score <- cfg$t
  cfg$t_cnn_input <- cfg$t_score

  # Representative canvas must contain all residual translations of a curve
  # observed on [-window, window].
  max_shift <- max(abs(cfg$t_score))
  rep_window <- cfg$window + max_shift
  cfg$t_rep <- seq(-rep_window, rep_window, by = cfg$dt)
  cfg$p_rep <- length(cfg$t_rep)

  cfg
}

shift_signal <- function(x, t_in, t_out, tau) {
  stats::approx(
    t_in, x,
    xout = t_out - tau,
    method = "linear",
    yleft = 0,
    yright = 0,
    rule = 1,
    ties = "ordered"
  )$y
}

translation_center <- function(x, t) {
  w <- x^2
  den <- sum(w)
  if (!is.finite(den) || den <= 1e-15) return(0)
  sum(t * w) / den
}

canonicalize_translation <- function(Z, cfg) {
  Z <- as.matrix(Z)
  out <- matrix(0, nrow(Z), cfg$p)
  for (i in seq_len(nrow(Z))) {
    c0 <- translation_center(Z[i, ], cfg$t)
    out[i, ] <- shift_signal(Z[i, ], cfg$t, cfg$t, -c0)
  }
  out
}

embed_translation_matrix <- function(Z, t_out, cfg) {
  Z <- as.matrix(Z)
  t(vapply(
    seq_len(nrow(Z)),
    function(i) {
      stats::approx(
        cfg$t, Z[i, ],
        xout = t_out,
        method = "linear",
        yleft = 0,
        yright = 0,
        rule = 1,
        ties = "ordered"
      )$y
    },
    numeric(length(t_out))
  ))
}

translation_energy_probs <- function(Zc, cfg) {
  score_signal <- embed_translation_matrix(Zc, cfg$t_score, cfg)
  E <- score_signal^2
  E <- E + cfg$cnn_energy_eps_rel * rowMeans(E) + 1e-20
  E / rowSums(E)
}

make_translation_cnn <- function(cfg) {
  d <- cfg$cnn_dilation
  k <- cfg$cnn_k
  p <- as.integer(d * (k - 1L) / 2L)

  Net <- torch::nn_module(
    classname = "TranslationSamplerCNNFinal",
    initialize = function() {
      self$conv1 <- torch::nn_conv1d(
        1L, cfg$cnn_channels,
        kernel_size = k[1],
        dilation = d[1],
        padding = p[1],
        bias = FALSE
      )
      self$conv2 <- torch::nn_conv1d(
        cfg$cnn_channels, cfg$cnn_channels,
        kernel_size = k[2],
        dilation = d[2],
        padding = p[2],
        bias = FALSE
      )
      self$conv3 <- torch::nn_conv1d(
        cfg$cnn_channels, 1L,
        kernel_size = k[3],
        dilation = d[3],
        padding = p[3],
        bias = FALSE
      )
    },
    forward = function(x) {
      z <- self$conv1(x)$tanh()
      z <- self$conv2(z)$tanh()
      self$conv3(z)
    }
  )

  model <- Net()
  torch::nn_init_zeros_(model$conv3$weight)
  model
}

translation_logits <- function(model, Zc, cfg) {
  Xin <- embed_translation_matrix(Zc, cfg$t_cnn_input, cfg)
  rms <- sqrt(rowMeans(Xin^2))
  rms[!is.finite(rms) | rms < 1e-12] <- 1
  Xn <- Xin / rms

  X <- torch::torch_tensor(Xn, dtype = torch::torch_float32())$unsqueeze(2)
  residual <- model(X)$squeeze(2)

  if (as.integer(residual$size(2)) != length(cfg$t_score)) {
    stop("Translation CNN output width mismatch")
  }

  base <- translation_energy_probs(Zc, cfg)
  torch::torch_tensor(log(base), dtype = torch::torch_float32()) + residual
}

translation_probs <- function(model, Z, cfg) {
  Zc <- canonicalize_translation(Z, cfg)
  model$eval()
  P <- torch::with_no_grad(
    translation_logits(model, Zc, cfg)$softmax(dim = 2)
  )
  matrix(
    as.numeric(as.array(P)),
    nrow = nrow(Z),
    ncol = length(cfg$t_score),
    byrow = FALSE
  )
}

translation_candidate_rep <- function(xc, g, cfg) {
  shift_signal(xc, cfg$t, cfg$t_rep, -g)
}

build_translation_candidate_cache <- function(Xtr, Ytr, cfg, rff) {
  Z <- rbind(Xtr, Ytr)
  Zc <- canonicalize_translation(Z, cfg)
  N <- nrow(Z)
  G <- length(cfg$t_score)
  D <- rff$D

  # Build all (observation, residual group element) representatives first,
  # then evaluate the RFF map in one batched call.
  reps <- array(0, dim = c(N, G, cfg$p_rep))
  for (q in seq_len(G)) {
    reps[, q, ] <- t(vapply(
      seq_len(N),
      function(i) {
        translation_candidate_rep(
          Zc[i, ],
          cfg$t_score[q],
          cfg
        )
      },
      numeric(cfg$p_rep)
    ))
  }

  reps_flat <- matrix(reps, nrow = N * G, ncol = cfg$p_rep)
  Fflat <- rff_features(reps_flat, rff, cfg$rff_chunk_size)
  A <- array(Fflat, dim = c(N, G, D))
  rm(reps, reps_flat, Fflat)

  list(
    Z = Z,
    Zc = Zc,
    candidate_rff = A
  )
}

translation_training_tensors <- function(cache, cfg) {
  # Everything independent of theta is transferred to LibTorch once.
  Xin <- embed_translation_matrix(cache$Zc, cfg$t_cnn_input, cfg)
  rms <- sqrt(rowMeans(Xin^2))
  rms[!is.finite(rms) | rms < 1e-12] <- 1

  X <- torch::torch_tensor(
    Xin / rms,
    dtype = torch::torch_float32()
  )$unsqueeze(2)

  B <- torch::torch_tensor(
    log(translation_energy_probs(cache$Zc, cfg)),
    dtype = torch::torch_float32()
  )

  F <- torch::torch_tensor(
    cache$candidate_rff,
    dtype = torch::torch_float32()
  )

  list(
    input = X,
    base_log = B,
    candidate_rff = F
  )
}

translation_logits_precomputed <- function(
    model,
    input_tensor,
    base_log_tensor,
    cfg
) {
  residual <- model(input_tensor)$squeeze(2)
  if (as.integer(residual$size(2)) != length(cfg$t_score)) {
    stop("Translation CNN output width mismatch")
  }
  base_log_tensor + residual
}

fit_translation_cnn <- function(
    Xtr,
    Ytr,
    cfg,
    rff_train,
    torch_seed
) {
  t0 <- proc.time()[3]

  torch::torch_manual_seed(safe_seed(torch_seed))
  try(
    torch::torch_set_num_threads(cfg$cnn_torch_threads),
    silent = TRUE
  )

  cache <- build_translation_candidate_cache(
    Xtr, Ytr, cfg, rff_train
  )
  tc <- translation_training_tensors(cache, cfg)

  n <- nrow(Xtr)
  model <- make_translation_cnn(cfg)

  project_spectral_norm_layers_(
    list(model$conv1, model$conv2, model$conv3),
    cfg$cnn_spectral_caps
  )

  opt <- torch::optim_adam(
    cnn_model_parameters(model),
    lr = cfg$cnn_lr,
    weight_decay = cfg$cnn_weight_decay
  )

  # With the final default n_train == batch_size, the full training fold is
  # normally one balanced batch and no R-to-Torch conversion occurs inside
  # the epoch loop.
  full_idx <- c(seq_len(n), n + seq_len(n))
  use_full <- min(as.integer(cfg$cnn_batch_size), n) >= n

  for (ep in seq_len(cfg$cnn_epochs)) {
    batches <- balanced_batches(n, cfg$cnn_batch_size)
    model$train()
    opt$zero_grad()
    nb <- length(batches)

    for (idx in batches) {
      b <- length(idx) %/% 2L

      if (
        use_full &&
        b == n &&
        identical(as.integer(idx), as.integer(full_idx))
      ) {
        logits <- translation_logits_precomputed(
          model,
          tc$input,
          tc$base_log,
          cfg
        )
        Fcan <- tc$candidate_rff
      } else {
        # Fallback for non-default minibatch settings.
        logits <- translation_logits(
          model,
          cache$Zc[idx, , drop = FALSE],
          cfg
        )
        Fcan <- torch::torch_tensor(
          cache$candidate_rff[idx, , , drop = FALSE],
          dtype = torch::torch_float32()
        )
      }

      prob <- logits$softmax(dim = 2)
      Fbar <- torch::torch_einsum(
        "ng,ngd->nd",
        list(prob, Fcan)
      )

      obj <- mmd_power_objective_torch(
        Fbar,
        b,
        cfg$cnn_lambda
      )

      (obj$loss / nb)$backward()
    }

    opt$step()

    project_spectral_norm_layers_(
      list(model$conv1, model$conv2, model$conv3),
      cfg$cnn_spectral_caps
    )
  }

  model$eval()

  list(
    model = model,
    train_seconds = proc.time()[3] - t0
  )
}

translation_prob_features <- function(
    Z,
    S,
    cfg,
    rff,
    sampler = c("energy", "cnn"),
    model = NULL,
    seed = 1L
) {
  sampler <- match.arg(sampler)
  set.seed(safe_seed(seed))

  Z <- as.matrix(Z)
  N <- nrow(Z)

  if (sampler == "energy") {
    # Fixed competitor: the sampler is defined directly on the raw curve,
    # A_x(g) proportional to |x(g)|^2.
    E <- Z^2 + 1e-20
    P <- E / rowSums(E)

    reps <- matrix(0, N * S, cfg$p)
    for (i in seq_len(N)) {
      q <- sample.int(
        cfg$p,
        S,
        replace = TRUE,
        prob = P[i, ]
      )
      rr <- ((i - 1L) * S + 1L):(i * S)

      for (ss in seq_len(S)) {
        reps[rr[ss], ] <- shift_signal(
          Z[i, ],
          cfg$t,
          cfg$t,
          -cfg$t[q[ss]]
        )
      }
    }
  } else {
    Zc <- canonicalize_translation(Z, cfg)
    P <- translation_probs(model, Z, cfg)

    reps <- matrix(0, N * S, cfg$p_rep)

    for (i in seq_len(N)) {
      q <- sample.int(
        ncol(P),
        S,
        replace = TRUE,
        prob = P[i, ]
      )
      rr <- ((i - 1L) * S + 1L):(i * S)

      for (ss in seq_len(S)) {
        reps[rr[ss], ] <- translation_candidate_rep(
          Zc[i, ],
          cfg$t_score[q[ss]],
          cfg
        )
      }
    }
  }

  FF <- rff_features(
    reps,
    rff,
    cfg$rff_chunk_size
  )

  F <- matrix(0, N, rff$D)
  for (i in seq_len(N)) {
    rr <- ((i - 1L) * S + 1L):(i * S)
    F[i, ] <- colMeans(
      FF[rr, , drop = FALSE]
    )
  }

  F
}

weighted_median <- function(x, w) {
  o <- order(x)
  x <- x[o]
  w <- w[o]
  x[which(cumsum(w) >= 0.5 * sum(w))[1]]
}

align_energy_median_translation <- function(Z, cfg) {
  t(vapply(
    seq_len(nrow(Z)),
    function(i) {
      c0 <- weighted_median(
        cfg$t,
        Z[i, ]^2 + 1e-20
      )
      shift_signal(
        Z[i, ],
        cfg$t,
        cfg$t,
        -c0
      )
    },
    numeric(cfg$p)
  ))
}

pooled_medoid <- function(Z, dt) {
  D <- sqrt(l2_sq_rows(Z, dt = dt))
  Z[which.min(rowSums(D)), ]
}

align_xcorr_translation <- function(Z, template, cfg) {
  grid <- seq(-3, 3, by = cfg$dt)
  nt <- sqrt(sum(template^2))

  t(vapply(
    seq_len(nrow(Z)),
    function(i) {
      x <- Z[i, ]
      best <- -Inf
      best_tau <- 0

      for (tau in grid) {
        z <- shift_signal(
          x,
          cfg$t,
          cfg$t,
          tau
        )
        nz <- sqrt(sum(z^2))
        sc <- if (nz <= 1e-15 || nt <= 1e-15) {
          -Inf
        } else {
          sum(z * template) / (nz * nt)
        }

        if (sc > best) {
          best <- sc
          best_tau <- tau
        }
      }

      shift_signal(
        x,
        cfg$t,
        cfg$t,
        best_tau
      )
    },
    numeric(cfg$p)
  ))
}
