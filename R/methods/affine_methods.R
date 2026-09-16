# Methods for G = Aff^+(1) ------------------------------------------------------

align_floor <- function(x, step) floor(x / step + 1e-12) * step
align_ceil <- function(x, step) ceiling(x / step - 1e-12) * step
make_uniform_grid <- function(lo, hi, step) {
  lo <- align_floor(lo, step); hi <- align_ceil(hi, step)
  lo + (0:as.integer(round((hi - lo) / step))) * step
}
make_group_grid <- function(alpha_min, alpha_max, b_min, b_max, cfg) {
  a <- make_uniform_grid(alpha_min, alpha_max, cfg$affine_alpha_step)
  b <- make_uniform_grid(b_min, b_max, cfg$affine_b_step)
  list(alpha = a, b = b, na = length(a), nb = length(b), alpha_min = min(a), alpha_max = max(a),
       b_min = min(b), b_max = max(b), dalpha = cfg$affine_alpha_step, db = cfg$affine_b_step)
}
expand_group_grid_for_layer <- function(out_grid, kalpha, kbeta, dil_alpha, dil_beta, cfg) {
  ra <- ((kalpha - 1L) %/% 2L) * dil_alpha * cfg$affine_alpha_step
  rb_local <- ((kbeta - 1L) %/% 2L) * dil_beta * cfg$affine_b_step
  rb_world <- exp(out_grid$alpha_max) * rb_local
  make_group_grid(out_grid$alpha_min - ra, out_grid$alpha_max + ra,
                  out_grid$b_min - rb_world - cfg$affine_b_step,
                  out_grid$b_max + rb_world + cfg$affine_b_step, cfg)
}
affine_canvas_spec <- function(cfg) {
  score <- make_group_grid(-cfg$affine_alpha_max, cfg$affine_alpha_max, -cfg$affine_b_max, cfg$affine_b_max, cfg)
  g2 <- expand_group_grid_for_layer(score, cfg$gcnn_kalpha[3], cfg$gcnn_kbeta[3], cfg$gcnn_alpha_dilations[3], cfg$gcnn_beta_dilations[3], cfg)
  g1 <- expand_group_grid_for_layer(g2, cfg$gcnn_kalpha[2], cfg$gcnn_kbeta[2], cfg$gcnn_alpha_dilations[2], cfg$gcnn_beta_dilations[2], cfg)
  g0 <- expand_group_grid_for_layer(g1, cfg$gcnn_kalpha[1], cfg$gcnn_kbeta[1], cfg$gcnn_alpha_dilations[1], cfg$gcnn_beta_dilations[1], cfg)
  list(input = g0, layer1 = g1, layer2 = g2, score = score)
}

make_affine_cfg <- function(shared) {
  cfg <- shared
  cfg$p_obs <- env_int("P_OBS", 192L); cfg$obs_window <- env_num("OBS_WINDOW", 7.5)
  cfg$t_obs <- seq(-cfg$obs_window, cfg$obs_window, length.out = cfg$p_obs); cfg$dt <- cfg$t_obs[2] - cfg$t_obs[1]
  cfg$nuisance_alpha_gap <- env_num("AFFINE_ALPHA_GAP", 0.1); cfg$nuisance_alpha_sd <- 0.10; cfg$nuisance_alpha_absmax <- 0.40
  cfg$nuisance_b_gap <- env_num("AFFINE_B_GAP", 0.50); cfg$nuisance_b_sd <- 0.35; cfg$nuisance_b_absmax <- 1.60 
  cfg$canonical_target_sd <- 1.0; cfg$canonical_sd_floor <- 0.10
  cfg$wavelet_sd <- 0.45
  cfg$affine_alpha_max <- 0.45; cfg$affine_alpha_step <- 0.15
  cfg$affine_b_max <- 2.40; cfg$affine_b_step <- 2 * cfg$dt
  cfg$gcnn_channels <- 8L; cfg$gcnn_kalpha <- c(3L,3L,3L); cfg$gcnn_kbeta <- c(3L,3L,3L)
  cfg$gcnn_alpha_dilations <- c(1L,1L,1L); cfg$gcnn_beta_dilations <- c(1L,2L,4L)
  cfg$canvas <- affine_canvas_spec(cfg); cfg$G <- cfg$canvas$score$na * cfg$canvas$score$nb
  cfg$rep_window <- (cfg$obs_window + cfg$canvas$score$b_max) * exp(cfg$canvas$score$alpha_max) + 0.50
  cfg$t_rep <- (-ceiling(cfg$rep_window/cfg$dt):ceiling(cfg$rep_window/cfg$dt)) * cfg$dt; cfg$p_rep <- length(cfg$t_rep)
  cfg$gen_window <- (cfg$obs_window + cfg$nuisance_b_absmax) * exp(cfg$nuisance_alpha_absmax) + 1.0
  cfg$t_gen <- (-ceiling(cfg$gen_window/cfg$dt):ceiling(cfg$gen_window/cfg$dt)) * cfg$dt
  cfg$score_table <- score_group_table(cfg)
  cfg$rep_maps <- lapply(seq_len(nrow(cfg$score_table)), function(q) make_inverse_rep_map(cfg$score_table$alpha[q], cfg$score_table$b[q], cfg))
  cfg$wavelet_input <- build_wavelet_dictionary(cfg$canvas$input, cfg)
  cfg$wavelet_score <- build_wavelet_dictionary(cfg$canvas$score, cfg)
  cfg
}

affine_apply <- function(x, t_in, t_out, alpha, b) {
  u <- exp(-alpha) * (t_out - b)
  exp(-alpha/2) * stats::approx(t_in, x, xout = u, method = "linear", yleft = 0, yright = 0, rule = 1, ties = "ordered")$y
}
affine_apply_inverse <- function(x, t_in, t_out, alpha, b) {
  u <- exp(alpha) * t_out + b
  exp(alpha/2) * stats::approx(t_in, x, xout = u, method = "linear", yleft = 0, yright = 0, rule = 1, ties = "ordered")$y
}
energy_affine_anchor <- function(Z, cfg) {
  Z <- as.matrix(Z); E <- Z^2; den <- rowSums(E); den[den < 1e-15] <- 1
  mu <- as.numeric(E %*% cfg$t_obs) / den
  V <- matrix(cfg$t_obs, nrow(Z), cfg$p_obs, byrow = TRUE) - mu
  sd <- sqrt(rowSums(E * V^2) / den); sd[!is.finite(sd) | sd < cfg$canonical_sd_floor] <- cfg$canonical_sd_floor
  data.frame(alpha = log(sd / cfg$canonical_target_sd), b = mu)
}
canonicalize_affine <- function(Z, cfg) {
  Z <- as.matrix(Z); a <- energy_affine_anchor(Z, cfg); out <- matrix(0, nrow(Z), cfg$p_obs)
  for (i in seq_len(nrow(Z))) out[i,] <- affine_apply_inverse(Z[i,], cfg$t_obs, cfg$t_obs, a$alpha[i], a$b[i])
  out
}
embed_obs_to_rep <- function(Z, cfg) {
  Z <- as.matrix(Z)
  t(vapply(seq_len(nrow(Z)), function(i) stats::approx(cfg$t_obs, Z[i,], xout = cfg$t_rep,
    method = "linear", yleft = 0, yright = 0, rule = 1, ties = "ordered")$y, numeric(cfg$p_rep)))
}
make_inverse_rep_map <- function(alpha, b, cfg) {
  u <- exp(alpha) * cfg$t_rep + b; pos <- (u - cfg$t_obs[1]) / cfg$dt + 1
  lo <- floor(pos); frac <- pos - lo; inside <- lo >= 1 & lo < cfg$p_obs
  at_right <- abs(pos - cfg$p_obs) < 1e-10; lo[at_right] <- cfg$p_obs - 1L; frac[at_right] <- 1; inside[at_right] <- TRUE
  lo2 <- pmax(1L, pmin(cfg$p_obs - 1L, as.integer(lo)))
  list(lo = lo2, hi = lo2 + 1L, frac = frac, inside = inside, scale = exp(alpha/2))
}
apply_inverse_rep_map_matrix <- function(Z, mp) {
  Z <- as.matrix(Z); N <- nrow(Z); out <- matrix(0, N, length(mp$inside)); jj <- which(mp$inside)
  if (length(jj)) {
    l <- mp$lo[jj]; h <- mp$hi[jj]; f <- mp$frac[jj]
    out[,jj] <- mp$scale * (Z[,l,drop=FALSE] * rep(1-f, each=N) + Z[,h,drop=FALSE] * rep(f, each=N))
  }
  out
}
score_group_table <- function(cfg) {
  g <- cfg$canvas$score; out <- expand.grid(ib = seq_len(g$nb), ia = seq_len(g$na)); out <- out[order(out$ia,out$ib),]
  out$alpha <- g$alpha[out$ia]; out$b <- g$b[out$ib]; out$q <- seq_len(nrow(out)); out
}
transform_canonical_by_candidate <- function(Zc, q, cfg) apply_inverse_rep_map_matrix(Zc, cfg$rep_maps[[q]])

mexican_hat <- function(u, sd) { v <- u/sd; (1-v^2) * exp(-0.5*v^2) }
build_wavelet_dictionary <- function(grid, cfg) {
  G <- grid$na * grid$nb; atoms <- matrix(0, cfg$p_obs, G); alpha <- numeric(G); q <- 0L
  for (ia in seq_len(grid$na)) for (ib in seq_len(grid$nb)) {
    q <- q+1L; al <- grid$alpha[ia]; b <- grid$b[ib]; a <- exp(al)
    atoms[,q] <- cfg$dt * exp(-al/2) * mexican_hat((cfg$t_obs-b)/a, cfg$wavelet_sd); alpha[q] <- al
  }
  list(atoms=atoms, haar=exp(-alpha)*grid$dalpha*grid$db, na=grid$na, nb=grid$nb)
}
normalize_signal_l2 <- function(Z,cfg) { Z<-as.matrix(Z); nr<-sqrt(cfg$dt*rowSums(Z^2)); nr[nr<1e-12]<-1; Z/nr }
wavelet_coeff_matrix <- function(Z, dict, cfg) normalize_signal_l2(Z,cfg) %*% dict$atoms
wavelet_lift_canonical <- function(Zc,cfg) {
  W <- wavelet_coeff_matrix(Zc,cfg$wavelet_input,cfg); arr <- array(0,c(nrow(W),cfg$canvas$input$na,cfg$canvas$input$nb))
  for(ia in seq_len(cfg$canvas$input$na)) { jj<-((ia-1L)*cfg$canvas$input$nb+1L):(ia*cfg$canvas$input$nb); arr[,ia,]<-W[,jj,drop=FALSE] }
  arr
}
wavelet_base_mass_canonical <- function(Zc,cfg) {
  W<-wavelet_coeff_matrix(Zc,cfg$wavelet_score,cfg); M<-sweep(W^2,2,cfg$wavelet_score$haar,"*"); den<-rowSums(M)
  if(any(den<=1e-30 | !is.finite(den))) stop("Zero wavelet mass"); M/den
}
wavelet_base_logmass_canonical <- function(Zc,cfg) { P<-wavelet_base_mass_canonical(Zc,cfg); P[P<1e-30]<-1e-30; log(P) }

build_layer_sampling_grids <- function(in_grid,out_grid,kalpha,kbeta,dil_alpha,dil_beta,cfg) {
  oa<-seq.int(-(kalpha-1L)%/%2L,(kalpha-1L)%/%2L)*dil_alpha*cfg$affine_alpha_step
  ob<-seq.int(-(kbeta-1L)%/%2L,(kbeta-1L)%/%2L)*dil_beta*cfg$affine_b_step
  grids<-vector("list",length(oa)*length(ob)); q<-0L
  for(da in oa) for(dbeta in ob) {
    q<-q+1L; A<-matrix(rep(out_grid$alpha+da,times=out_grid$nb),nrow=out_grid$na,ncol=out_grid$nb,byrow=FALSE)
    B<-matrix(0,out_grid$na,out_grid$nb); for(ia in seq_len(out_grid$na)) B[ia,]<-out_grid$b+exp(out_grid$alpha[ia])*dbeta
    yn<-2*(A-in_grid$alpha_min)/(in_grid$alpha_max-in_grid$alpha_min)-1; xn<-2*(B-in_grid$b_min)/(in_grid$b_max-in_grid$b_min)-1
    ar<-array(0,c(1L,out_grid$na,out_grid$nb,2L)); ar[1,,,1]<-xn; ar[1,,,2]<-yn
    grids[[q]]<-torch::torch_tensor(ar,dtype=torch::torch_float32())
  }
  grids
}

AffineGroupConv <- torch::nn_module(
  classname="AffineGroupConvFinal",
  initialize=function(in_channels,out_channels,sampling_grids,kalpha,kbeta,zero_init=FALSE) {
    self$kalpha<-as.integer(kalpha); self$kbeta<-as.integer(kbeta); fanin<-max(1,in_channels*kalpha*kbeta)
    w<-torch::torch_randn(c(out_channels,in_channels,kalpha,kbeta),dtype=torch::torch_float32())*sqrt(2/fanin); if(zero_init) w$zero_()
    self$weight<-torch::nn_parameter(w); self$sampling_grids<-sampling_grids
  },
  forward=function(x) {
    B<-as.integer(x$size(1)); ans<-NULL; q<-0L
    for(ia in seq_len(self$kalpha)) for(ib in seq_len(self$kbeta)) {
      q<-q+1L; grid<-self$sampling_grids[[q]]$`repeat`(c(B,1L,1L,1L))
      s<-torch::nnf_grid_sample(x,grid,mode="bilinear",padding_mode="zeros",align_corners=TRUE)
      w<-self$weight$narrow(3,ia,1L)$narrow(4,ib,1L)$squeeze(4)$squeeze(3)
      z<-torch::torch_einsum("bihw,oi->bohw",list(s,w)); ans<-if(is.null(ans)) z else ans+z
    }; ans
  }
)
make_affine_gcnn <- function(cfg) {
  c0<-cfg$canvas$input; c1<-cfg$canvas$layer1; c2<-cfg$canvas$layer2; c3<-cfg$canvas$score
  s1<-build_layer_sampling_grids(c0,c1,cfg$gcnn_kalpha[1],cfg$gcnn_kbeta[1],cfg$gcnn_alpha_dilations[1],cfg$gcnn_beta_dilations[1],cfg)
  s2<-build_layer_sampling_grids(c1,c2,cfg$gcnn_kalpha[2],cfg$gcnn_kbeta[2],cfg$gcnn_alpha_dilations[2],cfg$gcnn_beta_dilations[2],cfg)
  s3<-build_layer_sampling_grids(c2,c3,cfg$gcnn_kalpha[3],cfg$gcnn_kbeta[3],cfg$gcnn_alpha_dilations[3],cfg$gcnn_beta_dilations[3],cfg)
  Net<-torch::nn_module(classname="AffineSamplerGCNNFinal", initialize=function() {
    self$conv1<-AffineGroupConv(1L,cfg$gcnn_channels,s1,cfg$gcnn_kalpha[1],cfg$gcnn_kbeta[1],FALSE)
    self$conv2<-AffineGroupConv(cfg$gcnn_channels,cfg$gcnn_channels,s2,cfg$gcnn_kalpha[2],cfg$gcnn_kbeta[2],FALSE)
    self$conv3<-AffineGroupConv(cfg$gcnn_channels,1L,s3,cfg$gcnn_kalpha[3],cfg$gcnn_kbeta[3],TRUE)
  }, forward=function(x) self$conv3(torch::torch_tanh(self$conv2(torch::torch_tanh(self$conv1(x))))))
  Net()
}

affine_logits_precomputed <- function(model,lift,base_logmass,cfg) {
  r<-model(lift)$reshape(c(as.integer(lift$size(1)),cfg$G)); base_logmass+r
}
affine_probs <- function(model,Z,cfg) {
  Zc<-canonicalize_affine(Z,cfg); L<-wavelet_lift_canonical(Zc,cfg); d<-dim(L)
  Lt<-torch::torch_tensor(array(L,dim=c(d[1],1L,d[2],d[3])),dtype=torch::torch_float32())
  Bt<-torch::torch_tensor(wavelet_base_logmass_canonical(Zc,cfg),dtype=torch::torch_float32())
  model$eval(); P<-torch::with_no_grad(affine_logits_precomputed(model,Lt,Bt,cfg)$softmax(dim=2))
  matrix(as.numeric(as.array(P)),nrow=nrow(Z),ncol=cfg$G,byrow=FALSE)
}

build_affine_candidate_cache <- function(Xtr, Ytr, cfg, rff) {
  Z <- rbind(Xtr, Ytr)
  Zc <- canonicalize_affine(Z, cfg)
  N <- nrow(Z); G <- cfg$G; D <- rff$D

  # Build all affine representatives first and evaluate their RFFs in one call.
  reps <- array(0, dim = c(N, G, cfg$p_rep))
  for (q in seq_len(G)) reps[, q, ] <- transform_canonical_by_candidate(Zc, q, cfg)
  reps_flat <- matrix(reps, nrow = N * G, ncol = cfg$p_rep)
  Fflat <- rff_features(reps_flat, rff, cfg$rff_chunk_size)
  A <- array(Fflat, dim = c(N, G, D))
  rm(reps, reps_flat, Fflat)

  list(
    Z = Z,
    Zc = Zc,
    candidate_rff = A,
    lift = wavelet_lift_canonical(Zc, cfg),
    base = wavelet_base_logmass_canonical(Zc, cfg)
  )
}

affine_lift_tensor <- function(A) {
  d <- dim(A)
  if (length(d) != 3L) stop("Affine lift must have dimensions N x alpha x b")
  torch::torch_tensor(array(A, dim = c(d[1], 1L, d[2], d[3])), dtype = torch::torch_float32())
}

affine_training_tensors <- function(cache) {
  list(
    lift = affine_lift_tensor(cache$lift),
    base_log = torch::torch_tensor(cache$base, dtype = torch::torch_float32()),
    candidate_rff = torch::torch_tensor(cache$candidate_rff, dtype = torch::torch_float32())
  )
}

fit_affine_gcnn <- function(Xtr, Ytr, cfg, rff_train, torch_seed) {
  t0 <- proc.time()[3]
  torch::torch_manual_seed(safe_seed(torch_seed))
  try(torch::torch_set_num_threads(cfg$cnn_torch_threads), silent = TRUE)

  cache <- build_affine_candidate_cache(Xtr, Ytr, cfg, rff_train)
  tc <- affine_training_tensors(cache)
  n <- nrow(Xtr)
  model <- make_affine_gcnn(cfg)
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
        Lt <- tc$lift; Bt <- tc$base_log; Fc <- tc$candidate_rff
      } else {
        # Fallback for non-final custom minibatch settings.
        Lt <- affine_lift_tensor(cache$lift[idx, , , drop = FALSE])
        Bt <- torch::torch_tensor(cache$base[idx, , drop = FALSE], dtype = torch::torch_float32())
        Fc <- torch::torch_tensor(cache$candidate_rff[idx, , , drop = FALSE], dtype = torch::torch_float32())
      }
      P <- affine_logits_precomputed(model, Lt, Bt, cfg)$softmax(dim = 2)
      Fbar <- torch::torch_einsum("ng,ngd->nd", list(P, Fc))
      obj <- mmd_power_objective_torch(Fbar, b, cfg$cnn_lambda)
      (obj$loss / nb)$backward()
    }
    opt$step()
    project_spectral_norm_layers_(list(model$conv1, model$conv2, model$conv3), cfg$cnn_spectral_caps)
  }
  model$eval()
  list(model = model, train_seconds = proc.time()[3] - t0)
}

affine_prob_features <- function(Z,S,cfg,rff,sampler=c("wavelet","cnn"),model=NULL,seed=1L) {
  sampler<-match.arg(sampler); set.seed(safe_seed(seed)); Z<-as.matrix(Z); Zc<-canonicalize_affine(Z,cfg); N<-nrow(Z)
  P<-if(sampler=="wavelet") wavelet_base_mass_canonical(Zc,cfg) else affine_probs(model,Z,cfg)
  reps<-matrix(0,N*S,cfg$p_rep)
  for(i in seq_len(N)) { q<-sample.int(cfg$G,S,replace=TRUE,prob=P[i,]); rr<-((i-1L)*S+1L):(i*S); for(s in seq_len(S)) reps[rr[s],]<-transform_canonical_by_candidate(Zc[i,,drop=FALSE],q[s],cfg)[1,] }
  FF<-rff_features(reps,rff,cfg$rff_chunk_size); F<-matrix(0,N,rff$D)
  for(i in seq_len(N)) { rr<-((i-1L)*S+1L):(i*S); F[i,]<-colMeans(FF[rr,,drop=FALSE]) }; F
}

pooled_medoid_affine <- function(Z,cfg) { R<-embed_obs_to_rep(canonicalize_affine(Z,cfg),cfg); D<-sqrt(l2_sq_rows(R,dt=cfg$dt)); R[which.min(rowSums(D)),] }
align_wavelet_max_affine <- function(Z,cfg) {
  Zc<-canonicalize_affine(Z,cfg); P<-wavelet_base_mass_canonical(Zc,cfg); out<-matrix(0,nrow(Z),cfg$p_rep)
  for(i in seq_len(nrow(Z))) out[i,]<-transform_canonical_by_candidate(Zc[i,,drop=FALSE],which.max(P[i,]),cfg)[1,]; out
}
align_xcorr_affine <- function(Z,template,cfg) {
  Zc<-canonicalize_affine(Z,cfg); out<-matrix(0,nrow(Z),cfg$p_rep); nt<-sqrt(sum(template^2))
  for(i in seq_len(nrow(Z))) { best<--Inf; bestq<-1L; for(q in seq_len(cfg$G)) { z<-transform_canonical_by_candidate(Zc[i,,drop=FALSE],q,cfg)[1,]; nz<-sqrt(sum(z^2)); sc<-if(nt<=1e-15||nz<=1e-15) -Inf else sum(template*z)/(nt*nz); if(sc>best){best<-sc;bestq<-q} }; out[i,]<-transform_canonical_by_candidate(Zc[i,,drop=FALSE],bestq,cfg)[1,] }
  out
}
