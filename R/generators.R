# Group-specific data generators ------------------------------------------------

rtruncnorm_simple <- function(n, mean, sd, lo, hi) {
  out <- numeric(n); filled <- 0L
  while (filled < n) {
    z <- rnorm(max(16L, 2L * (n - filled)), mean, sd)
    z <- z[z >= lo & z <= hi]
    if (!length(z)) next
    take <- min(length(z), n - filled)
    out[filled + seq_len(take)] <- z[seq_len(take)]
    filled <- filled + take
  }
  out
}

generate_translation_split <- function(scenario, delta, cfg) {
  nall <- cfg$n_train + cfg$n_test; templates <- make_aperiodic_templates(cfg)
  dt <- cfg$dt; t_gen <- seq(min(cfg$t)-cfg$gen_halo, max(cfg$t)+cfg$gen_halo, by=dt)
  one_group <- function(group) {
    mu <- if (group=="X") cfg$shift_gap/2 else -cfg$shift_gap/2
    h <- rnorm(nall, mu, cfg$sigma_shift); X <- matrix(0,nall,cfg$p)
    for(i in seq_len(nall)) {
      z <- generate_intrinsic_line(t_gen,scenario,delta,group,cfg,templates)
      X[i,] <- stats::approx(t_gen,z,xout=cfg$t-h[i],method="linear",yleft=0,yright=0,rule=1,ties="ordered")$y
    }
    X
  }
  X<-one_group("X"); Y<-one_group("Y")
  list(Xtr=X[seq_len(cfg$n_train),,drop=FALSE],Ytr=Y[seq_len(cfg$n_train),,drop=FALSE],
       Xte=X[cfg$n_train+seq_len(cfg$n_test),,drop=FALSE],Yte=Y[cfg$n_train+seq_len(cfg$n_test),,drop=FALSE])
}

generate_circular_split <- function(scenario, delta, cfg) {
  nall<-cfg$n_train+cfg$n_test; templates<-make_circular_templates(cfg)
  one_group<-function(group) {
    mu<-if(group=="X") cfg$phase_gap/2 else -cfg$phase_gap/2
    h<-as.integer(round(rnorm(nall,mean=mu/cfg$dt,sd=cfg$sigma_phase/cfg$dt))); X<-matrix(0,nall,cfg$p)
    for(i in seq_len(nall)) X[i,]<-circ_shift(generate_intrinsic_circle(scenario,delta,group,cfg,templates),h[i])
    X
  }
  X<-one_group("X");Y<-one_group("Y")
  list(Xtr=X[seq_len(cfg$n_train),,drop=FALSE],Ytr=Y[seq_len(cfg$n_train),,drop=FALSE],
       Xte=X[cfg$n_train+seq_len(cfg$n_test),,drop=FALSE],Yte=Y[cfg$n_train+seq_len(cfg$n_test),,drop=FALSE])
}

generate_affine_split <- function(scenario, delta, cfg) {
  nall<-cfg$n_train+cfg$n_test; templates<-make_aperiodic_templates(cfg)
  one_group<-function(group) {
    ma<-if(group=="X") cfg$nuisance_alpha_gap/2 else -cfg$nuisance_alpha_gap/2
    mb<-if(group=="X") cfg$nuisance_b_gap/2 else -cfg$nuisance_b_gap/2
    al<-rtruncnorm_simple(nall,ma,cfg$nuisance_alpha_sd,-cfg$nuisance_alpha_absmax,cfg$nuisance_alpha_absmax)
    bb<-rtruncnorm_simple(nall,mb,cfg$nuisance_b_sd,-cfg$nuisance_b_absmax,cfg$nuisance_b_absmax)
    X<-matrix(0,nall,cfg$p_obs)
    for(i in seq_len(nall)) {
      z<-generate_intrinsic_line(cfg$t_gen,scenario,delta,group,cfg,templates)
      X[i,]<-affine_apply(z,cfg$t_gen,cfg$t_obs,al[i],bb[i])
    }
    X
  }
  X<-one_group("X");Y<-one_group("Y")
  list(Xtr=X[seq_len(cfg$n_train),,drop=FALSE],Ytr=Y[seq_len(cfg$n_train),,drop=FALSE],
       Xte=X[cfg$n_train+seq_len(cfg$n_test),,drop=FALSE],Yte=Y[cfg$n_train+seq_len(cfg$n_test),,drop=FALSE])
}
