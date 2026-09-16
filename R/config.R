# Final simulation configuration ------------------------------------------------
# One source of truth for all shared numerical choices.

SCENARIOS <- c(
  M1 = "M1",
  M2 = "M2",
  M3 = "M3",
  M4 = "M4"
)

env_bool <- function(name, default = FALSE) {
  z <- tolower(trimws(Sys.getenv(name, unset = if (default) "1" else "0")))
  z %in% c("1", "true", "t", "yes", "y", "on")
}
env_int <- function(name, default) as.integer(Sys.getenv(name, unset = as.character(default)))
env_num <- function(name, default) as.numeric(Sys.getenv(name, unset = as.character(default)))
parse_num_vec <- function(x, default) {
  x <- trimws(x)
  if (!nzchar(x)) return(as.numeric(default))
  as.numeric(strsplit(x, "[,;[:space:]]+")[[1]])
}
parse_chr_vec <- function(x, default) {
  x <- trimws(x)
  if (!nzchar(x)) return(default)
  trimws(strsplit(x, "[,;]+")[[1]])
}

make_shared_cfg <- function() {
  quick <- env_bool("QUICK", FALSE)
  list(
    seed = env_int("SEED", 20260915L),
    nrep = if (quick) env_int("NREP", 3L) else env_int("NREP", 300L),
    cores = env_int("CORES", 1L),
    n_train = env_int("N_TRAIN", 20L),
    n_test = env_int("N_TEST", 40L),
    deltas = parse_num_vec(Sys.getenv("DELTAS", unset = ""), c(0, .2, .4, .6, .8, 1)),
    scenarios = parse_chr_vec(Sys.getenv("SCENARIOS", unset = ""), unname(SCENARIOS)),
    S_grid = as.integer(parse_num_vec(Sys.getenv("S_GRID", unset = ""), c(4, 8, 16, 32))),
    Bperm = if (quick) env_int("BPERM", 49L) else env_int("BPERM", 500L),
    alpha_test = env_num("ALPHA", 0.05),
    shift_gap = env_num("SHIFT_GAP", 0.5), #For R and S1
    sigma_shift = env_num("SIGMA_SHIFT", 0.8), #For R and S1
    rff_dim = if (quick) env_int("RFF_DIM", 128L) else env_int("RFF_DIM", 256L),
    rff_train_dim = if (quick) env_int("RFF_TRAIN_DIM", 128L) else env_int("RFF_TRAIN_DIM", 256L),
    rff_chunk_size = env_int("RFF_CHUNK_SIZE", 2048L),
    rff_seed_offset = env_int("RFF_SEED_OFFSET", 314159L),
    rff_test_seed_gap = env_int("RFF_TEST_SEED_GAP", 7000003L),
    noise_model = "multiplicative",
    noise_mean_mult = 1.0,
    noise_sd_mult = 0.50,
    cnn_batch_size = env_int("CNN_BATCH_SIZE", 20L),
    cnn_epochs = env_int("CNN_EPOCHS", 30L),
    cnn_lr = env_num("CNN_LR", 0.01),
    cnn_weight_decay = env_num("CNN_WEIGHT_DECAY", 1e-4),
    cnn_lambda = env_num("CNN_LAMBDA", 1e-3),
    cnn_torch_threads = env_int("CNN_TORCH_THREADS", 1L),
    cnn_spectral_caps = c(5, 5, 5),
    # Rounded intrinsic parameters, shared by R, S^1 and Aff^+(1).
    sigma_gamma_common = 0.10,
    m1_mix_shift = 0.25,
    m2_coef_sd = 0.25,
    m2_rho_max = 0.80,
    m2_g1_sd = 0.50,
    m2_g2_sd = 0.50,
    m3_peak_effect = 0.50,
    m3_peak_sdlog = 0.20,
    m4_nuisance_k = 3L,
    m4_nuisance_pos_min = -3.0,
    m4_nuisance_pos_max = 3.0,
    m4_nuisance_min_sep = 0.50,
    m4_nuisance_guard = 0.50,
    m4_nuisance_amp = 1.00,
    m4_nuisance_amp_sdlog = 0.20,
    m4_nuisance_sd_min = 0.15,
    m4_nuisance_sd_max = 0.30,
    m4_motif_amp = 0.50,
    m4_motif_amp_sdlog = 0.15,
    m4_motif_center_sd = 0.20,
    m4_motif_peak_sd = 0.10,
    m4_halfsep0 = 0.20,
    m4_halfsep_effect = 0.10
  )
}

validate_shared_cfg <- function(cfg) {
  if (!all(cfg$scenarios %in% unname(SCENARIOS))) stop("SCENARIOS must be a subset of the four final models M1--M4")
  if (cfg$n_train < 2L || cfg$n_test < 2L) stop("N_TRAIN and N_TEST must be >= 2")
  if (any(cfg$S_grid < 1L)) stop("S_GRID must contain positive integers")
  if (cfg$rff_dim %% 2L != 0L || cfg$rff_train_dim %% 2L != 0L) stop("RFF dimensions must be even")
  if (length(cfg$cnn_spectral_caps) != 3L || any(cfg$cnn_spectral_caps <= 0)) stop("Need three positive spectral caps")
  if (!identical(cfg$noise_model, "multiplicative")) stop("Final simulations use multiplicative noise only")
  invisible(cfg)
}
