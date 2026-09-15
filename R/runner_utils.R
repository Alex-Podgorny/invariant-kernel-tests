source_project <- function(root = ".") {
  source(file.path(root,"R","config.R")); source(file.path(root,"R","utils.R")); source(file.path(root,"R","models.R"))
  source(file.path(root,"R","methods","translation_methods.R")); source(file.path(root,"R","methods","circular_methods.R")); source(file.path(root,"R","methods","affine_methods.R"))
  source(file.path(root,"R","generators.R"))
  for (f in c("base_rff.R","energy_prob.R","align_energy_median.R","align_xcorr.R","circular_haar.R",
              "affine_wavelet_prob.R","affine_wavelet_max.R","learned_translation.R","learned_circular.R","learned_affine.R"))
    source(file.path(root,"R","methods",f))
}

make_result_row <- function(group, scenario, delta, rep_id, method, S, tst, train_seconds=0, eval_seconds=0) {
  data.frame(group=group,model=names(SCENARIOS)[match(scenario,SCENARIOS)],scenario=scenario,delta=delta,replicate=rep_id,
             method=method,S=as.integer(S),stat=tst$stat,pvalue=tst$pvalue,reject=tst$reject,
             train_seconds=train_seconds,eval_seconds=eval_seconds,total_seconds=train_seconds+eval_seconds,
             stringsAsFactors=FALSE)
}

parallel_jobs <- function(jobs, run_rep, cfg, root, label = "job") {
  worker <- function(i) run_rep(jobs$scenario[i], jobs$delta[i], jobs$replicate[i])
  if (cfg$cores <= 1L) {
    out <- vector("list", nrow(jobs))
    for (i in seq_len(nrow(jobs))) {
      cat(sprintf("[%s] %d/%d %s delta=%.1f rep=%d\n", label, i, nrow(jobs), jobs$scenario[i], jobs$delta[i], jobs$replicate[i]))
      out[[i]] <- worker(i)
    }
    return(out)
  }
  if (.Platform$OS.type != "windows") return(parallel::mclapply(seq_len(nrow(jobs)), worker, mc.cores = cfg$cores))
  cl <- parallel::makeCluster(cfg$cores); on.exit(parallel::stopCluster(cl), add = TRUE)
  parallel::clusterExport(cl, "root", envir = environment())
  parallel::clusterEvalQ(cl, {
    source(file.path(root,"R","runner_utils.R")); source_project(root); require_torch(); NULL
  })
  parallel::clusterExport(cl, c("jobs","cfg","run_rep"), envir = environment())
  parallel::parLapply(cl, seq_len(nrow(jobs)), function(i) run_rep(jobs$scenario[i], jobs$delta[i], jobs$replicate[i]))
}
