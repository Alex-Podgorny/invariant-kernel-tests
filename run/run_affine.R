source("R/check_environment.R")
check_environment()

args0 <- commandArgs(trailingOnly=FALSE); ff <- sub("^--file=","",args0[grep("^--file=",args0)]); root <- if(length(ff)) normalizePath(file.path(dirname(ff[1]),"..")) else normalizePath(".")

source(file.path(root,"R","runner_utils.R")); source_project(root); require_torch()

shared<-make_shared_cfg();validate_shared_cfg(shared);cfg<-make_affine_cfg(shared)

outdir<-Sys.getenv("OUTDIR",unset=file.path(root,"results","affine"));dir.create(outdir,recursive=TRUE,showWarnings=FALSE)

run_rep<-function(scenario,delta,rep_id){
  seed<-mix_seed(cfg$seed,mult=1000003,add=rep_id+1009*match(scenario,cfg$scenarios)+100000*round(delta*10));set.seed(seed)
  dat<-generate_affine_split(scenario,delta,cfg);XA<-dat$Xtr;YA<-dat$Ytr;XB<-dat$Xte;YB<-dat$Yte;ZB<-rbind(XB,YB);rows<-list();k<-0L
  ttrain<-proc.time()[3];ZAc<-canonicalize_affine(rbind(XA,YA),cfg);sigmaA<-median_bandwidth(embed_obs_to_rep(ZAc,cfg),cfg$dt);rffTr<-make_rff_map(cfg$p_rep,sigmaA,cfg$dt,cfg$rff_train_dim,mix_seed(seed,add=11));rffTe<-make_rff_map(cfg$p_rep,sigmaA,cfg$dt,cfg$rff_dim,mix_seed(seed,add=cfg$rff_test_seed_gap));fit<-method_fit_learned_affine(XA,YA,cfg,rffTr,mix_seed(seed,add=271828));train_time<-proc.time()[3]-ttrain
  planB<-make_perm_plan(2*cfg$n_test,cfg$n_test,cfg$Bperm,mix_seed(seed,add=4001))
  for(S in cfg$S_grid){te<-proc.time()[3];F<-method_eval_learned_affine(ZB,S,cfg,rffTe,fit,mix_seed(seed,add=50000+S));tst<-perm_test_features(F,cfg$n_test,cfg$alpha_test,planB);k<-k+1L;rows[[k]]<-make_result_row("Aff",scenario,delta,rep_id,"Learned-Affine-GCNN-RFF",S,tst,train_time,proc.time()[3]-te)}
  Xc<-rbind(XA,XB);Yc<-rbind(YA,YB);Zc<-rbind(Xc,Yc);nc<-nrow(Xc);planC<-make_perm_plan(2*nc,nc,cfg$Bperm,mix_seed(seed,add=7001))
  # Base raw curves.
  te<-proc.time()[3];sig<-median_bandwidth(Zc,cfg$dt);rffRaw<-make_rff_map(cfg$p_obs,sig,cfg$dt,cfg$rff_dim,mix_seed(seed,add=101));F<-method_base_rff(Zc,rffRaw,cfg);tst<-perm_test_features(F,nc,cfg$alpha_test,planC);k<-k+1L;rows[[k]]<-make_result_row("Aff",scenario,delta,rep_id,"Base-RFF",0,tst,0,proc.time()[3]-te)
  # Deterministic affine alignments, then bandwidth on aligned representative canvas.
  for(m in c("wavelet","xcorr")){te<-proc.time()[3];Za<-if(m=="wavelet")method_affine_wavelet_max(Zc,cfg) else method_align_xcorr_affine(Zc,cfg);sig<-median_bandwidth(Za,cfg$dt);rff<-make_rff_map(cfg$p_rep,sig,cfg$dt,cfg$rff_dim,mix_seed(seed,add=if(m=="wavelet")202 else 303));F<-rff_features(Za,rff,cfg$rff_chunk_size);tst<-perm_test_features(F,nc,cfg$alpha_test,planC);name<-if(m=="wavelet")"Align-wavelet-max-affine" else "Align-xcorr-affine";k<-k+1L;rows[[k]]<-make_result_row("Aff",scenario,delta,rep_id,name,0,tst,0,proc.time()[3]-te)}
  # Fixed wavelet sampler. Median bandwidth on canonical curves embedded in representative canvas.
  tp<-proc.time()[3];Zcc<-canonicalize_affine(Zc,cfg);sig<-median_bandwidth(embed_obs_to_rep(Zcc,cfg),cfg$dt);rffP<-make_rff_map(cfg$p_rep,sig,cfg$dt,cfg$rff_dim,mix_seed(seed,add=404));psetup<-proc.time()[3]-tp
  for(S in cfg$S_grid){te<-proc.time()[3];F<-method_affine_wavelet_prob(Zc,S,cfg,rffP,mix_seed(seed,add=80000+S));tst<-perm_test_features(F,nc,cfg$alpha_test,planC);k<-k+1L;rows[[k]]<-make_result_row("Aff",scenario,delta,rep_id,"Affine-MovingChart-Wavelet-energy-prob",S,tst,0,psetup+proc.time()[3]-te)}
  rows
}

jobs<-expand.grid(scenario=cfg$scenarios,delta=cfg$deltas,replicate=seq_len(cfg$nrep),stringsAsFactors=FALSE)
all<-parallel_jobs(jobs,run_rep,cfg,root,"Aff")
write_results(unlist(all,recursive=FALSE),cfg,outdir);cat("Wrote",outdir,"\n")
