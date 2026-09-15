source("R/check_environment.R")
check_environment()

args0 <- commandArgs(trailingOnly=FALSE); ff <- sub("^--file=","",args0[grep("^--file=",args0)]); root <- if(length(ff)) normalizePath(file.path(dirname(ff[1]),"..")) else normalizePath(".")

for (script in c("run_translation.R","run_circular.R","run_affine.R")) {
  status <- system2(file.path(R.home("bin"),"Rscript"), file.path(root,"run",script))
  if (status != 0) stop("Failed: ", script)
}
