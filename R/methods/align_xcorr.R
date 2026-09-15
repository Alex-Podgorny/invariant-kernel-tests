method_align_xcorr_translation <- function(Z,cfg) align_xcorr_translation(Z,pooled_medoid(Z,cfg$dt),cfg)
method_align_xcorr_circular <- function(Z,cfg) align_xcorr_circular(Z,pooled_medoid_circular(Z,cfg),cfg)
method_align_xcorr_affine <- function(Z,cfg) align_xcorr_affine(Z,pooled_medoid_affine(Z,cfg),cfg)
