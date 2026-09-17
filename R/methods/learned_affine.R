method_fit_learned_affine <- function(Xtr,Ytr,cfg,rff_train,seed) fit_affine_cnn2d(Xtr,Ytr,cfg,rff_train,seed)
method_eval_learned_affine <- function(Z,S,cfg,rff,fit,seed) affine_prob_features(Z,S,cfg,rff,"cnn",fit$model,seed)
