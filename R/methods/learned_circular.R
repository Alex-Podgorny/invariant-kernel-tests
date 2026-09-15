method_fit_learned_circular <- function(Xtr,Ytr,cfg,rff_train,seed) fit_circular_cnn(Xtr,Ytr,cfg,rff_train,seed)
method_eval_learned_circular <- function(Z,S,cfg,rff,fit,seed) circular_prob_features(Z,S,cfg,rff,"cnn",fit$model,seed)
