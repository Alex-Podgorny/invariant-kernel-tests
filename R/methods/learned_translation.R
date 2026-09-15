method_fit_learned_translation <- function(Xtr,Ytr,cfg,rff_train,seed) fit_translation_cnn(Xtr,Ytr,cfg,rff_train,seed)
method_eval_learned_translation <- function(Z,S,cfg,rff,fit,seed) translation_prob_features(Z,S,cfg,rff,"cnn",fit$model,seed)
