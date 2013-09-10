function [ nlZ ] = neg_pred_lik(hyp, mean_func, cov_func, lik_func, ...
                                 X_train, y_train, X_valid, y_valid)
%NEG_PRED_LIK Negative predictive likelihood
%   CAUTION - ignores mean function and likelihood function
%   James Lloyd, August 2013
  l = NaN(length(X_train), 1);
  for fold = 1:length(X_train)
    
    %%%% TODO - pay attention to mean function
    
    K = feval(cov_func{:}, hyp.cov, X_train{fold}) + ...
        exp(2*hyp.lik)*eye(length(y_train{fold}));
    %K = K + 1e-5*max(max(K))*eye(size(K));
    Ks = feval(cov_func{:}, hyp.cov, X_train{fold}, X_valid{fold});
    Kss = feval(cov_func{:}, hyp.cov, X_valid{fold});
    Kss = Kss + exp(2*hyp.lik)*eye(size(Kss));
    
    ymu = Ks' * (K \ y_train{fold});
    ys2 = Kss - Ks' * (K \ Ks);
    %ys2 = ys2 + 1e-5*max(max(ys2))*eye(size(ys2));
    Ls2 = chol(ys2);
    
    npll = 0.5 * length(y_valid{fold}) * log(2 * pi) + ...
           sum(log(diag(Ls2))) + ...
           0.5 * (y_valid{fold} - ymu)' * (ys2 \ (y_valid{fold} - ymu));
    
    l(fold) = npll;
  end
  nlZ = sum(l);
end

