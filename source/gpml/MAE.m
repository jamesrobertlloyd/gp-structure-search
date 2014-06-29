function [ MAE ] = MAE(hyp, mean_func, cov_func, lik_func, ...
                                 X_train, y_train, X_valid, y_valid)
%MAE Cross validated mean absolute error
%   CAUTION - ignores mean function and likelihood function
%   James Lloyd, August 2013
  e = NaN(length(X_train), 1);
  for fold = 1:length(X_train)
    
    %%%% TODO - pay attention to mean function
    
    K = feval(cov_func{:}, hyp.cov, X_train{fold}) + ...
        exp(2*hyp.lik)*eye(length(y_train{fold}));
    %K = K + 1e-5*max(max(K))*eye(size(K));
    Ks = feval(cov_func{:}, hyp.cov, X_train{fold}, X_valid{fold});
    
    ymu = Ks' * (K \ y_train{fold});
    
    e(fold) = mean(abs(y_valid{fold} - ymu));
  end
  MAE = mean(e);
end

