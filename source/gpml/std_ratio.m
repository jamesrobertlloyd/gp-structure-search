function [ std_r ] = std_ratio(hyp, mean_func, cov_func, lik_func, ...
                               X, y)
%STD_RATIO Empirical std divided by fitted std
%   
%   James Lloyd, August 2013

  %ymu = gp(hyp, @infExact, mean_func, cov_func, lik_func, X, y, X);

  K = feval(cov_func{:}, hyp.cov, X) + ...
      exp(2*hyp.lik)*eye(length(y));
  Ks = feval(cov_func{:}, hyp.cov, X);
    
  ymu = Ks' * (K \ y);
  
  resid = y - ymu;
  std_r = std(resid) / exp(hyp.lik);
end

