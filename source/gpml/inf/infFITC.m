function [post nlZ dnlZ] = infFITC(hyp, mean, cov, lik, x, y)

% FITC approximation to the posterior Gaussian process. The function is
% equivalent to infExact with the covariance function:
%   Kt = Q + G;   G = diag(diag(K-Q);   Q = Ku'*inv(Kuu + snu2*eye(nu))*Ku;
% where Ku and Kuu are covariances w.r.t. to inducing inputs xu and
% snu2 = sn2/1e6 is the noise of the inducing inputs. We fixed the standard
% deviation of the inducing inputs to be a one per mil of the measurement noise
% standard deviation.
% The implementation exploits the Woodbury matrix identity
%   inv(Kt) = inv(G) - inv(G)*Ku'*inv(Kuu+Ku*inv(G)*Ku')*Ku*inv(G)
% in order to be applicable to large datasets. The computational complexity
% is O(n nu^2) where n is the number of data points x and nu the number of
% inducing inputs in xu.
% The function takes a specified covariance function (see covFunction.m) and
% likelihood function (see likFunction.m), and is designed to be used with
% gp.m and in conjunction with covFITC and likGauss. 
%
% Copyright (c) by Ed Snelson, Carl Edward Rasmussen 
%                                               and Hannes Nickisch, 2010-11-26.
%
% See also INFMETHODS.M, COVFITC.M.

likstr = lik; if ~ischar(lik), likstr = func2str(lik); end 
if ~strcmp(likstr,'likGauss')               % NOTE: no explicit call to likGauss
  error('FITC inference only possible with Gaussian likelihood');
end
cov1 = cov{1}; if isa(cov1, 'function_handle'), cov1 = func2str(cov1); end
if ~strcmp(cov1,'covFITC'); error('Only covFITC supported.'), end    % check cov

[diagK,Kuu,Ku] = feval(cov{:}, hyp.cov, x);         % evaluate covariance matrix
m = feval(mean{:}, hyp.mean, x);                          % evaluate mean vector
[n, D] = size(x); nu = size(Kuu,1);

sn2  = exp(2*hyp.lik);                              % noise variance of likGauss
snu2 = 1e-6*sn2;                              % hard coded inducing inputs noise
Luu  = chol(Kuu + snu2*eye(nu));                       % Kuu + snu2*I = Luu'*Luu
V  = Luu'\Ku;                                     % V = inv(Luu')*Ku => V'*V = Q
dg = diagK + sn2 - sum(V.*V,1)';      % D + sn2*eye(n) = diag(K) + sn2 - diag(Q)
V  = V./repmat(sqrt(dg)',nu,1);
Lu = chol(eye(nu) + V*V');
r  = (y-m)./sqrt(dg);
be = Lu'\(V*r);
iKuu = solve_chol(Luu,eye(nu));                       % inv(Kuu + snu2*I) = iKuu
post.alpha = Luu\(Lu\be);                      % return the posterior parameters
post.L  = solve_chol(Lu*Luu,eye(nu)) - iKuu; % Sigma-inv(Kuu)
post.sW = [];                                                  % unused for FITC

if nargout>1                                % do we want the marginal likelihood
  nlZ = sum(log(diag(Lu))) + (sum(log(dg)) + n*log(2*pi) + r'*r - be'*be)/2; 
  if nargout>2                                         % do we want derivatives?
    dnlZ = hyp;                                 % allocate space for derivatives
    W = Ku./repmat(sqrt(dg)',nu,1); 
    W = chol(Kuu+W*W'+snu2*eye(nu))'\Ku; % inv(K) = inv(G) - inv(G)*W'*W*inv(G);  
    al = (y-m - W'*(W*((y-m)./dg)))./dg;
    B = iKuu*Ku; Wdg = W./repmat(dg',nu,1); w = B*al;
    for i = 1:numel(hyp.cov)
      [ddiagKi,dKuui,dKui] = feval(cov{:}, hyp.cov, x, [], i);  % eval cov deriv
      R = 2*dKui-dKuui*B; v = ddiagKi - sum(R.*B,1)';   % diag part of cov deriv
      dnlZ.cov(i) = (ddiagKi'*(1./dg) +w'*(dKuui*w-2*(dKui*al)) -al'*(v.*al) ...
                         - sum(Wdg.*Wdg,1)*v - sum(sum((R*Wdg').*(B*Wdg'))) )/2;
    end  
    dnlZ.lik = sn2*(sum(1./dg) -sum(sum(W.*W,1)'./(dg.*dg)) -al'*al);
    % since snu2 is a fixed fraction of sn2, there is a covariance-like term in
    % the derivative as well
    dKuui = 2*snu2; R = -dKuui*B; v = -sum(R.*B,1)';   % diag part of cov deriv
    dnlZ.lik = dnlZ.lik + (w'*dKuui*w -al'*(v.*al)...
                         - sum(Wdg.*Wdg,1)*v - sum(sum((R*Wdg').*(B*Wdg'))) )/2; 
    for i = 1:numel(hyp.mean)
      dnlZ.mean(i) = -feval(mean{:}, hyp.mean, x, i)'*al;
    end
  end
end