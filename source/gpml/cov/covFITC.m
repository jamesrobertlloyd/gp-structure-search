function [K,Kuu,Ku] = covFITC(cov, xu, hyp, x, z, i)

% Covariance function to be used together with the FITC approximation.
%
% The function allows for more than one output argument and does not respect the
% interface of a proper covariance function. In fact, it wraps a proper
% covariance function such that it can be used together with infFITC.m.
% Instead of outputing the full covariance, it returns cross-covariances between
% the inputs x, z and the inducing inputs xu as needed by infFITC.m
%
% Copyright (c) by Ed Snelson, Carl Edward Rasmussen 
%                                               and Hannes Nickisch, 2010-12-21.
%
% See also COVFUNCTIONS.M, INFFITC.M.

if nargin<4,  K = feval(cov{:}); return, end
if nargin<5, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

if size(xu,2) ~= size(x,2)
  error('Dimensionality of inducing inputs must match training inputs');
end

if nargin<6                                                        % covariances
  if dg
    K = feval(cov{:},hyp,x,'diag');
  else
    if xeqz
        K   = feval(cov{:},hyp,x,'diag');
        Kuu = feval(cov{:},hyp,xu);
        Ku  = feval(cov{:},hyp,xu,x);
    else
      K = feval(cov{:},hyp,xu,z);
    end
  end
else                                                               % derivatives
  if dg
    K = feval(cov{:},hyp,x,'diag',i);
  else
    if xeqz
        K   = feval(cov{:},hyp,x,'diag',i);
        Kuu = feval(cov{:},hyp,xu,[],i);
        Ku  = feval(cov{:},hyp,xu,x,i);
    else
      K = feval(cov{:},hyp,xu,z,i);
    end
  end
end