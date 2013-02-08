function K = covChange(hyp, x, z, i)

% Changepoint covariance function with scale and input space shift. The
% covariance function is parameterized as:
%
% k(x,x') = sf2 * sigmoid(sum((x - shift)/ell))*sigmoid(sum((x - shift)/ell))
%
% The hyperparameters are:
%
% hyp = [ log(sqrt(sf2)
%         log(ell)
%         shifts ]
%
% David Duvenaud
% February 2013


if nargin<2, K = '1 + 2*D'; return; end            % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0;
dg = strcmp(z,'diag') && numel(z)>0;                            % determine mode

[n,D] = size(x);
sf2 = exp(hyp(1));
ell = exp(hyp(2:D+1));
shifts = hyp((D+2):(2*D+1));

x = sum((x-repmat(shifts',n,1))*diag(1./ell));        % x is now one-dimensional
z = sum((z-repmat(shifts',n,1))*diag(1./ell));        % z is now one-dimensional

sx = 1 ./ (1 + exp(-x));   % sigmoid
sz = 1 ./ (1 + exp(-z));   % sigmoid

if dg                                                               % vector kxx
  K = sx.*sx;
else
  if xeqz                                                 % symmetric matrix Kxx
    K = x*x';
  else                                                   % cross covariances Kxz
    K = x*z';
  end
end

if nargin<4                                                        % covariances
  K = sf2*K;
else                                                               % derivatives
  if i==1                                                          % magnitude
    V = K;
  elseif i<=D+1  % ell
    i = i - D;
    if dg  % derivative of vector kxx w.r.t. lengthscales
      V = sf2 * -2*K.*x./ell(i);
    else
      V = sf2 * -2*K.*x./ell(i);
    end
  elseif i<=2*D+1                                             % shifts
    i = i - D*2;
    K = sf2*K./ell(i);
  else
    error('Unknown hyperparameter')
  end
end
