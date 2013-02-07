function K = covLINscaleshift(hyp, x, z, i)

% Linear covariance function with scale and input space shift. The
% covariance function is parameterized as:
%
% k(x^p,x^q) = (x^p - shift)'*inv(P)*(x^q - shift)
%
% where the P matrix is diagonal with parameters ell_1^2,
% The hyperparameters are:
%
% hyp = [ log(ell_1)
%         shift      ]
%
% Note that there is no bias term; use covConst to add a bias.
%
% Based on covLINard by
% Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
% Adapted by James Robert Lloyd, 2013
%
% N.B. Assumes 1-d input - although some of the code will work with multi-d
%
% See also COVFUNCTIONS.M.

if nargin<2, K = '2*D'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[n,D] = size(x);
ell = exp(hyp(1:D));
shifts = hyp((D+1):(2*D));
x_old = x;
x = (x-repmat(shifts',n,1))*diag(1./ell);

% precompute inner products
if dg                                                               % vector kxx
  K = sum(x.*x,2);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = x*x';
  else                                                   % cross covariances Kxz
    [nz,~] = size(z); 
    z_old = z;
    z = (z-repmat(shifts',nz,1))*diag(1./ell);
    K = x*z';
  end
end

if nargin>3                                                        % derivatives
  if i<=D
    if dg
      K = -2*x(:,i).*x(:,i);
    else
      if xeqz
        K = -2*x(:,i)*x(:,i)';
      else
        K = -2*x(:,i)*z(:,i)';
      end
    end
  elseif i <= 2*D
    %%%% Not tested with D > 1
    %%%% Also really ugly!
    if dg
      K = (-2 * x_old(:,i-D) + 2 * shifts(i-D)) / (ell * ell);
    else
      if xeqz
        K = (- (repmat(x_old(:,i-D), 1, n) + repmat(x_old(:,i-D)', n, 1)) + 2 * shifts(i-D)) / (ell * ell);
      else
        K = (- (repmat(x_old(:,i-D), 1, length(z_old(:,i-D))) + repmat(z_old(:,i-D)', n, 1)) + 2 * shifts(i-D)) /(ell * ell);
      end
    end
  else
    error('Unknown hyperparameter')
  end
end