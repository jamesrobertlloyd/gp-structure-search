function [varargout] = likT(hyp, y, mu, s2, inf, i)

% likT - Student's t likelihood function for regression. 
% The expression for the likelihood is
%   likT(t) = Z * ( 1 + (t-y)^2/(nu*sn^2) ).^(-(nu+1)/2),
% where Z = gamma((nu+1)/2) / (gamma(nu/2)*sqrt(nu*pi)*sn)
% and y is the mean (for nu>1) and nu*sn^2/(nu-2) is the variance (for nu>2).
%
% The hyperparameters are:
%
% hyp = [ log(nu-1)
%         log(sn)  ]
%
% Note that the parametrisation guarantees nu>1, thus the mean always exists.
%
% Several modes are provided, for computing likelihoods, derivatives and moments
% respectively, see likelihoods.m for the details. In general, care is taken
% to avoid numerical issues when the arguments are extreme. 
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-07-21.
%
% See also likFunctions.m.

if nargin<2, varargout = {'2'}; return; end   % report number of hyperparameters

numin = 1;                                                 % minimum value of nu
nu = exp(hyp(1))+numin; sn2 = exp(2*hyp(2));           % extract hyperparameters
lZ = loggamma(nu/2+1/2) - loggamma(nu/2) - log(nu*pi*sn2)/2;

if nargin<5                              % prediction mode if inf is not present
  if numel(y)==0,  y = zeros(size(mu)); end
  s2zero = 1; if nargin>3, if norm(s2)>0, s2zero = 0; end, end         % s2==0 ?
  if s2zero                                         % log probability evaluation
    lp = lZ - (nu+1)*log( 1+(y-mu).^2./(nu.*sn2) )/2;
  else                                                              % prediction
    lp = likT(hyp, y, mu, s2, 'infEP');
  end
  ymu = {}; ys2 = {};
  if nargout>1
    ymu = mu;                    % first y moment; for nu<=1 we this is the mode
    if nargout>2
      if nu<=2
        ys2 = Inf(size(s2));
      else
        ys2 = s2 + nu*sn2/(nu-2);                              % second y moment
      end
    end
  end
  varargout = {lp,ymu,ys2};
else
  switch inf 
  case 'infLaplace'
    r = y-mu; r2 = r.^2;
    if nargin<6                                             % no derivative mode
      dlp = {}; d2lp = {}; d3lp = {};
      lp = lZ - (nu+1)*log( 1+r2./(nu.*sn2) )/2;
      if nargout>1
        a = r2+nu*sn2;
        dlp = (nu+1)*r./a;                   % dlp, derivative of log likelihood
        if nargout>2                    % d2lp, 2nd derivative of log likelihood
          d2lp = (nu+1)*(r2-nu*sn2)./a.^2;
          if nargout>3                  % d3lp, 3rd derivative of log likelihood
            d3lp = (nu+1)*2*r.*(r2-3*nu*sn2)./a.^3;
          end
        end
      end
      varargout = {sum(lp),dlp,d2lp,d3lp};
    else                                                       % derivative mode
      a3 = (r2+nu*sn2).^3;
      if i==1                                             % derivative w.r.t. nu
        lp_dhyp =  nu*( dloggamma(nu/2+1/2)-dloggamma(nu/2) )/2 - 1/2 ...
                  -nu*log(1+r2/(nu*sn2))/2 +(nu/2+1/2)*r2./(nu*sn2+r2);
        lp_dhyp = (1-numin/nu)*lp_dhyp;          % correct for lower bound on nu
        d2lp_dhyp = nu*( r2.*(r2-3*sn2*(1+nu)) + nu*sn2^2 )./a3;
        d2lp_dhyp = (1-numin/nu)*d2lp_dhyp;      % correct for lower bound on nu
      else                                                % derivative w.r.t. sn
        lp_dhyp = (nu+1)*r2./(r2+nu*sn2) - 1; 
        d2lp_dhyp = (nu+1)*2*nu*sn2*(nu*sn2-3*r2)./a3;
      end
      varargout = {lp_dhyp,d2lp_dhyp};
    end

  case 'infEP'
    if nargout>1
      error('infEP not supported since likT is not log-concave')
    end
    n = max([length(y),length(mu),length(s2)]); on = ones(n,1);
    y = y(:).*on; mu = mu(:).*on; sig = sqrt(s2(:)).*on;          % vectors only
    % since we are not aware of an analytical expression of the integral, 
    % we use Gaussian-Hermite quadrature
    N = 20; [t,w] = gauher(N); oN = ones(1,N);
    lZ = likT(hyp, y*oN, sig*t'+mu*oN, []);
    lZ = log_expA_x(lZ,w); % log( exp(lZ)*w )
    varargout = {lZ};

  case 'infVB'
    if nargin<6
      % variational lower site bound
      % t(s) \propto (1+(s-y)^2/(nu*s2))^(-nu/2+1/2)
      % the bound has the form: b*s - s.^2/(2*ga) - h(ga)/2 with b=y/ga!!
      ga = s2; n = numel(ga); b = y./ga; y = y.*ones(n,1);
      db = -y./ga.^2; d2b = 2*y./ga.^3;
      id = ga<=sn2*nu/(nu+1);
      h   =  (nu+1)*( log(ga*(1+1/nu)/sn2) - 1 ) + (nu*sn2+y.^2)./ga;
      h(id) = y(id).^2./ga(id); h = h - 2*lZ;
      dh  =  (nu+1)./ga - (nu*sn2+y.^2)./ga.^2;
      dh(id) = -y(id).^2./ga(id).^2;
      d2h = -(nu+1)./ga.^2 + 2*(nu*sn2+y.^2)./ga.^3;
      d2h(id) = 2*y(id).^2./ga(id).^3;
      id = ga<0; h(id) = Inf; dh(id) = 0; d2h(id) = 0;     % neg. var. treatment
      varargout = {h,b,dh,db,d2h,d2b};
    else
      ga = s2; n = numel(ga); dhhyp = zeros(n,1);
      id = ga>sn2*nu/(nu+1); % dhhyp(~id) = 0
      if i==1 % log(nu)
        % h = (nu+1)*log(1+1/nu) - nu + nu*sn2./ga;
        dhhyp(id) = nu*log(ga(id)*(1+1/nu)/sn2) - 1 - nu + nu*sn2./ga(id);
        % lZ = loggamma(nu/2+1/2) - loggamma(nu/2) - log(nu)/2       
        dhhyp = dhhyp - nu*dloggamma(nu/2+1/2) + nu*dloggamma(nu/2) + 1; % -2*lZ
        dhhyp = (1-numin/nu)*dhhyp;              % correct for lower bound on nu
      else % log(sn)
        % h = (nu+1)*log(1/sn2) + nu*sn2./ga;
        dhhyp(id) = -2*(nu+1) + 2*nu*sn2./ga(id);
        % lZ = - log(sn2)/2
        dhhyp = dhhyp + 2; % -2*lZ
      end
      dhhyp(ga<0) = 0;              % negative variances get a special treatment
      varargout = {dhhyp};                                  % deriv. wrt hyp.lik
    end
  end
end

% Returns the log of the gamma function.  Source: Pike, M.C., and
% I.D. Hill, Algorithm 291, Communications of the ACM, 9,9:p.684 (Sept, 1966).
% Accuracy to 10 decimal places.  Uses Sterling's formula.
% Derivative from Abromowitz and Stegun, formula 6.3.18 with recurrence 6.3.5.
function f = loggamma(x)
  x = x+6;
  f = 1./(x.*x);
  f = (((-0.000595238095238*f+0.000793650793651).*f-1/360).*f+1/12)./x;
  f = (x-0.5).*log(x)-x+0.918938533204673+f;
  f = f-log(x-1)-log(x-2)-log(x-3)-log(x-4)-log(x-5)-log(x-6);

function df = dloggamma(x)
  x = x+6;
  df = 1./(x.*x);
  df = (((df/240-0.003968253986254).*df+1/120).*df-1/120).*df;
  df = df+log(x)-0.5./x-1./(x-1)-1./(x-2)-1./(x-3)-1./(x-4)-1./(x-5)-1./(x-6);

%  computes y = log( exp(A)*x ) in a numerically safe way by subtracting the
%  maximal value in each row to avoid cancelation after taking the exp
function y = log_expA_x(A,x)
  N = size(A,2);  maxA = max(A,[],2);      % number of columns, max over columns
  y = log(exp(A-maxA*ones(1,N))*x) + maxA;  % exp(A) = exp(A-max(A))*exp(max(A))

% compute abscissas and weight factors for Gaussian-Hermite quadrature
%
% CALL:  [x,w]=gauher(N)
%  
%  x = base points (abscissas)
%  w = weight factors
%  N = number of base points (abscissas) (integrates a (2N-1)th order
%      polynomial exactly)
%
%  p(x)=exp(-x^2/2)/sqrt(2*pi), a =-Inf, b = Inf 
%
%  The Gaussian Quadrature integrates a (2n-1)th order
%  polynomial exactly and the integral is of the form
%           b                         N
%          Int ( p(x)* F(x) ) dx  =  Sum ( w_j* F( x_j ) )
%           a                        j=1		          
%
%      this procedure uses the coefficients a(j), b(j) of the
%      recurrence relation
%
%           b p (x) = (x - a ) p   (x) - b   p   (x)
%            j j            j   j-1       j-1 j-2
%
%      for the various classical (normalized) orthogonal polynomials,
%      and the zero-th moment
%
%           1 = integral w(x) dx
%
%      of the given polynomial's weight function w(x).  Since the
%      polynomials are orthonormalized, the tridiagonal matrix is
%      guaranteed to be symmetric.
function [x,w]=gauher(N)
  if N==20 % return precalculated values
      x=[ -7.619048541679757;-6.510590157013656;-5.578738805893203;
          -4.734581334046057;-3.943967350657318;-3.18901481655339 ;
          -2.458663611172367;-1.745247320814127;-1.042945348802751;
          -0.346964157081356; 0.346964157081356; 1.042945348802751;
           1.745247320814127; 2.458663611172367; 3.18901481655339 ;
           3.943967350657316; 4.734581334046057; 5.578738805893202;
           6.510590157013653; 7.619048541679757];
      w=[  0.000000000000126; 0.000000000248206; 0.000000061274903;
           0.00000440212109 ; 0.000128826279962; 0.00183010313108 ;
           0.013997837447101; 0.061506372063977; 0.161739333984   ;
           0.260793063449555; 0.260793063449555; 0.161739333984   ;
           0.061506372063977; 0.013997837447101; 0.00183010313108 ;
           0.000128826279962; 0.00000440212109 ; 0.000000061274903;
           0.000000000248206; 0.000000000000126 ];
  else
      b = sqrt( (1:N-1)/2 )';    
      [V,D] = eig( diag(b,1) + diag(b,-1) );
      w = V(1,:)'.^2;
      x = sqrt(2)*diag(D);
  end
