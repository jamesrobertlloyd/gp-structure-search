function K = covChange(hyp, x, z, i)

% 1D changepoint covariance function with steepness and center
% and input space shift. The covariance function is parameterized as:
%
% k(x,x') = sigmoid(sum(x - shift)*steepness)*sigmoid(sum(x - shift)*steepness)
%
% Sigmoid ranges from 0 to 1.
%
% The hyperparameters are:
%
% hyp = [ steepness
%         center    ]
%
% David Duvenaud
% February 2013


if nargin<2, K = '2'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0;
dg = strcmp(z,'diag') && numel(z)>0;                            % determine mode

[n,D] = size(x);
steepness = hyp(1);
center = hyp(2);

sumx = sum(x-center,2);
x = sumx*steepness;        % x is now one-dimensional
if xeqz                % This is the slow way.
    z = x;
else
    sumz = sum(z-center,2);
    z = sumz*steepness;
end 

sx = 1 ./ (1 + exp(-x));   % sigmoid
sz = 1 ./ (1 + exp(-z));   % sigmoid

if dg                                                               % vector kxx
    K = sx.*sx;
else
    K = sx*sz';
end

if nargin>3                                                        % covariances
  if i==1                                                          % alpha
    if dg                    % derivative of vector kxx w.r.t. alpha
      K = diag(K .* bsxfun(@plus,(1 - sx).*sumx, (1 - sz').*sumz'));
    else
      K = K .* bsxfun(@plus,(1 - sx).*sumx, (1 - sz').*sumz');
    end
  elseif i==2                                          % shifts
      % Todo: replace D with sum over centers
    K = -D*steepness * K .* ( 2 - bsxfun(@plus,sx, sz') );
  else
    error('Unknown hyperparameter')
  end
end
