function K = covChange(hyp, x, z, i)

% 1D changepoint covariance function with inverse smoothness alpha,
% and input space shift. The covariance function is parameterized as:
%
% k(x,x') = sigmoid(sum((x - shift)*alpha))*sigmoid(sum((x - shift)*alpha))
%
% Sigmoid ranges from 0 to 1.
%
% The hyperparameters are:
%
% hyp = [ alpha
%         shift ]
%
% David Duvenaud
% February 2013


if nargin<2, K = '2'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0;
dg = strcmp(z,'diag') && numel(z)>0;                            % determine mode

[n,D] = size(x);
alpha = hyp(1);
shift = hyp(2);

sumx = sum(x-shift,2);
x = sumx*alpha;        % x is now one-dimensional
if xeqz                % This is the slow way.
    z = x;
else
    sumz = sum(z-shift,2);
    z = sumz*alpha;
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
      % Todo: replace D with sum over shifts
    K = -D*alpha * K .* ( 2 - bsxfun(@plus,sx, sz') );
  else
    error('Unknown hyperparameter')
  end
end
