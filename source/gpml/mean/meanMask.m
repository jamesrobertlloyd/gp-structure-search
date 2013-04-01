function A = meanMask(mean, hyp, x, i)

% Apply a mean function to a subset of the dimensions only. The subset can
% either be specified by a 0/1 mask by a boolean mask or by an index set.
%
% This function doesn't actually compute very much on its own, it merely does
% some bookkeeping, and calls another mean function to do the actual work.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-07-15.
%
% See also MEANFUNCTIONS.M.

mask = mean{1}(:);                        % either a binary mask or an index set
mean = mean(2);                                     % mean function to be masked
nh_string = feval(mean{:});         % number of hyperparameters of the full mean

if max(mask)<2 && length(mask)>1, mask = boolean(mask); end   % convert 1/0->T/F
if islogical(mask), D = sum(mask); else D = length(mask); end % masked dimension
if nargin<3, A = num2str(eval(nh_string)); return; end    % number of parameters

if eval(nh_string)~=length(hyp)                          % check hyperparameters
  error('number of hyperparameters does not match size of masked data')
end

if nargin==3
  A = feval(mean{:}, hyp, x(:,mask));
else                                                        % compute derivative
  A = feval(mean{:}, hyp, x(:,mask), i);
end