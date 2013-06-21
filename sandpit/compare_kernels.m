
a='Load the data, it should contain X and y.'
load 'tmpWqk4yD.mat'

%Load GPML
addpath(genpath('/Users/JamesLloyd/Documents/MATLAB/GPML/gpml-matlab-v3.1-2010-09-27'));

%Set up model.
meanfunc = {@meanConst}
hyp.mean = mean(y)

likfunc = @likGauss
hyp.lik = -2.30080756849

% Create list of covariance functions
covs{1} = {@covMask, {[0 0 0 1], @covSEiso}}
hyps{1} = [ 0.0 0.0 ]
covs{2} = {@covMask, {[0 0 1 0], @covSEiso}}
hyps{2} = [ 0.0 0.0 ]
covs{3} = {@covMask, {[0 1 0 0], @covSEiso}}
hyps{3} = [ 0.0 0.0 ]
covs{4} = {@covMask, {[1 0 0 0], @covSEiso}}
hyps{4} = [ 0.0 0.0 ]
covs{5} = {@covMask, {[1 0 0 1], @covSEiso}}
hyps{5} = [ 0.0 0.0 ]
covs{6} = {@covMask, {[1 0 0 1], @covSEiso}}
hyps{6} = [ 0.0 0.001 ]
covs{7} = {@covMask, {[1 0 0 1], @covSEiso}}
hyps{7} = [ 0.1 0.0 ]
covs{8} = {@covMask, {[1 0 0 1], @covSEiso}}
hyps{8} = [ 0.1 0.1 ]
covs{9} = {@covMask, {[1 0 0 0], @covSEiso}}
hyps{9} = [ 0.1 0.1 ]

% Evaluate similarities
n_kernels = length(covs);
sim_matrix = zeros(n_kernels);
% Note - quadratic time algorithm to avoid huge memory requirements
for i = 1:n_kernels
  fprintf('Evaluating kernel %d of %d.\n', i, n_kernels);
  cov_i = feval(covs{i}{:}, hyps{i}, X);
  for j = (i+1):n_kernels
    cov_j = feval(covs{j}{:}, hyps{j}, X);
    % Compute Frobenius norm
    sq_diff = (cov_i - cov_j) .^ 2;
    frobenius = sqrt(sum(sq_diff(:)));
    % Put in sim matrix
    sim_matrix(i, j) = frobenius;
  end
end
% Make symmetric
sim_matrix = sim_matrix + sim_matrix';

save( 'temp.mat', 'sim_matrix' );

imagesc(sim_matrix)

fprintf('\nGoodbye, World\n');
