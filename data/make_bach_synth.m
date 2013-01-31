% demo script for regression with HKL
clear all


% fixing the seed of the random generators
seed=1;
randn('state',seed);
rand('state',seed);

% toy example characteristics
p = 1024;           % total number of variables (used to generate a Wishart distribution)
psub = 8;          % kept number of variables = dimension of the problem
n = 200;            % number of observations
s = 4;              % number of relevant variables
noise_std = .2;		% standard deviation of noise
proptrain = .5;     % proportion of data kept for training (the rest is used for testing)


% generate random covariance matrix from a Wishart distribution
Sigma_sqrt = randn(p,p);
Sigma = Sigma_sqrt' * Sigma_sqrt;


% normalize to unit trace and sample
diagonal = diag(Sigma);
Sigma = diag( 1./diagonal.^.5) * Sigma * diag( 1./diagonal.^.5);
Sigma_sqrt =   Sigma_sqrt * diag( 1./diagonal.^.5);
X = randn(n,p) * Sigma_sqrt;

X = X(:,1:psub);
p=psub;

% generate nonlinear function of X as the sum of all cross-products
J =  1:s;    % select the first s variables
Y = zeros(n,1);
for i=1:s
	for j=1:i-1
		Y = Y + X(:,J(i)) .* X(:,J(j));
	end
end
% normalize to unit standard deviation
Y = Y / std(Y);

% add some noise with known standard deviation
Y =  Y + randn(n,1) * noise_std;

y = Y;
save('bach_synth_r_200.mat', 'X', 'y' );

X = X(1:100,:);
y = y(1:100,:);
save('bach_synth_r_100.mat', 'X', 'y' );