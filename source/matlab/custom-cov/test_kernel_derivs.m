function test_cov_derivs()
% A simple script to check the derivatives of a covariance function.
%
% David Duvenaud
% Feb 2013

addpath(genpath('/scratch/Dropbox/code/gpml/'))

n = 3;
D = 2;
x = randn(n,D);
z = randn(n,D);

covfunc = @covChange;
covwrap = @(hypers) covfunc(hypers, x, z);
delta = 0.00001;

num_hypers = eval(covfunc());
hypers = randn(1, num_hypers);

% Check derivatives w.r.t. each input in turn.  Should print all 1's
for i = 1:num_hypers
    ng = numerical_grad( covwrap, hypers, i, delta );
    tg = covfunc(hypers, x, z, i);
    ng./tg
end

end

function ng = numerical_grad( f, x, i, delta )
% Takes a function of x, and computes gradient w.r.t. dimension i.
    x2 = x;
    x2(i) = x2(i) + delta;
    ng = (f(x2) - f(x))./delta;
end
