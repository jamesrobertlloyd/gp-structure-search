function test_covChange()
% A simple script to check the derivatives of covChange.
%
% David Duvenaud
% Feb 2013

% The hyperparameters are:
%
% hyp = [ log(sqrt(sf2)
%         log(ell)
%         shifts ]

addpath(genpath('/scratch/Dropbox/code/gpml/'))

n = 3;
D = 1;
x = [0; -1; 2];%randn(n,1);
z = [];%randn(n,1);

covfunc = @covChange;
covwrap = @(hypers) covfunc(hypers, x, z);
delta = 0.00001;

num_hypers = eval(covfunc());
hypers = (1:num_hypers)./10%randn(1, num_hypers);
%hypers(3) = 0;
%hypers(2) = 0;
% Check derivatives w.r.t. each input in turn.
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
