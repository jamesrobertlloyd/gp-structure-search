%% Load data

load 'r_concrete_500_fold_10_of_10.mat'

%% Load GPML

%%%% CHANGE ME ON DIFFERENT MACHINES
addpath(genpath('/home/jrl44/Documents/MATLAB/GPML'));

%% Set up model structure and previously learned parameters

meanfunc = {@meanConst};
hyp.mean = mean(y);

covfunc = {@covProd, {{@covMask, {[1 0 0 0 0 0 0 0], {@covPoly, 3}}}, {@covMask, {[0 0 0 0 0 0 0 1], @covPeriodic}}}};
hyp.cov = [ 1.757755 7.084045 -2.70108 -0.380918 -0.071214 ];

likfunc = @likGauss;
hyp.lik = [-1.77276072];

[hyp_opt, nlls] = minimize(hyp, @gp, +10000000, @infExact, meanfunc, covfunc, likfunc, X, y);
best_nll = nlls(end);

%% Plot minimum found in dimension 4

my_range = (-100):100;
values = zeros(length(my_range),1);
j = 1;
delta = 1e-4;
d = 4;
for i = my_range
    dhyp_opt = hyp_opt;
    dhyp_opt.cov(d) = dhyp_opt.cov(d) + i*delta;
    [nll_delta, dnll_delta] = gp(dhyp_opt, @infExact, meanfunc, covfunc, likfunc, X, y);
    values(j) = nll_delta;
    j = j + 1;
    i
end
figure;
plot(my_range, values);

%% Plot the lack of minimum in dimension 1

my_range = (-100):100;
values = zeros(length(my_range),1);
j = 1;
delta = 1e-4;
d = 1;
for i = my_range
    dhyp_opt = hyp_opt;
    dhyp_opt.cov(d) = dhyp_opt.cov(d) + i*delta;
    [nll_delta, dnll_delta] = gp(dhyp_opt, @infExact, meanfunc, covfunc, likfunc, X, y);
    values(j) = nll_delta;
    j = j + 1;
    i
end
figure;
plot(my_range, values);

%% Plot the lack of minimum in dimension 1 - zooming out

my_range = (-100):100;
values = zeros(length(my_range),1);
j = 1;
delta = 1e-2;
d = 1;
for i = my_range
    dhyp_opt = hyp_opt;
    dhyp_opt.cov(d) = dhyp_opt.cov(d) + i*delta;
    [nll_delta, dnll_delta] = gp(dhyp_opt, @infExact, meanfunc, covfunc, likfunc, X, y);
    values(j) = nll_delta;
    j = j + 1;
    i
end
figure;
plot(my_range, values);

%% Rerun the optimiser

[hyp_opt, nlls] = minimize(hyp_opt, @gp, +1000000, @infExact, meanfunc, covfunc, likfunc, X, y);
