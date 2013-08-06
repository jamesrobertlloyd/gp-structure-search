%% Standard plotting code

load 'fur-sales-mink-h-b-co-18481911.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('../../source/gpml/'));
addpath(genpath('../../source/matlab'));

plot_decomp(X, y, {@covProd, {{@covMask, {[1], {@covRQiso}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covPeriodic}}}}}, [ -1.42263 7.012301 -5.254423 -2.451057 -2.337529 -0.001647 0.002361 -0.000449 0.942026 2.259133 8.583919 ], { {@covProd, {{@covMask, {[1], {@covRQiso}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covPeriodic}}}}} }, { [ -1.42263 7.012301 -5.254423 -2.451057 -2.337529 -0.001647 0.002361 -0.000449 0.942026 2.259133 8.583919 ] }, [5.4321688], 'dummy', { ' RQ_{0} \times CS_{0} \times CS_{0} \times CS_{0} \times CS_{0} \times CS_{0} \times Per_{0} ' }, { 'RQ_{0} \times CS_{0} \times CS_{0} \times CS_{0} \times CS_{0} \times CS_{0} \times Per_{0}' }, 0.000000, 1.000000, 0.000000, 1.000000)

%% Unpacked

load 'fur-sales-mink-h-b-co-18481911.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('../../source/gpml/'));
addpath(genpath('../../source/matlab'));

complete_covfunc = {@covProd, {{@covMask, {[1], {@covRQiso}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covPeriodic}}}}};

complete_hypers = [ -1.42263 7.012301 -5.254423 -2.451057 -2.337529 -0.001647 0.002361 -0.000449 0.942026 2.259133 8.583919 ];
hyp.cov = complete_hypers;

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = [5.4321688];
log_noise = hyp.lik;

[nlZ, dnlZ] = gp(hyp, @infExact, mean_func, complete_covfunc, lik_func, X, y);
nlZ

%%%% TESTME - this might have been causing problems
%y = y - mean(y);

left_extend = 0.4;  % What proportion to extend beyond the data range.
right_extend = 0.4;

num_interpolation_points = 2000;

x_left = min(X) - (max(X) - min(X))*left_extend;
x_right = max(X) + (max(X) - min(X))*right_extend;
xrange = linspace(x_left, x_right, num_interpolation_points)';

noise_var = exp(2*log_noise);
complete_sigma = feval(complete_covfunc{:}, complete_hypers, X, X) + eye(length(y)).*noise_var;
complete_sigmastar = feval(complete_covfunc{:}, complete_hypers, X, xrange);
complete_sigmastarstart = feval(complete_covfunc{:}, complete_hypers, xrange, xrange);

% First, plot the original, combined kernel
%complete_mean = complete_sigmastar' / complete_sigma * y;
complete_mean = complete_sigmastar' * (complete_sigma \ y); % The same?
%complete_var = diag(complete_sigmastarstart - complete_sigmastar' / complete_sigma * complete_sigmastar);
complete_var = diag(complete_sigmastarstart - complete_sigmastar' * (complete_sigma \ complete_sigmastar)); % The same?
    
h = figure(1); clf; hold on;
plot(X, y, 'ko');
plot(xrange, complete_mean, 'b', 'LineWidth', 2);
%plot(xrange, complete_mean + 2.*sqrt(complete_var), 'g');
%plot(xrange, complete_mean - 2.*sqrt(complete_var), 'g');
hold off;
title(['alpha ', num2str(hyp.cov(3)), ' nlZ ', num2str(nlZ)]);
save2pdf(['RQ_', num2str(hyp.cov(3)), '_', num2str(nlZ), '.pdf'], ...
         h, 600);

% Plot residuals.
%data_complete_mean = feval(complete_covfunc{:}, complete_hypers, X, X)' / complete_sigma * y;
%figure(2); clf; hold on;
%plot(X, y - data_complete_mean, 'ko');
%plot(xrange, +2.*sqrt(noise_var).*ones(size(xrange)), 'g');
%%plot(xrange, -2.*sqrt(noise_var).*ones(size(xrange)), 'g');
hold off;
%close all;

%% Optimise kernel directly

load 'fur-sales-mink-h-b-co-18481911.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('../../source/gpml/'));
addpath(genpath('../../source/matlab'));

complete_covfunc = {@covProd, {@covRQiso, @covPeriodic}};

complete_hypers = [ +5.4, 7, -5, 0.94, 2.26, 0 ];
hyp.cov = complete_hypers;

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = 9;
log_noise = hyp.lik;

hyp_opt = minimize(hyp, @gp, -1000, @infExact, ...
                   mean_func, complete_covfunc, lik_func, X, y);
[nlZ, dnlZ] = gp(hyp_opt, @infExact, mean_func, complete_covfunc, lik_func, X, y);
nlZ

complete_hypers = hyp_opt.cov;
log_noise = hyp_opt.lik;

%%%% TESTME - this might have been causing problems
%y = y - mean(y);

left_extend = 0.4;  % What proportion to extend beyond the data range.
right_extend = 0.4;

num_interpolation_points = 2000;

x_left = min(X) - (max(X) - min(X))*left_extend;
x_right = max(X) + (max(X) - min(X))*right_extend;
xrange = linspace(x_left, x_right, num_interpolation_points)';

noise_var = exp(2*log_noise);
complete_sigma = feval(complete_covfunc{:}, complete_hypers, X, X) + eye(length(y)).*noise_var;
complete_sigmastar = feval(complete_covfunc{:}, complete_hypers, X, xrange);
complete_sigmastarstart = feval(complete_covfunc{:}, complete_hypers, xrange, xrange);

% First, plot the original, combined kernel
%complete_mean = complete_sigmastar' / complete_sigma * y;
complete_mean = complete_sigmastar' * (complete_sigma \ y); % The same?
%complete_var = diag(complete_sigmastarstart - complete_sigmastar' / complete_sigma * complete_sigmastar);
complete_var = diag(complete_sigmastarstart - complete_sigmastar' * (complete_sigma \ complete_sigmastar)); % The same?
    
figure(1); clf; hold on;
plot(X, y, 'ko');
plot(xrange, complete_mean, 'b', 'LineWidth', 2);
%plot(xrange, complete_mean + 2.*sqrt(complete_var), 'g');
%plot(xrange, complete_mean - 2.*sqrt(complete_var), 'g');
hold off;
title(['alpha ', num2str(hyp_opt.cov(3)), ' nlZ ', num2str(nlZ)]);
save2pdf(['RQ_', num2str(hyp_opt.cov(3)), '_', num2str(nlZ), '.pdf'], ...
         h, 600);

% Plot residuals.
%data_complete_mean = feval(complete_covfunc{:}, complete_hypers, X, X)' / complete_sigma * y;
%figure(2); clf; hold on;
%plot(X, y - data_complete_mean, 'ko');
%plot(xrange, +2.*sqrt(noise_var).*ones(size(xrange)), 'g');
%plot(xrange, -2.*sqrt(noise_var).*ones(size(xrange)), 'g');
%hold off;

%% Unpacked

load 'fur-sales-mink-h-b-co-18481911.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('../../source/gpml/'));
addpath(genpath('../../source/matlab'));

complete_covfunc = {@covProd, {{@covMask, {[1], {@covRQiso}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covPeriodic}}}}};

complete_hypers = [ -1.42263 7.012301 -5.254423 -2.451057 -2.337529 -0.001647 0.002361 -0.000449 0.942026 2.259133 8.583919 ];
hyp.cov = complete_hypers;

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = [5.4321688];
log_noise = hyp.lik;

[nlZ, dnlZ] = gp(hyp, @infExact, mean_func, complete_covfunc, lik_func, X, y);
nlZ

left_extend = 0.4;  % What proportion to extend beyond the data range.
right_extend = 0.4;

num_interpolation_points = 2000;

x_left = min(X) - (max(X) - min(X))*left_extend;
x_right = max(X) + (max(X) - min(X))*right_extend;
xrange = linspace(x_left, x_right, num_interpolation_points)';

noise_var = exp(2*log_noise);
complete_sigma = feval(complete_covfunc{:}, complete_hypers, X, X) + eye(length(y)).*noise_var;
complete_sigmastar = feval(complete_covfunc{:}, complete_hypers, X, X);% + eye(length(y)).*noise_var;

% New thing - remove all 'noise'
complete_sigmastar_nonoise = complete_sigmastar - diag(diag(complete_sigmastar));
complete_sigmastar_nonoise = complete_sigmastar_nonoise + diag(max(complete_sigmastar_nonoise));

complete_sigmastarstart = feval(complete_covfunc{:}, complete_hypers, X, X);

% First, plot the original, combined kernel
%complete_mean = complete_sigmastar' / complete_sigma * y;
complete_mean = 0.5*(complete_sigmastar_nonoise + complete_sigmastar)' * (complete_sigma \ y); % The same?
%complete_var = diag(complete_sigmastarstart - complete_sigmastar' / complete_sigma * complete_sigmastar);
complete_var = diag(complete_sigmastarstart - complete_sigmastar' * (complete_sigma \ complete_sigmastar)); % The same?
    
h = figure(2); clf; hold on;
plot(X, y, 'ko');
plot(X, complete_mean, 'b', 'LineWidth', 2);
%plot(xrange, complete_mean + 2.*sqrt(complete_var), 'g');
%plot(xrange, complete_mean - 2.*sqrt(complete_var), 'g');
hold off;

% Plot residuals.
%data_complete_mean = feval(complete_covfunc{:}, complete_hypers, X, X)' / complete_sigma * y;
%figure(2); clf; hold on;
%plot(X, y - data_complete_mean, 'ko');
%plot(xrange, +2.*sqrt(noise_var).*ones(size(xrange)), 'g');
%%plot(xrange, -2.*sqrt(noise_var).*ones(size(xrange)), 'g');
hold off;

%% Be more Bayesian

load 'fur-sales-mink-h-b-co-18481911.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('../../source/gpml/'));
addpath(genpath('../../source/matlab'));

complete_covfunc = {@covProd, {@covRQiso, @covPeriodic}};

complete_hypers = [ +5.4, 7, -6, 0.942026 2.259133 8.583919 ];
hyp.cov = complete_hypers;

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = 5.4;
log_noise = hyp.lik;

hyp_opt = minimize(hyp, @gp, -1000, @infExact, ...
                   mean_func, complete_covfunc, lik_func, X, y);
[nlZ, dnlZ] = gp(hyp_opt, @infExact, mean_func, complete_covfunc, lik_func, X, y);
nlZ

complete_hypers = hyp_opt.cov;
log_noise = hyp_opt.lik;

left_extend = 0.4;  % What proportion to extend beyond the data range.
right_extend = 0.4;

num_interpolation_points = 2000;

x_left = min(X) - (max(X) - min(X))*left_extend;
x_right = max(X) + (max(X) - min(X))*right_extend;
xrange = linspace(x_left, x_right, num_interpolation_points)';

noise_var = exp(2*log_noise);
complete_sigma = feval(complete_covfunc{:}, complete_hypers, X, X) + eye(length(y)).*noise_var;
complete_sigmastar = feval(complete_covfunc{:}, complete_hypers, X, xrange);
%complete_sigmastarstart = feval(complete_covfunc{:}, complete_hypers, xrange, xrange);

complete_sigmastar_nonoise = complete_sigmastar - diag(diag(complete_sigmastar));
complete_sigmastar_nonoise = complete_sigmastar_nonoise + diag(max(complete_sigmastar_nonoise));

% First, plot the original, combined kernel
%complete_mean = complete_sigmastar' / complete_sigma * y;

%complete_sigma = complete_sigma - 0.5 * (min(diag(complete_sigma)) - max(max(complete_sigma - diag(diag(complete_sigma))))) * eye(size(complete_sigma));

complete_mean = complete_sigmastar_nonoise' * (complete_sigma \ y); % The same?
%complete_var = diag(complete_sigmastarstart - complete_sigmastar' / complete_sigma * complete_sigmastar);
%complete_var = diag(complete_sigmastarstart - complete_sigmastar' * (complete_sigma \ complete_sigmastar)); % The same?
    
figure(1); clf; hold on;
plot(X, y, 'ko');
plot(xrange, complete_mean, 'b', 'LineWidth', 2);
%plot(xrange, complete_mean + 2.*sqrt(complete_var), 'g');
%plot(xrange, complete_mean - 2.*sqrt(complete_var), 'g');
hold off;
title(['alpha ', num2str(hyp_opt.cov(3)), ' nlZ ', num2str(nlZ)]);

%% Compute Hessian numerically for laplace approx
num_hypers = length(hyp_opt.cov);
hessian = NaN(num_hypers+1, num_hypers+1);
delta = 1e-6;
a='Get original gradients';
[nll_orig, dnll_orig] = gp(hyp_opt, @infExact, mean_func, complete_covfunc, lik_func, X, y);
for d = 1:(num_hypers+1)
    dhyp_opt = hyp_opt;
    if d <= num_hypers
        dhyp_opt.cov(d) = dhyp_opt.cov(d) + delta;
    else
        dhyp_opt.lik = dhyp_opt.lik + delta;
    end
    [nll_delta, dnll_delta] = gp(dhyp_opt, @infExact, mean_func, complete_covfunc, lik_func, X, y);
    hessian(d, :) = ([dnll_delta.cov, dnll_delta.lik] - [dnll_orig.cov, dnll_orig.lik]) ./ delta;
end
hessian = 0.5 * (hessian + hessian');
hessian = hessian + 10e-6*max(max(hessian))*eye(size(hessian));

%% Average of many samples

left_extend = 0.4;  % What proportion to extend beyond the data range.
right_extend = 0.4;

num_interpolation_points = 2000;

x_left = min(X) - (max(X) - min(X))*left_extend;
x_right = max(X) + (max(X) - min(X))*right_extend;
xrange = linspace(x_left, x_right, num_interpolation_points)';

average_mean = zeros(size(xrange));

iters = 100;

for iter = 1:iters
    iter

    sample_hyp = hyp_opt;
    sample = [hyp_opt.cov, hyp_opt.lik] + mvnrnd(zeros(size(hessian, 1), 1), hessian^-1);
    sample_hyp.cov = sample(1:num_hypers);
    sample_hyp.lik = sample(end);

    sample_hyp

    complete_hypers = sample_hyp.cov;
    log_noise = sample_hyp.lik;

    noise_var = exp(2*log_noise);
    complete_sigma = feval(complete_covfunc{:}, complete_hypers, X, X) + eye(length(y)).*noise_var;
    complete_sigmastar = feval(complete_covfunc{:}, complete_hypers, X, xrange);
    complete_sigmastarstart = feval(complete_covfunc{:}, complete_hypers, xrange, xrange);

    % First, plot the original, combined kernel
    %complete_mean = complete_sigmastar' / complete_sigma * y;
    complete_mean = complete_sigmastar' * (complete_sigma \ y); % The same?
    average_mean = average_mean + complete_mean;
    %complete_var = diag(complete_sigmastarstart - complete_sigmastar' / complete_sigma * complete_sigmastar);
    %complete_var = diag(complete_sigmastarstart - complete_sigmastar' * (complete_sigma \ complete_sigmastar)); % The same?
end

average_mean = average_mean / iters;

figure(2); clf; hold on;
plot(X, y, 'ko');
plot(xrange, average_mean, 'b', 'LineWidth', 2);
%plot(xrange, complete_mean + 2.*sqrt(complete_var), 'g');
%plot(xrange, complete_mean - 2.*sqrt(complete_var), 'g');
hold off;

%% Bootstrap residuals

load 'fur-sales-mink-h-b-co-18481911.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('../../source/gpml/'));
addpath(genpath('../../source/matlab'));

%complete_covfunc = {@covProd, {@covRQiso, @covPeriodic}};

%complete_hypers = [ +7.4, 7, -5, 0.942026 2.259133 8.583919 ];

complete_covfunc = {@covProd, {{@covMask, {[1], {@covRQiso}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covPeriodic}}}}};

complete_hypers = [ -1.42263 7.012301 -5.254423 -2.451057 -2.337529 -0.001647 0.002361 -0.000449 0.942026 2.259133 8.583919 ];

hyp.cov = complete_hypers;

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
%hyp.lik = 5.4;
hyp.lik = [5.4321688];
log_noise = hyp.lik;

%hyp_opt = minimize(hyp, @gp, -1000, @infExact, ...
%                   mean_func, complete_covfunc, lik_func, X, y);
hyp_opt = hyp;
[nlZ, dnlZ] = gp(hyp_opt, @infExact, mean_func, complete_covfunc, lik_func, X, y);
nlZ

complete_hypers = hyp_opt.cov;
log_noise = hyp_opt.lik;

left_extend = 0.4;  % What proportion to extend beyond the data range.
right_extend = 0.4;

num_interpolation_points = 2000;

x_left = min(X) - (max(X) - min(X))*left_extend;
x_right = max(X) + (max(X) - min(X))*right_extend;
xrange = linspace(x_left, x_right, num_interpolation_points)';

x_boot = unifrnd(min(X), max(X), 200, 1);

noise_var = exp(2*log_noise);
complete_sigma = feval(complete_covfunc{:}, complete_hypers, X, X) + eye(length(y)).*noise_var;
y_fit_sigma = feval(complete_covfunc{:}, complete_hypers, X, X);
y_boot_fit_sigma = feval(complete_covfunc{:}, complete_hypers, X, x_boot);
complete_sigmastar = feval(complete_covfunc{:}, complete_hypers, X, xrange);
%complete_sigmastarstart = feval(complete_covfunc{:}, complete_hypers, xrange, xrange);

%complete_sigmastar_nonoise = complete_sigmastar - diag(diag(complete_sigmastar));
%complete_sigmastar_nonoise = complete_sigmastar_nonoise + diag(max(complete_sigmastar_nonoise));

% First, plot the original, combined kernel
%complete_mean = complete_sigmastar' / complete_sigma * y;

%complete_sigma = complete_sigma - 0.5 * (min(diag(complete_sigma)) - max(max(complete_sigma - diag(diag(complete_sigma))))) * eye(size(complete_sigma));

complete_mean = complete_sigmastar' * (complete_sigma \ y); % The same?
y_fit = y_fit_sigma' * (complete_sigma \ y); % The same?
y_boot_fit = y_boot_fit_sigma' * (complete_sigma \ y); % The same?
    
figure(1); clf; hold on;
plot(X, y, 'ko');
plot(xrange, complete_mean, 'b', 'LineWidth', 2);
hold off;
title(['alpha ', num2str(hyp_opt.cov(3)), ' nlZ ', num2str(nlZ)]);

%% Bootstrap iters

residuals = y - y_fit;% + randn(size(y)) * std(y) * 0.1;
B = 3;
y_boot = [];
X_boot = [];
for b = 1:B
    y_boot = [y_boot; y_boot_fit + randsample(residuals, length(x_boot), true)];
    %X_boot = [X_boot; X + randn(size(X))];
    X_boot = [X_boot; x_boot];
end

hyp_opt = minimize(hyp_opt, @gp, -100, @infExact, ...
                   mean_func, complete_covfunc, lik_func, X_boot, y_boot);
[nlZ, dnlZ] = gp(hyp_opt, @infExact, mean_func, complete_covfunc, lik_func, X_boot, y_boot);
nlZ

complete_hypers = hyp_opt.cov;
log_noise = hyp_opt.lik;

left_extend = 0.4;  % What proportion to extend beyond the data range.
right_extend = 0.4;

num_interpolation_points = 2000;

x_left = min(X) - (max(X) - min(X))*left_extend;

x_right = max(X) + (max(X) - min(X))*right_extend;
xrange = linspace(x_left, x_right, num_interpolation_points)';

x_boot = unifrnd(min(X), max(X), 200, 1);

noise_var = exp(2*log_noise);
complete_sigma = feval(complete_covfunc{:}, complete_hypers, X_boot, X_boot) + eye(length(y_boot)).*noise_var;
y_fit_sigma = feval(complete_covfunc{:}, complete_hypers, X_boot, X);
y_boot_fit_sigma = feval(complete_covfunc{:}, complete_hypers, X_boot, x_boot);
complete_sigmastar = feval(complete_covfunc{:}, complete_hypers, X_boot, xrange);
%complete_sigmastarstart = feval(complete_covfunc{:}, complete_hypers, xrange, xrange);

%complete_sigmastar_nonoise = complete_sigmastar - diag(diag(complete_sigmastar));
%complete_sigmastar_nonoise = complete_sigmastar_nonoise + diag(max(complete_sigmastar_nonoise));

% First, plot the original, combined kernel
%complete_mean = complete_sigmastar' / complete_sigma * y;

%complete_sigma = complete_sigma - 0.5 * (min(diag(complete_sigma)) - max(max(complete_sigma - diag(diag(complete_sigma))))) * eye(size(complete_sigma));

complete_mean = complete_sigmastar' * (complete_sigma \ y_boot); % The same?
y_fit = y_fit_sigma' * (complete_sigma \ y_boot); % The same?
y_boot_fit = y_boot_fit_sigma' * (complete_sigma \ y_boot); % The same?
    
figure(2); clf; hold on;
plot(X, y, 'ko');
plot(xrange, complete_mean, 'b', 'LineWidth', 2);
hold off;
title(['alpha ', num2str(hyp_opt.cov(3)), ' nlZ ', num2str(nlZ)]);

%% If we optimise from here???

complete_covfunc = {@covProd, {@covRQiso, @covPeriodic}};

hyp_opt = minimize(hyp_opt, @gp, -1000, @infExact, ...
                   mean_func, complete_covfunc, lik_func, X, y);
[nlZ, dnlZ] = gp(hyp_opt, @infExact, mean_func, complete_covfunc, lik_func, X, y);
nlZ

complete_hypers = hyp_opt.cov;
log_noise = hyp_opt.lik;

left_extend = 0.4;  % What proportion to extend beyond the data range.
right_extend = 0.4;

num_interpolation_points = 2000;

x_left = min(X) - (max(X) - min(X))*left_extend;
x_right = max(X) + (max(X) - min(X))*right_extend;
xrange = linspace(x_left, x_right, num_interpolation_points)';

noise_var = exp(2*log_noise);
complete_sigma = feval(complete_covfunc{:}, complete_hypers, X, X) + eye(length(y)).*noise_var;
y_fit_sigma = feval(complete_covfunc{:}, complete_hypers, X, X);
complete_sigmastar = feval(complete_covfunc{:}, complete_hypers, X, xrange);
%complete_sigmastarstart = feval(complete_covfunc{:}, complete_hypers, xrange, xrange);

%complete_sigmastar_nonoise = complete_sigmastar - diag(diag(complete_sigmastar));
%complete_sigmastar_nonoise = complete_sigmastar_nonoise + diag(max(complete_sigmastar_nonoise));

% First, plot the original, combined kernel
%complete_mean = complete_sigmastar' / complete_sigma * y;

%complete_sigma = complete_sigma - 0.5 * (min(diag(complete_sigma)) - max(max(complete_sigma - diag(diag(complete_sigma))))) * eye(size(complete_sigma));

complete_mean = complete_sigmastar' * (complete_sigma \ y); % The same?
y_fit = y_fit_sigma' * (complete_sigma \ y); % The same?
    
figure(3); clf; hold on;
plot(X, y, 'ko');
plot(xrange, complete_mean, 'b', 'LineWidth', 2);
hold off;
title(['alpha ', num2str(hyp_opt.cov(3)), ' nlZ ', num2str(nlZ)]);

%% Bootstrap mark two

load 'fur-sales-mink-h-b-co-18481911.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('../../source/gpml/'));
addpath(genpath('../../source/matlab'));

%complete_covfunc = {@covProd, {@covRQiso, @covPeriodic}};

%complete_hypers = [ +7.4, 7, -5, 0.942026 2.259133 8.583919 ];

%complete_covfunc = {@covProd, {{@covMask, {[1], {@covRQiso}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covPeriodic}}}}};

%complete_hypers = [ -1.42263 7.012301 -5.254423 -2.451057 -2.337529 -0.001647 0.002361 -0.000449 0.942026 2.259133 8.583919 ];

hyp.cov = complete_hypers;

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
%hyp.lik = 5.4;
hyp.lik = [5.4321688];
log_noise = hyp.lik;

%hyp_opt = minimize(hyp, @gp, -1000, @infExact, ...
%                   mean_func, complete_covfunc, lik_func, X, y);
hyp_opt = hyp;
[nlZ, dnlZ] = gp(hyp_opt, @infExact, mean_func, complete_covfunc, lik_func, X, y);
nlZ

complete_hypers = hyp_opt.cov;
log_noise = hyp_opt.lik;

left_extend = 0.4;  % What proportion to extend beyond the data range.
right_extend = 0.4;

num_interpolation_points = 2000;

x_left = min(X) - (max(X) - min(X))*left_extend;
x_right = max(X) + (max(X) - min(X))*right_extend;
xrange = linspace(x_left, x_right, num_interpolation_points)';

noise_var = exp(2*log_noise);
complete_sigma = feval(complete_covfunc{:}, complete_hypers, X, X) + eye(length(y)).*noise_var;
y_fit_sigma = feval(complete_covfunc{:}, complete_hypers, X, X);
complete_sigmastar = feval(complete_covfunc{:}, complete_hypers, X, xrange);
%complete_sigmastarstart = feval(complete_covfunc{:}, complete_hypers, xrange, xrange);

%complete_sigmastar_nonoise = complete_sigmastar - diag(diag(complete_sigmastar));
%complete_sigmastar_nonoise = complete_sigmastar_nonoise + diag(max(complete_sigmastar_nonoise));

% First, plot the original, combined kernel
%complete_mean = complete_sigmastar' / complete_sigma * y;

%complete_sigma = complete_sigma - 0.5 * (min(diag(complete_sigma)) - max(max(complete_sigma - diag(diag(complete_sigma))))) * eye(size(complete_sigma));

complete_mean = complete_sigmastar' * (complete_sigma \ y); % The same?
y_fit = y_fit_sigma' * (complete_sigma \ y); % The same?
    
figure(1); clf; hold on;
plot(X, y, 'ko');
plot(xrange, complete_mean, 'b', 'LineWidth', 2);
hold off;
title(['alpha ', num2str(hyp_opt.cov(3)), ' nlZ ', num2str(nlZ)]);

y_cv_fit = y_fit;
for i = 1:length(y_cv_fit)
    y_cv_fit(i) = y_fit_sigma(i,[1:(i-1),(i+1):end]) * (complete_sigma([1:(i-1),(i+1):end],[1:(i-1),(i+1):end]) \ y([1:(i-1),(i+1):end]));
end

%% Bootstrap iters

for dummy = 1:100

    %residuals = y - y_fit;% + randn(size(y)) * std(y) * 0.1;
    %residuals = randn(size(y)) * std(y) * 0.1;
    %B = 5;
    %y_boot = [];
    %X_boot = [];
    %for b = 1:B
    %    y_boot = [y_boot; y_cv_fit + randn(size(y)) * std(y - y_cv_fit)];
    %    X_boot = [X_boot; X];
    %    %X_boot = [X_boot; x_boot];
    %end

    X_boot = X;
    y_boot = y_cv_fit;

    hyp_opt = minimize(hyp_opt, @gp, -250, @infExact, ...
                       mean_func, complete_covfunc, lik_func, X_boot, y_boot);
    [nlZ, dnlZ] = gp(hyp_opt, @infExact, mean_func, complete_covfunc, lik_func, X_boot, y_boot);
    nlZ

    complete_hypers = hyp_opt.cov;
    log_noise = hyp_opt.lik;

    left_extend = 0.4;  % What proportion to extend beyond the data range.
    right_extend = 0.4;

    num_interpolation_points = 2000;

    x_left = min(X) - (max(X) - min(X))*left_extend;

    x_right = max(X) + (max(X) - min(X))*right_extend;
    xrange = linspace(x_left, x_right, num_interpolation_points)';

    noise_var = exp(2*log_noise);
    complete_sigma = feval(complete_covfunc{:}, complete_hypers, X_boot, X_boot) + eye(length(y_boot)).*noise_var;
    y_fit_sigma = feval(complete_covfunc{:}, complete_hypers, X_boot, X);
    complete_sigmastar = feval(complete_covfunc{:}, complete_hypers, X_boot, xrange);
    %complete_sigmastarstart = feval(complete_covfunc{:}, complete_hypers, xrange, xrange);

    %complete_sigmastar_nonoise = complete_sigmastar - diag(diag(complete_sigmastar));
    %complete_sigmastar_nonoise = complete_sigmastar_nonoise + diag(max(complete_sigmastar_nonoise));

    % First, plot the original, combined kernel
    %complete_mean = complete_sigmastar' / complete_sigma * y;

    %complete_sigma = complete_sigma - 0.5 * (min(diag(complete_sigma)) - max(max(complete_sigma - diag(diag(complete_sigma))))) * eye(size(complete_sigma));

    complete_mean = complete_sigmastar' * (complete_sigma \ y_boot); % The same?
    y_fit = y_fit_sigma' * (complete_sigma \ y_boot); % The same?

    y_cv_fit = y_fit;
    for i = 1:length(y_cv_fit)
        y_cv_fit(i) = y_fit_sigma(i,[1:(i-1),(i+1):end]) * (complete_sigma([1:(i-1),(i+1):end],[1:(i-1),(i+1):end]) \ y_boot([1:(i-1),(i+1):end]));
    end

    figure(2); clf; hold on;
    plot(X, y, 'ko');
    plot(xrange, complete_mean, 'b', 'LineWidth', 2);
    hold off;
    title(['alpha ', num2str(hyp_opt.cov(3)), ' nlZ ', num2str(nlZ)]);

    drawnow;
    
    std(y_cv_fit - y)
    
end