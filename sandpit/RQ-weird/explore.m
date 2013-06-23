%% Standard plotting code

load 'fur-sales-mink-h-b-co-18481911.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('../source/gpml/'));
addpath(genpath('/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gp-structure-search/source/matlab'));

plot_decomp(X, y, {@covProd, {{@covMask, {[1], {@covRQiso}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covPeriodic}}}}}, [ -1.42263 7.012301 -5.254423 -2.451057 -2.337529 -0.001647 0.002361 -0.000449 0.942026 2.259133 8.583919 ], { {@covProd, {{@covMask, {[1], {@covRQiso}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covConst}}}, {@covMask, {[1], {@covPeriodic}}}}} }, { [ -1.42263 7.012301 -5.254423 -2.451057 -2.337529 -0.001647 0.002361 -0.000449 0.942026 2.259133 8.583919 ] }, [5.4321688], '/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gp-structure-search/sandpit/dummy', { ' RQ_{0} \times CS_{0} \times CS_{0} \times CS_{0} \times CS_{0} \times CS_{0} \times Per_{0} ' }, { 'RQ_{0} \times CS_{0} \times CS_{0} \times CS_{0} \times CS_{0} \times CS_{0} \times Per_{0}' }, 0.000000, 1.000000, 0.000000, 1.000000)

%% Unpacked

load 'fur-sales-mink-h-b-co-18481911.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('../../source/gpml/'));
addpath(genpath('/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gp-structure-search/source/matlab'));

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
addpath(genpath('/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gp-structure-search/source/matlab'));

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
addpath(genpath('/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gp-structure-search/source/matlab'));

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
    
h = figure(1); clf; hold on;
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
addpath(genpath('/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gp-structure-search/source/matlab'));

complete_covfunc = {@covProd, {@covRQiso, @covPeriodic}};

complete_hypers = [ +5.4, 7, -5, 0.94, 2.26, 0 ];
hyp.cov = complete_hypers;

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = 5;
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