%% Set random seed

rand('twister', 0);
randn('state', 0);

%% Create synthetic data

covs{1} = {@covSum, {...
                    {@covLINscaleshift}, ...
                    {@covPeriodic} ...
                    } ...
                    };
cov_params{1} = [0 0 0 0 0];
dims{1} = 1;

i = 1;
n = 100;
x_max = 10;

X = (rand(n,dims{i}))*x_max;
K = feval(covs{i}{:}, cov_params{i}, X);
K = K +  1e-6*eye(n);
y = chol(K)' * randn(n,1);
y = y - mean(y);
y = y / std(y);
y = y + randn(size(y)) * 0.1;

plot(X, y, '+');

%% Plot a basic smoother fit

covs{2} = {@covSEiso};
cov_params{2} = [-2 0];
likfunc = @likGauss;
hyp.lik = 0;
hyp.cov = cov_params{2};
meanfunc = @meanConst;
hyp.mean = 0;
covfunc = covs{2}

hyp_opt = minimize(hyp, @gp, -100, @infExact, ...
                   meanfunc, covfunc, likfunc, X, y);
               
x_predict = linspace(0, 15, 300)';
               
[m s2] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, ...
            X, y, x_predict);

figure(1); clf; hold on;

% Figure settings.
lw = 1.2;
opacity = 1;
light_blue = [227 237 255]./255;

% Plot confidence bears.
jbfill( x_predict', ...
    m' + s2', ...
    m' - s2', ...
    light_blue, 'none', 1, opacity); hold on;   

set(gca,'Layer','top');  % Stop axes from being overridden.

plot( X, y, 'k.');

plot(x_predict, m, 'Color', colorbrew(2), 'LineWidth', lw); hold on;

% Make plot prettier.
set(gcf, 'color', 'white');
set(gca, 'TickDir', 'out');
title('Local smooth structure')

% Plot a vertical bar to indicate the start of extrapolation.
y_lim = get(gca,'ylim');
line( [10, 10], y_lim, 'Linestyle', '--', 'Color', [0.3 0.3 0.3 ]);

set_fig_units_cm( 16,8 );

%saveas( gcf, 'linear.pdf' );
save2pdf('synth_extrap_bad.pdf', gcf, 600);

%% Plot the true fit

likfunc = @likGauss;
hyp.lik = 0;
hyp.cov = cov_params{1};
meanfunc = @meanConst;
hyp.mean = 0;
covfunc = covs{1}

hyp_opt = minimize(hyp, @gp, -100, @infExact, ...
                   meanfunc, covfunc, likfunc, X, y);
               
x_predict = linspace(0, 15, 300)';
               
[m s2] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, ...
            X, y, x_predict);

figure(1); clf; hold on;

% Figure settings.
lw = 1.2;
opacity = 1;
light_blue = [227 237 255]./255;

% Plot confidence bears.
jbfill( x_predict', ...
    m' + s2', ...
    m' - s2', ...
    light_blue, 'none', 1, opacity); hold on;   

set(gca,'Layer','top');  % Stop axes from being overridden.

plot( X, y, 'k.');

plot(x_predict, m, 'Color', colorbrew(2), 'LineWidth', lw); hold on;

% Make plot prettier.
set(gcf, 'color', 'white');
set(gca, 'TickDir', 'out');
title('True structure')

% Plot a vertical bar to indicate the start of extrapolation.
y_lim = get(gca,'ylim');
line( [10, 10], y_lim, 'Linestyle', '--', 'Color', [0.3 0.3 0.3 ]);

set_fig_units_cm( 16,8 );

%saveas( gcf, 'linear.pdf' );
save2pdf('synth_extrap_good.pdf', gcf, 600);

