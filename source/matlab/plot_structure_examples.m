function plot_structure_examples()

% A script to make a series of plots demonstrating how structure can be
% reflected in a kernel.
%
% David Duvenaud
% Jan 2013

seed=0;   % fixing the seed of the random generators
randn('state',seed);
rand('state',seed);

savefigs = false;
figpath = '../../figures/structure_examples/';

% Make up some data
x = [ -2 -1 0 1 2 ]';
y = [ -1 1 1 -0.8 1.1  ]';
N = length(x);

n_samples = 4;

n_xstar = 600;
xrange = linspace(-10, 10, n_xstar)';
x0 = 0;
noise = 1e-6;

se_length_scale = 2.5;
se_outout_var = 1;
se_kernel = @(x,y) se_outout_var*exp( - 0.5 * ( ( x - y ) .^ 2 ) ./ se_length_scale^2 );


%lin_output_var = 0.1;
%lin_kernel = @(x,y) lin_output_var*( x .* y );
lin_length_scale = 20;
lin_output_var = 10;
lin_kernel = @(x,y) lin_output_var*exp( - 0.5 * ( ( x - y ) .^ 2 ) ./ lin_length_scale^2 );


per_length_scale = 1;
per_period = 4;
per_outout_var = 1;
per_kernel = @(x,y) per_outout_var*exp( - 2 * ( sin(pi*( x - y )./per_period) .^ 2 ) ./ per_length_scale^2 );

se_plus_lin = @(x,y) se_kernel(x, y) + lin_kernel(x, y);
se_plus_per = @(x,y) se_kernel(x, y) + per_kernel(x, y);
se_times_lin = @(x,y) se_kernel(x, y) .* lin_kernel(x, y);
se_times_per = @(x,y) se_kernel(x, y) .* per_kernel(x, y);
lin_times_per = @(x,y) lin_kernel(x, y) .* per_kernel(x, y);
lin_plus_per = @(x,y) lin_kernel(x, y) + per_kernel(x, y);

kernels = {se_kernel, lin_kernel, per_kernel, se_plus_lin, se_plus_per, se_times_lin, se_times_per, lin_times_per, lin_plus_per};
kernel_names = {'se_kernel', 'lin_kernel', 'per_kernel', 'se_plus_lin', 'se_plus_per', 'se_times_lin', 'se_times_per', 'lin_times_per', 'lin_plus_per'};
col_ixs = [ 1, 2, 3, 10, 10, 10, 10, 10, 10 ]; 
% plot the kernels

for k = 1:numel(kernels)
    figure(k); clf;
    cur_kernel = kernels{k};
    kvals = bsxfun(cur_kernel, xrange, x0 );
    kernel_plot( xrange, kvals, col_ixs(k) );
    if savefigs
        save2pdf([ figpath, kernel_names{k}, '.pdf'], gcf, 600, true);
    end
end

for k = 1:numel(kernels)
    figure(10 + k); clf;
    K = bsxfun(kernels{k}, xrange', xrange ) + eye(n_xstar).*noise; % Evaluate prior.
    %L = chol(K);
    samples = mvnrnd( zeros(size(xrange)), K, n_samples)';
    samples_plot( xrange, samples, [1:n_samples] );

    if savefigs
        save2pdf([ figpath, kernel_names{k} '_draws.pdf'], gcf, 600, true);
    end
end

% Perform GP inference to get posterior mean function.
K = bsxfun(se_kernel, xrange, xrange ); % Fill in gram matrix
C = inv( K + noise^2 .* diag(ones(n_f_samples,1)) ); % Compute inverse covariance
weights = C * y;  % Now compute kernel function weights.
posterior = @(x)(bsxfun(se_kernel, function_sample_points, x) * weights); % Construct posterior function.
posterior_variance = @(x)(bsxfun(se_kernel, x, x) - diag((bsxfun(se_kernel, x, function_sample_points) * C) * bsxfun(se_kernel, function_sample_points, x)'));


% TODO: check if noise formula is correct.
complete_sigma = feval(complete_covfunc{:}, complete_hypers, X, X) + eye(length(y)).*exp(noise);
complete_sigmastar = feval(complete_covfunc{:}, complete_hypers, X, xrange);
complete_sigmastarstart = feval(complete_covfunc{:}, complete_hypers, xrange, xrange);

% First, plot the original, combined kernel
complete_mean = complete_sigmastar' / complete_sigma * y;
complete_var = diag(complete_sigmastarstart - complete_sigmastar' / complete_sigma * complete_sigmastar);
    
figure(1); clf; hold on;
plot( X, y, '.' ); hold on; 
mean_var_plot(xrange, complete_mean, 2.*sqrt(complete_var));
combined_latex_name = [sprintf('%s + ',latex_names{1:end-1}), latex_names{end}];
title(combined_latex_name);
filename = sprintf('%s_all.fig', figname);
saveas( gcf, filename );
%filename = sprintf('%s_all.pdf', figname);
%save2pdf( filename, gcf, 400, true )

for i = 1:numel(decomp_list)
    cur_cov = decomp_list{i};
    cur_hyp = decomp_hypers{i};
    
    % Compute mean and variance for this kernel.
    decomp_sigma = feval(cur_cov{:}, cur_hyp, X, X);
    decomp_sigma_star = feval(cur_cov{:}, cur_hyp, X, xrange);
    decomp_sigma_starstar = feval(cur_cov{:}, cur_hyp, xrange, xrange);
    decomp_mean = decomp_sigma_star' / complete_sigma * y;
    decomp_var = diag(decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star);
    
    % Compute the remaining signal after removing the mean prediction from all
    % other parts of the kernel.
    removed_mean = y - (complete_sigma - decomp_sigma)' / complete_sigma * y;
    
    figure(i + 1); clf; hold on;
    plot( X, removed_mean, '.' ); hold on; 
    mean_var_plot(xrange, decomp_mean, 2.*sqrt(decomp_var));
    title(latex_names{i});
    fprintf([latex_names{i}, '\n']);
    filename = sprintf('%s_%d.fig', figname, i);
    saveas( gcf, filename );
    %filename = sprintf('%s_%d.pdf', figname, i);
    %save2pdf( filename, gcf, 400, true );
end
end


function mean_var_plot( xrange, forecast_mu, forecast_scale )
    % Figure settings.
    lw = 2;
    opacity = 0.2;
 
    plot(xrange, forecast_mu, 'Color', colorbrew(2), 'LineWidth', lw); hold on;
    
    % Plot confidence bears.
    jbfill( xrange', ...
            forecast_mu' + forecast_scale', ...
            forecast_mu' - forecast_scale', ...
            colorbrew(2), 'none', 1, opacity); hold on;
    
    % Make plot prettier.
    set(gcf, 'color', 'white');
    set(gca, 'TickDir', 'out');
    
    xlim([min(xrange), max(xrange)]);
end

function kernel_plot( xrange, vals, color_ix )
    % Figure settings.
    lw = 2;
    fontsize = 10;
 
    plot(xrange, vals, 'Color', colorbrew(color_ix), 'LineWidth', lw); hold on;
       
    % Make plot prettier.  
    xlim([min(xrange), max(xrange)]);
    ylim([0, max(vals) * 1.05]);
    %set( gca, 'XTick', [ -1 0 1 ] );
    %set( gca, 'yTick', [ 0 1 ] );
    %set( gca, 'XTickLabel', '' );
    %set( gca, 'yTickLabel', '' );
    xlabel( '$x - x''$', 'Fontsize', fontsize );
    ylabel( '$k(x, x'')$', 'Fontsize', fontsize );
    set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
    set(get(gca,'YLabel'),'Rotation',90,'Interpreter','latex', 'Fontsize', fontsize);
    %set(gca, 'TickDir', 'out')
    set(gca, 'Box', 'off');
    set(gcf, 'color', 'white');
    set(gca, 'YGrid', 'off');
    
    set_fig_units_cm( 4,4 );
end

function samples_plot( xrange, samples, color_ix )
    % Figure settings.
    lw = 2;
    fontsize = 10;
 
    for i = 1:size(samples, 2);
        plot(xrange, samples(:,i), 'Color', colorbrew(color_ix(i)), 'LineWidth', lw); hold on;
    end
    
    % Make plot prettier.  
    xlim([min(xrange), max(xrange)]);
    %set( gca, 'XTick', [ -1 0 1 ] );
    %set( gca, 'yTick', [ 0 1 ] );
    %set( gca, 'XTickLabel', '' );
    %set( gca, 'yTickLabel', '' );
    xlabel( '$x$', 'Fontsize', fontsize );
    ylabel( '$f(x)$', 'Fontsize', fontsize );
    set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
    set(get(gca,'YLabel'),'Rotation',90,'Interpreter','latex', 'Fontsize', fontsize);
    %set(gca, 'TickDir', 'out')
    set(gca, 'Box', 'off');
    set(gcf, 'color', 'white');
    set(gca, 'YGrid', 'off');
    
    set_fig_units_cm( 4,4 );
end

