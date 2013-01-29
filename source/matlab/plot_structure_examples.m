function plot_structure_examples()

% A script to make a series of plots demonstrating how structure can be
% reflected in a kernel.
%
% David Duvenaud
% Jan 2013

seed=0;   % fixing the seed of the random generators
randn('state',seed);
rand('state',seed);

savefigs = true;
figpath = '../../figures/structure_examples/';

% Make up some data
X = [ -2 -1 0 1 2 ]' .* 2;
y = [ -1 1 1 -0.8 1.1  ]';
N = length(X);

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
lin_output_var = 20;
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


% plot the kernels.
for k = 1:numel(kernels)
    figure(k); clf;
    cur_kernel = kernels{k};
    kvals = bsxfun(cur_kernel, xrange, x0 );
    kernel_plot( xrange, kvals, col_ixs(k) );
    if savefigs
        save2pdf([ figpath, kernel_names{k}, '.pdf'], gcf, 600, true);
    end
end


% Plot draws from the kernels.
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


% Plot posterior dists of the kernels.
for k = 1:numel(kernels)
    figure(20 + k); clf;
    cur_kernel = kernels{k};
    K = bsxfun(cur_kernel, X, X' ) + eye(N).*noise; % Evaluate prior.
    weights = K \ y;
    posterior = @(x)(bsxfun(cur_kernel, xrange, X') * weights); % Construct posterior function.
    posterior_variance = @(x)diag(bsxfun(cur_kernel, xrange', xrange) - (bsxfun(cur_kernel, X', xrange) / K * bsxfun(cur_kernel, xrange', X)));
    posterior_plot( xrange, posterior(xrange), posterior_variance(xrange), X, y );

    if savefigs
        save2pdf([ figpath, kernel_names{k} '_post.pdf'], gcf, 600, true);
    end
end

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


function posterior_plot( xrange, f, v, X, y )
    % Figure settings.
    lw = 2;
    fontsize = 10;
    opacity = 1;
    fake_opacity = 0.1;
    
    jbfill( xrange', ...
            f' + 2.*sqrt(v)', ...
            f' - 2.*sqrt(v)', ...
            colorbrew(2).^fake_opacity, 'none', 1, opacity); hold on;
    plot( xrange, f, '-', 'Linewidth', lw, 'Color', colorbrew(2)); hold on;

    % Plot data.
    plot( X, y, 'kx', 'Linewidth', lw, 'MarkerSize', 6); hold on;

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
