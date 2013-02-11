function plot_decomp(X, y, complete_covfunc, complete_hypers, decomp_list, ...
                     decomp_hypers, log_noise, figname, latex_names, ...
                     full_name, X_mean, X_scale, y_mean, y_scale)

% TODO: Assert that the sum of all kernels is the same as the complete kernel.

% Convert to double in case python saved as integers
X = double(X);
y = double(y);
y = y - mean(y);

left_extend = 0.05;  % What proportion to extend beyond the data range.
right_extend = 0.2;

x_left = min(X) - (max(X) - min(X))*left_extend;
x_right = max(X) + (max(X) - min(X))*right_extend;
xrange = linspace(x_left, x_right, 2000)';


noise_var = exp(2*log_noise);
complete_sigma = feval(complete_covfunc{:}, complete_hypers, X, X) + eye(length(y)).*noise_var;
complete_sigmastar = feval(complete_covfunc{:}, complete_hypers, X, xrange);
complete_sigmastarstart = feval(complete_covfunc{:}, complete_hypers, xrange, xrange);

% First, plot the original, combined kernel
complete_mean = complete_sigmastar' / complete_sigma * y;
complete_var = diag(complete_sigmastarstart - complete_sigmastar' / complete_sigma * complete_sigmastar);
    
figure(1); clf; hold on;
mean_var_plot(xrange*X_scale+X_mean, complete_mean*y_scale+y_mean, 2.*sqrt(complete_var)*y_scale);
plot( X*X_scale+X_mean, y*y_scale+y_mean, 'k.' ); hold on;

% Remove outer brackets and extra latex markup from name.
if iscell(full_name); full_name = full_name{1}; end
full_name = strrep(full_name, '\left', '');
full_name = strrep(full_name, '\right', '');
full_name = strtrim(full_name);
if full_name(1) == '('; full_name(1) = ''; end
if full_name(end) == ')'; full_name(end) = ''; end
title(full_name);
filename = sprintf('%s_all.fig', figname);
saveas( gcf, filename );
%filename = sprintf('%s_all.pdf', figname);
%save2pdf( filename, gcf, 400, true )

% Then plot the same thing, but just the end.
complete_mean = complete_sigmastar' / complete_sigma * y;
complete_var = diag(complete_sigmastarstart - complete_sigmastar' / complete_sigma * complete_sigmastar);
    
figure(100); clf; hold on;
small_plot(xrange*X_scale+X_mean, complete_mean*y_scale+y_mean, 2.*sqrt(complete_var)*y_scale);
plot( X*X_scale+X_mean, y*y_scale+y_mean, 'k.' ); hold on; 
title(full_name);
filename = sprintf('%s_all_small.fig', figname);
saveas( gcf, filename );
%filename = sprintf('%s_all.pdf', figname);
%save2pdf( filename, gcf, 400, true )

% Plot residuals.
figure(1000); clf; hold on;
data_complete_mean = feval(complete_covfunc{:}, complete_hypers, X, X)' / complete_sigma * y;
mean_var_plot(xrange*X_scale+X_mean, zeros(size(xrange)), ...
              2.*sqrt(noise_var).*ones(size(xrange)).*y_scale);
% Make plot prettier.
%set(gcf, 'color', 'white');
%set(gca, 'TickDir', 'out');
plot( X*X_scale+X_mean, (y-data_complete_mean)*y_scale, 'k.' ); hold on; 
%xlim([min(xrange*X_scale), max(xrange*X_scale)] + X_mean);
title('Residuals');
%set_fig_units_cm( 16,8 );  
filename = sprintf('%s_resid.fig', figname);
saveas( gcf, filename );

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
    mean_var_plot(xrange*X_scale+X_mean, decomp_mean*y_scale, 2.*sqrt(decomp_var)*y_scale);
    plot( X*X_scale+X_mean, removed_mean*y_scale, 'k.' ); hold on; 
    latex_names{i} = strrep(latex_names{i}, '\left', '');
    latex_names{i} = strrep(latex_names{i}, '\right', '');
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
    lw = 1.2;
    opacity = 0.14;
 
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
    
    %set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
    %set(get(gca,'YLabel'),'Rotation',90,'Interpreter','latex', 'Fontsize', fontsize);
    %set(gca, 'TickDir', 'out')
    
    set_fig_units_cm( 16,8 );    
end


function small_plot( xrange, forecast_mu, forecast_scale )
    % Figure settings.
    lw = 1.2;
    opacity = 0.14;
 
    plot(xrange, forecast_mu, 'Color', colorbrew(2), 'LineWidth', lw); hold on;
    
    % Plot confidence bears.
    jbfill( xrange', ...
            forecast_mu' + forecast_scale', ...
            forecast_mu' - forecast_scale', ...
            colorbrew(2), 'none', 1, opacity); hold on;
    
    % Make plot prettier.
    set(gcf, 'color', 'white');
    set(gca, 'TickDir', 'out');
    
    xlim([min(xrange) + (max(xrange) - min(xrange))*0.6, max(xrange)]);
    
    %set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
    %set(get(gca,'YLabel'),'Rotation',90,'Interpreter','latex', 'Fontsize', fontsize);
    %set(gca, 'TickDir', 'out')
    
    set_fig_units_cm( 8,8 );    
end
