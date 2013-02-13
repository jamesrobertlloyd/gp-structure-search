% Produces a plot of extrapolation results.
%
% This version also gives the baseline methods hints about what lengthscales to
% use, based on what hyperparameters had the best training set marginal
% likelihood on the previous fold.
%
% David Duvenaud
% James Lloyd
% Feb 2013


seed=1;   % fixing the seed of the random generators
randn('state',seed);
rand('state',seed);

results_directory = '../../results/9-Feb 1d learning curves/';
full_data_directory = '../../data/1d_data_rescaled/';
fold_data_directory = '../../data/1d_extrap_folds/';
figure_directory = '../../figures/extrapolation_curves/';
clear experiments
experiments{1} = '01-airline-s';
%experiments{2} = '02-solar-s';
%experiments{1} = '03-mauna2003-s';

addpath(genpath('/scratch/Dropbox/code/gpml/'))

folds = 10;
percentiles = 100 * (1:(folds-1)) / folds;

save_plots = true;

restart_sd = 4;
restarts = 3;

max_iters = 100; % Max iterations for GP training.
lw = 2;        % Line width.

covfunc{1} = {@covSum, {@covConst, @covLINard}};
covfunc{2} = {@covSEiso};
covfunc{3} = {@covPeriodic};
covfunc{4} = {@covSum, {@covSEiso, @covPeriodic}};
covfunc{5} = {@covProd, {@covSEiso, @covPeriodic}};
num_methods = numel(covfunc);

D = 1;

% Init 'best' hypers randomly.
for m = 1:num_methods
    likfunc = @likGauss;
    hyp.lik = log(std(y));
    meanfunc = {@meanConst};
    hyp.mean = mean(y);
    best_nll = Inf;
    n_hypers = eval(feval(covfunc{m}{:}));
    hyp.cov = zeros(1,n_hypers);
    best_hyp{m} = hyp;
end
            
            
for i = 1:length(experiments)
       
    for fold = 1:(folds-1)

        % Load data
        fold_file = [fold_data_directory experiments{i} '-ex-fold-' ...
                     int2str(fold) 'of' int2str(folds) '.mat'];
        load(fold_file);
        X = double(X);
        y = double(y);
        Xtest = double(Xtest);
        ytest = double(ytest);
        
        
        % Extract GPSS result
        gpss_file = [results_directory experiments{i} '-ex-fold-' ...
                     int2str(fold) 'of' int2str(folds)  '_predictions.mat'];
        load(gpss_file);
        MSEs(num_methods+1,fold) = mean((actuals - predictions) .^ 2);
        
        for m = 1:num_methods
            
            % Try old good params.
            [new_hyp nlls] = minimize(best_hyp{m}, @gp, -max_iters, @infExact, ...
                                      meanfunc, covfunc{m}, likfunc, X, y);
            best_nll = nlls(end);
            best_hyp{m} = new_hyp;
                                  
            % Score model
            [pred s2] = gp(new_hyp, @infExact, meanfunc, covfunc{m}, likfunc,...
                        X, y, Xtest);
            MSEs(m,fold) = mean((ytest - pred) .^ 2);
            
            % Now try some random restarts too.
            for r = 1:restarts
                % Randomly init params.
                hyp.lik = log(std(y));
                hyp.mean = mean(y);
                n_hypers = eval(feval(covfunc{m}{:}));
                hyp.cov = zeros(1,n_hypers);
                hyp.cov = hyp.cov + restart_sd * randn(size(hyp.cov));
    
                [new_hyp nlls] = minimize(hyp, @gp, -max_iters, @infExact, ...
                                          meanfunc, covfunc{m}, likfunc, X, y);
                if nlls(end) < best_nll
                    best_nll = nlls(end);
                    best_hyp{m} = new_hyp;
                end
            end

            % Score model
            [pred s2] = gp(best_hyp{m}, @infExact, meanfunc, covfunc{m}, likfunc,...
                        X, y, Xtest);
            MSEs(m,fold) = mean((ytest - pred) .^ 2);             
        end
    end
    
    color_order = [ 4 5 7 2 1 3 ];
    for m = 1:(num_methods+1)
        semilogy(percentiles, MSEs(m,:), '-', 'Color', colorbrew(color_order(m)), ...
                'LineWidth', lw);
        hold on
    end
    xlabel('Proportion of training data (%)');
    ylabel('MSE');
    legend('Linear', 'SE GP', 'Periodic GP', 'SE + Per GP', 'SE x Per GP', 'Structure search', 'location', 'best');
    legend('boxoff')
    hold off
    xlim([10,90]);
    set_fig_units_cm( 12,11 );
    if save_plots
        saveas( gcf, [figure_directory experiments{i} '-ex-curve_hint.fig'] );
        save2pdf( [figure_directory experiments{i} '-ex-curve_hint.pdf'], gcf, 600, true );
    end
end
