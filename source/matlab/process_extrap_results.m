results_directory = '../../results/15-Feb-1d-extrap-roger/';
full_data_directory = '../../data/1d_data_rescaled/';
fold_data_directory = '../../data/1d_extrap_roger_folds/';
figure_directory = '../../figures/extrap_roger_curves/';
clear experiments
experiments{1} = '01-airline-s';
experiments{2} = '02-solar-s';
experiments{1} = '03-mauna2003-s';
folds = 10;
percentiles = 100 * (1:(folds-1)) / folds;

quick_test_mode = false;

restart_sd = 4;
restarts = 10;

max_iters = 100; % Max iterations for GP training.
lw = 2;        % Line width.

for i = 1:length(experiments)
    MSEs.gpss = zeros(folds-1,1);
    MSEs.gpse = zeros(folds-1,1);
    MSEs.lin = zeros(folds-1,1);
    if ~quick_test_mode
        MSEs.gpper = zeros(folds-1,1);
        MSEs.gpadd = zeros(folds-1,1);
        MSEs.gpprod = zeros(folds-1,1);
    end
    for fold = 1:(folds-1)
        fold
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
        MSEs.gpss(fold) = mean((actuals - predictions) .^ 2);
        % Fit and score SE GP
        %%%% Random restarts + averaging?
        covfunc = {@covSEiso};
        likfunc = @likGauss;
        hyp.lik = log(std(y));
        meanfunc = {@meanConst};
        hyp.mean = mean(y);
        best_nll = Inf;
        best_MSE = Inf;
        for unused = 1:restarts
            hyp.cov = [0; 0];
            hyp.cov = hyp.cov + restart_sd * randn(size(hyp.cov));
            [hyp_opt nlls] = minimize(hyp, @gp, -max_iters, @infExact, ...
                                      meanfunc, covfunc, likfunc, X, y);
            if nlls(end) < best_nll
                best_nll = nlls(end);
                [m s2] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc,...
                            X, y, Xtest);
                best_MSE = mean((ytest - m) .^ 2);
            end
        end
        MSEs.gpse(fold) = best_MSE;
        % Fit and score linear model
        %%%% Random restarts + averaging?
        covfunc = {@covSum, {@covConst, @covLINard}};
        likfunc = @likGauss;
        hyp.lik = log(std(y));
        meanfunc = {@meanConst};
        hyp.mean = mean(y);
        best_nll = Inf;
        best_MSE = Inf;
        for unused = 1:restarts
        hyp.cov = [0; 0];
            hyp.cov = hyp.cov + restart_sd * randn(size(hyp.cov));
            [hyp_opt nlls] = minimize(hyp, @gp, -max_iters, @infExact, ...
                                      meanfunc, covfunc, likfunc, X, y);
            if nlls(end) < best_nll
                best_nll = nlls(end);
                [m s2] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc,...
                            X, y, Xtest);
                best_MSE = mean((ytest - m) .^ 2);
            end
        end
        MSEs.lin(fold) = best_MSE;
        if ~quick_test_mode
            % Fit and score pure periodic
            %%%% Random restarts + averaging?
            covfunc = {@covPeriodic};
            likfunc = @likGauss;
            hyp.lik = log(std(y));
            meanfunc = {@meanConst};
            hyp.mean = mean(y);
            best_nll = Inf;
            best_MSE = Inf;
            for unused = 1:restarts
                hyp.cov = [0; -2; 0];
                hyp.cov = hyp.cov + restart_sd * randn(size(hyp.cov));
                [hyp_opt nlls] = minimize(hyp, @gp, -max_iters, @infExact, ...
                                          meanfunc, covfunc, likfunc, X, y);
                if nlls(end) < best_nll
                    best_nll = nlls(end);
                    [m s2] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc,...
                                X, y, Xtest);
                    best_MSE = mean((ytest - m) .^ 2);
                end
            end
            MSEs.gpper(fold) = best_MSE;
            % Fit and score pSE + Per
            %%%% Random restarts + averaging?
            covfunc = {@covSum, {@covSEiso, @covPeriodic}};
            likfunc = @likGauss;
            hyp.lik = log(std(y));
            meanfunc = {@meanConst};
            hyp.mean = mean(y);
            best_nll = Inf;
            best_MSE = Inf;
            for unused = 1:restarts
                hyp.cov = [0; 0; 0; -2; 0];
                hyp.cov = hyp.cov + restart_sd * randn(size(hyp.cov));
                [hyp_opt nlls] = minimize(hyp, @gp, -max_iters, @infExact, ...
                                          meanfunc, covfunc, likfunc, X, y);
                if nlls(end) < best_nll
                    best_nll = nlls(end);
                    [m s2] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc,...
                                X, y, Xtest);
                    best_MSE = mean((ytest - m) .^ 2);
                end
            end
            MSEs.gpadd(fold) = best_MSE;
            % Fit and score SE * Per
            %%%% Random restarts + averaging?
            covfunc = {@covProd, {@covSEiso, @covPeriodic}};
            likfunc = @likGauss;
            hyp.lik = log(std(y));
            meanfunc = {@meanConst};
            hyp.mean = mean(y);
            best_nll = Inf;
            best_MSE = Inf;
            for unused = 1:restarts
                hyp.cov = [0; 0; 0; -2; 0];
                hyp.cov = hyp.cov + restart_sd * randn(size(hyp.cov));
                [hyp_opt nlls] = minimize(hyp, @gp, -max_iters, @infExact, ...
                                          meanfunc, covfunc, likfunc, X, y);
                if nlls(end) < best_nll
                    best_nll = nlls(end);
                    [m s2] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc,...
                                X, y, Xtest);
                    best_MSE = mean((ytest - m) .^ 2);
                end
            end
            MSEs.gpprod(fold) = best_MSE;
        end
    end
    
    %hold on
    semilogy(percentiles, MSEs.lin, '-', 'Color', colorbrew(3), 'LineWidth', lw);
    hold on
    semilogy(percentiles, MSEs.gpse, '-', 'Color', colorbrew(2), 'LineWidth', lw);
    if ~quick_test_mode
        semilogy(percentiles, MSEs.gpper, '-', 'Color', colorbrew(4), 'LineWidth', lw);
        semilogy(percentiles, MSEs.gpadd, '-', 'Color', colorbrew(5), 'LineWidth', lw);
        semilogy(percentiles, MSEs.gpprod, '-', 'Color', colorbrew(10), 'LineWidth', lw);
    end
    semilogy(percentiles, MSEs.gpss, '-', 'Color', colorbrew(1), 'LineWidth', lw);  

    xlabel('Proportion of training data (%)');
    ylabel('MSE');
    if quick_test_mode
        legend('Linear', 'SE GP', 'Structure search', 'location', 'best');
    else
        legend('Linear', 'SE GP', 'Periodic GP', 'SE + Per GP', 'SE x Per GP', 'Structure search', 'location', 'best');
    end
    legend('boxoff')
    hold off
    xlim([10,90]);
    set_fig_units_cm( 12,11 );
    if ~quick_test_mode
        saveas( gcf, [figure_directory experiments{i} '-ex-curve.fig'] );
        save2pdf( [figure_directory experiments{i} '-ex-curve.pdf'], gcf, 600, true );
    end
    save([figure_directory experiments{i} '-MSEs.mat']);
end
