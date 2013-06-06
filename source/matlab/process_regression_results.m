%% Read results and print latex format table

experiment_name = '11 Mar';

experiments = {};
experiments{1} = 'bach_synth_r_200';
experiments{2} = 'r_concrete_500';
experiments{3} = 'r_pumadyn512';
experiments{4} = 'r_servo';
experiments{5} = 'r_housing';

%folder = '../saved_results/9-Jan/';
%folder = '../saved_results/18-Jan/';
%folder = '../saved_results/22-Jan/';
%folder = '../../results/';
%folder = '../../saved_results/28-Jan/';
folder = '../../results/11-Mar/';
%folder = '../../results/May-13-no-RQ/';

MSEs = zeros(length(experiments),1);
liks = zeros(length(experiments),1);

for i = 1:length(experiments)
    MSE = 0;
    lik = 0;
    for fold = 1:10
        data = load([folder experiments{i} '_fold_' int2str(fold) '_of_10_predictions.mat']);
        lik = lik + mean(data.loglik);
        MSE = MSE + mean((data.actuals - data.predictions) .^ 2);
    end
    MSE = MSE / 10;
    lik = lik / 10;
    MSEs(i) = MSE;
    liks(i) = lik;
end

fprintf('\n');
fprintf(experiment_name);
fprintf(' & %1.3f', MSEs);
fprintf('\\\\\n');
fprintf(experiment_name);
fprintf(' & %1.3f', -liks);
fprintf('\\\\\n');
