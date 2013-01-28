%% Read results and print latex format table

experiment_name = '28 Jan';

experiments = {};
experiments{1} = 'bach_synth_r_200';
experiments{2} = 'r_concrete_500';
experiments{3} = 'r_pumadyn512';
experiments{4} = 'r_servo';
experiments{5} = 'r_housing';

%folder = '../saved_results/9-Jan/';
%folder = '../saved_results/18-Jan/';
%folder = '../saved_results/22-Jan/';
%folder = '../results/';
folder = '../../saved_results/28-Jan/';

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

%% Old code

experiment = 'bach_synth_r_200';
% experiment = 'r_concrete_500';
% experiment = 'r_pumadyn512';
% experiment = 'r_servo';
% experiment = 'r_housing';

%folder = '../saved_results/9-Jan/';
%folder = '../saved_results/18-Jan/';
%folder = '../saved_results/22-Jan/';
folder = '../../saved_results/28-Jan/';
%folder = '../results/';

files = {};
for i = 1:10
  files{i} = [folder experiment '_fold_' int2str(i) '_of_10_predictions.mat'];
end

sum_log_lik = 0;
sum_MSE = 0;
for i =1:10
  data = load(files{i});
  sum_log_lik = sum_log_lik + mean(data.loglik);
  sum_MSE = sum_MSE + mean((data.actuals - data.predictions) .^ 2);
end

log_lik = sum_log_lik / 10;
MSE = sum_MSE / 10;

fprintf ('\nNegative Loglik = %f, MSE = %f\n\n', -log_lik, MSE);
