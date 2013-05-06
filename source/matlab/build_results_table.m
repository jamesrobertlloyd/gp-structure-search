% Read results from a set of directories and build a latex table of 
% results for both nll and mse.
%
% Inputs:
%     num_folds: number of folds of each dataset.
%     results_dirs: a cell array of directories where results files live.
%     experiment_names: a cell array of strings which will become row names.
%     tex_prefix: the prefix of the outputted tavle files.
%     use_additive_results: also include results from the additive GPs paper.
%
%
% David Duvenaud
% Jan 2013
% ==========================

function build_results_table( num_folds, results_dirs, experiment_names, tex_prefix, ...
                              use_additive_results )

dataset_names = {};
dataset_names{1} = 'bach_synth_r_200';
dataset_names{2} = 'r_concrete_500';
dataset_names{3} = 'r_pumadyn512';
dataset_names{4} = 'r_servo';
dataset_names{5} = 'r_housing';

if nargin < 1; num_folds = 10; end
%if nargin < 2; results_dirs = { '/scratch/Dropbox/results/22-Jan/', '/scratch/Dropbox/results/28-Jan/' }; end
%if nargin < 2; results_dirs = { '../../results/4-Feb/', '../../results/28-Jan/', '../../results/11-Mar/'  }; end
if nargin < 2; results_dirs = { '../../results/11-Mar/'  }; end
%if nargin < 3; experiment_names = { '$\\SE{}$', '$\\SE{}, \\RQ{}$', '$\\SE{}, \\RQ{}, \\Lin{}, \\Per{}$' }; end
if nargin < 3; experiment_names = { '$\\SE{}, \\RQ{}, \\Lin{}$' }; end
%if nargin < 4; tex_prefix = '../latex/tables/regression_results'; end
if nargin < 4; tex_prefix = '../../latex/tables/regression_results_ext_onlyall'; end
if nargin < 5; use_additive_results = true; end

num_new_methods = numel(results_dirs);
num_datasets = length(dataset_names);

    
if use_additive_results
    % Combine the results of the 2011 Additive GPs paper with the new results.
    load 'matlab/nips2011_fold_results.mat';
    nips2011_regression_methods = regression_methods;
    nips2011_mse_all = mse_all;
    nips2011_likelihood_all = likelihood_all;
    nips2011_method_names = method_names;
    nips2011_hkl_ix = hkl_ix;
    
    num_nips_methods = numel(nips2011_regression_methods);
    num_methods_total = num_new_methods + num_nips_methods;
    
    mse_all = zeros(num_folds, num_datasets, num_methods_total);
    mse_all(:, :, 1:num_nips_methods) = nips2011_mse_all;
    
    likelihood_all = zeros(num_folds, num_datasets, num_methods_total);
    likelihood_all(:, :, 1:num_nips_methods) = nips2011_likelihood_all;
    
    experiment_names = {method_names{:}, experiment_names{:}};
else
    % Start from scratch.
    mse_all = zeros(num_folds, num_datasets, num_methods);
    likelihood_all = zeros(num_folds, num_datasets, num_methods);
end

% Go through each results directory, and load the results into a giant table.
for m = 1:num_new_methods
    for d = 1:num_datasets
        for fold = 1:10
            data = load([results_dirs{m}, dataset_names{d}, '_fold_', int2str(fold), '_of_10_predictions.mat']);
            likelihood_all(fold, d, m + num_nips_methods) = mean(data.loglik);
            mse_all(fold, d, m + num_nips_methods) = mean((data.actuals - data.predictions) .^ 2);
            fprintf('.');
        end
        fprintf('\n');
    end
    fprintf('\n\n');
end

experiment_names = replace_method_names(experiment_names);
dataset_names = shorten_names(dataset_names);

% Turn the results tables into latex tables.
mse_table_file = [tex_prefix, '_mse.tex'];
fprintf('Writing to %s\n', mse_table_file);
resultsToLatex4( mse_table_file, mse_all, experiment_names, dataset_names, ....
                 'Regression Mean Squared Error' );

if use_additive_results
    %remove HKL log likelihood, since it isn't defined.
    experiment_names(hkl_ix) = []; likelihood_all(:,:,hkl_ix) = [];
end

nll_table_file = [tex_prefix, '_nll.tex'];
fprintf('Writing to %s\n', nll_table_file);
resultsToLatex4( nll_table_file, -likelihood_all, experiment_names, ...
                 dataset_names, 'Regression Negative Log Likelihood' );
end

function nicenames = shorten_names( names )
    for i = 1:length(names)
        nicenames{i} = names{i};
        nicenames{i} = strrep(nicenames{i}, '_', ' ' );
        nicenames{i} = strrep(nicenames{i}, '0', '' );
        nicenames{i} = strrep(nicenames{i}, '1', '' );
        nicenames{i} = strrep(nicenames{i}, '2', '' );
        nicenames{i} = strrep(nicenames{i}, '3', '' );
        nicenames{i} = strrep(nicenames{i}, '4', '' );
        nicenames{i} = strrep(nicenames{i}, '5', '' );
        nicenames{i} = strrep(nicenames{i}, '6', '' );
        nicenames{i} = strrep(nicenames{i}, '7', '' );
        nicenames{i} = strrep(nicenames{i}, '8', '' );
        nicenames{i} = strrep(nicenames{i}, '9', '' );        
        nicenames{i} = strrep(nicenames{i}, 'synth ', '' );
        nicenames{i} = strrep(nicenames{i}, 'c ', '' );
        nicenames{i} = strrep(nicenames{i}, 'r ', '' );
        nicenames{i} = strrep(nicenames{i}, 'sola', 'solar' );
        nicenames{i} = strrep(nicenames{i}, 'pumadyn', 'puma');%dyn-8nh' );        
    end
end

function nicenames = replace_method_names( names )
    for i = 1:length(names)
        nicenames{i} = names{i};
        nicenames{i} = strrep(nicenames{i}, 'gp_ard_class', 'GP Squared-exp' );
        nicenames{i} = strrep(nicenames{i}, 'gp_ard', 'GP Squared-exp' );
        nicenames{i} = strrep(nicenames{i}, 'gp_add_lo_1', 'GP GAM' );
        nicenames{i} = strrep(nicenames{i}, 'gp_add_lo', 'GP Additive' );
        nicenames{i} = strrep(nicenames{i}, 'gp_add_class_lo_1', 'GP GAM' );
        nicenames{i} = strrep(nicenames{i}, 'gp_add_class_lo', 'GP Additive' );
        nicenames{i} = strrep(nicenames{i}, 'lin_model', 'Linear Regression' );
        nicenames{i} = strrep(nicenames{i}, 'logistic', 'Logistic Regression' );
        nicenames{i} = strrep(nicenames{i}, 'hkl_regression', 'HKL' );
        nicenames{i} = strrep(nicenames{i}, 'hkl_classification', 'HKL' );
    end
end
