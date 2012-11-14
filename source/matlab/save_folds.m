% Save the different folds to files.
% for the gp-kernel learning paper.
%
% David Duvenaud
% Nov 2012
% =======================

addpath(genpath(pwd))
addpath('../utils/');

K = 10;
seed = 0;
outdir = 'results/';

[classification_datasets, classification_methods, ...
          regression_datasets, regression_methods] = define_datasets_and_methods();
    

for d_ix = 1:length(regression_datasets)
    dataset = regression_datasets{d_ix};
    [ia, shortname] = fileparts(dataset);
    
    load(dataset);
    all_X = X;
    all_y = y;
    assert(size(all_X,1) == size(all_y,1));
    assert(size(all_y,2) == 1 );

    % Normalize the data.
    all_X = all_X - repmat(mean(all_X), size(all_X,1), 1 );
    all_X = all_X ./ repmat(std(all_X), size(all_X,1), 1 );

    % Only normalize the y if it's not a classification experiment. Hacky.
    if ~all(all_y == 1 | all_y == -1 )
        all_y = all_y - mean(all_y);
        all_y = all_y / std(all_y);
    end


    % Reset the random seed, always the same for the datafolds.
    randn('state', 0);
    rand('twister', 0);    
    % Generate the folds, which should be the same for each call.
    %perm = randperm(size(y,1));
    perm = 1:size(all_y,1); fprintf('\n Not randomizing dataset order\n');
    [trainfolds, testfolds] = gen_kfolds(length(all_y), K, perm);
    
    for fold = 1:K
        X = all_X(trainfolds{fold},:);
        y = all_y(trainfolds{fold});
        Xtest = all_X(testfolds{fold},:);
        ytest = all_y(testfolds{fold});
        
        outputname = sprintf( 'kfold_data/%s_fold_%d_of_%d.mat', shortname, fold, K );
        fprintf('saving %s\n', outputname );
        save( outputname, 'X', 'y', 'Xtest', 'ytest' );
    end
end
