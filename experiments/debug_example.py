# An example experiment.

Experiment(description='An example experiment for debugging',
           data_dir='../data/kfold_data/r_pumadyn512_fold_3_of_10.mat',
           max_depth=2, 
           random_order=False,
           k=1,
           debug=True, 
           local_computation=True, 
           n_rand=2,
           sd=2, 
           max_jobs=500, 
           verbose=False,
           make_predictions=True,
           skip_complete=True,
           results_dir='../results/debug_results/',
           iters=10)
