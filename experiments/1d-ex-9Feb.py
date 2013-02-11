# Runs all 1d datasets.

Experiment(description='Run all 1D datasets',
           data_dir='../data/1d_extrap_folds/',
           max_depth=8, 
           random_order=False,
           k=1,
           debug=True, 
           local_computation=False, 
           n_rand=9,
           sd=4, 
           max_jobs=500, 
           verbose=False,
           make_predictions=True,
           skip_complete=True,
           results_dir='../results/9-Feb 1d learning curves/',
           iters=200)
           

           
