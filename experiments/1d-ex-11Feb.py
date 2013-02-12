# Runs all 1d datasets.

Experiment(description='Run all 1D datasets',
           data_dir='../data/1d_extrap_folds/',
           max_depth=8, 
           random_order=False,
           k=4,
           debug=False, 
           local_computation=False, 
           n_rand=3,
           sd=4, 
           max_jobs=500, 
           verbose=False,
           make_predictions=True,
           skip_complete=True,
           results_dir='../results/11-Feb-1d-extrap/',
           iters=200)
           

           
