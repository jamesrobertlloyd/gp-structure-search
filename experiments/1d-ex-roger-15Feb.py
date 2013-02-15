# Runs all 1d extrapolations.

Experiment(description='Run all 1D extrapolations',
           data_dir='../data/1d_extrap_roger_folds/',
           max_depth=8, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=False, 
           n_rand=3,
           sd=4, 
           max_jobs=500, 
           verbose=False,
           make_predictions=True,
           skip_complete=True,
           results_dir='../results/15-Feb-1d-extrap-roger/',
           iters=100)
           

           
