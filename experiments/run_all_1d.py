# Runs all 1d datasets.

Experiment(description='Run all 1D datasets',
           data_dir='../data/1d_data/',
           max_depth=8, 
           random_order=False,
           k=2,
           debug=False, 
           local_computation=True, 
           n_rand=3,
           sd=4, 
           max_jobs=500, 
           verbose=True,
           make_predictions=False,
           skip_complete=True,
           results_dir='../results/Feb 10 1D results/',
           iters=200)
           

           
