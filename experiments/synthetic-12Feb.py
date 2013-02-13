# Runs all synthetic datasets.

Experiment(description='Run all synthetic datasets',
           data_dir='../data/synthetic/',
           max_depth=8, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=False, 
           n_rand=8,
           sd=4, 
           max_jobs=500, 
           verbose=False,
           make_predictions=False,
           skip_complete=True,
           results_dir='../results/12-Feb-synthetic/',
           iters=100)
           

           
