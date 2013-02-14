# Runs all synthetic datasets.

Experiment(description='Run all synthetic datasets',
           data_dir='../data/synthetic-SNR-tenth/',
           max_depth=8, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=False, 
           n_rand=19,
           sd=4, 
           max_jobs=500, 
           verbose=False,
           make_predictions=False,
           skip_complete=True,
           results_dir='../results/14-Feb-synthetic-SNR-tenth/',
           iters=100)
           

           
