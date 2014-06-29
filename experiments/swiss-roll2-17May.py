# Runs on all swiss roll datasets, quickly.

Experiment(description='Swiss roll normalized slow',
           data_dir='../data/swiss-roll2/',
           max_depth=10, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=False, 
           n_rand=3,
           sd=4, 
           max_jobs=400, 
           verbose=False,
           make_predictions=True,
           skip_complete=True,
           results_dir='../results/swiss-roll4/',
           iters=200,
           base_kernels='SE,Lin,Per',
           use_min_period=False,
           zero_mean=False)
           

           
