Experiment(description='Test the new scaling code',
           data_dir='../data/time_series_unscaled/',
           max_depth=8, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=False, 
           n_rand=3,
           sd=4, 
           max_jobs=400, 
           verbose=False,
           make_predictions=False,
           skip_complete=True,
           results_dir='../results/Apr_29_1D_scaling_test/',
           iters=100,
           base_kernels='SE,RQ,Per,Lin,Const',
           zero_mean=True,
           random_seed=0,
           period_heuristic=5)
           

           
