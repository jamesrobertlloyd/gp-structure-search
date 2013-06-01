Experiment(description='Test the new scaling code without RQ',
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
           results_dir='../results/May_13_no_RQ/',
           iters=100,
           base_kernels='SE,Per,Lin,Const',
           zero_mean=True,
           random_seed=0,
           period_heuristic=5)
           

           
