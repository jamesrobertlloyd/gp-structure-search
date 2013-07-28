Experiment(description='Does it work on GEFcom',
           data_dir='../data/gefcom/',
           max_depth=8, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=True, 
           n_rand=3,
           sd=4, 
           max_jobs=1000, 
           verbose=False,
           make_predictions=False,
           skip_complete=True,
           results_dir='../results/Jul_22_gefcom_test/',
           iters=100,
           base_kernels='SE,RQ,Per,Const',
           zero_mean=True,
           random_seed=0,
           period_heuristic=5)
           

           
