# An example experiment.

Experiment(description='An example experiment for debugging',
           data_dir='../data/1d_data/01-airline.mat',
           max_depth=2, 
           random_order=False,
           k=1,
           debug=True, 
           local_computation=True, 
           n_rand=2,
           sd=2, 
           max_jobs=500, 
           verbose=True,
           make_predictions=False,
           skip_complete=False,
           results_dir='../examples/',
           iters=10,
           base_kernels='SE,RQ,Per,Lin,Const',
           zero_mean=True)
