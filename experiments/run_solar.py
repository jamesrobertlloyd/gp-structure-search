# Runs all 1d datasets.

Experiment(description='Run solar dataset.',
           data_dir='../data/1d_data/02-solar.mat',
           max_depth=8, 
           random_order=False,
           k=1,
           debug=True, 
           local_computation=False, 
           n_rand=3,
           sd=4, 
           max_jobs=500, 
           verbose=False,
           make_predictions=False,
           skip_complete=True,
           results_dir='../results/solar_results/',
           iters=10)
           

           
