# Testing list of kernels

Experiment(description='Testing list of kernels',
           data_dir='../data/kfold_data/',
           max_depth=2, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=False, 
           n_rand=1,
           sd=4, 
           max_jobs=500, 
           verbose=False,
           make_predictions=True,
           skip_complete=True,
           results_dir='../results/26-Feb/',
           iters=250,
           base_kernels='Per,Lin')
           

           
