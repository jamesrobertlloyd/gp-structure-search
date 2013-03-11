# Runs all multi-d datasets

Experiment(description='Run all multi D datasets',
           data_dir='../data/kfold_data/',
           max_depth=12, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=False, 
           n_rand=2,
           sd=4, 
           max_jobs=400, 
           verbose=False,
           make_predictions=True,
           skip_complete=True,
           results_dir='../results/11-Mar/',
           iters=100,
           base_kernels='SE,RQ,Lin,Const',
           zero_mean=True)
           

           
