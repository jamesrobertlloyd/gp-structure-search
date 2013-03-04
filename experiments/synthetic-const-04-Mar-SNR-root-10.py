# Runs all synthetic datasets

Experiment(description='Run all synthetic datasets',
           data_dir='../data/synthetic-SNR-root-10/',
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
           results_dir='../results/04-Mar-synthetic-const-SNR-root-10/',
           iters=100,
           base_kernels='SE,RQ,Per,Lin,Const',
           zero_mean=True)
           

           
