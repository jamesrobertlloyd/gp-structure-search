# Runs all synthetic datasets (the ones with shorter lengthscale)

Experiment(description='Run all synthetic datasets - but the wigglier ones with shorter lengthscales',
           data_dir='../data/synthetic-wigglier-SNR-1/',
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
           results_dir='../results/13-Feb-synthetic-wigglier-SNR-1/',
           iters=100)
           

           
