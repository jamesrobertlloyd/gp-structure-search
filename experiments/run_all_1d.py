# Runs all 1d datasets.

Experiment(description='Run all 1d datasets, not making predictions',
           data_dir='../data/1d_data/',
           output_dir='results/1d_data',
           max_depth=3, 
           random_order=False,
           k=1,
           debug=True, 
           local_computation=True, 
           n_rand=3,
           sd=2, 
           max_jobs=500, 
           verbose=False,
           make_predictions=False,
           skip_complete=True);
           

           
