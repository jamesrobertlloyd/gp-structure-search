# An example experiment that will take a while, but will probably find a good solution.

Experiment(description='An example experiment',
           data_dir='../data/1d_data/01-airline.mat',
           results_dir='../examples/',
           max_depth=10,                # How deep to run the search.
           k=1,                         # Keep the k best kernels at every iteration.  1 => greedy search.
           n_rand=2,                    # Number of random restarts.
           iters=100,                   # How long to optimize hyperparameters for.
           base_kernels='SE,Per,Lin',
           verbose=False,
           make_predictions=False,
           skip_complete=True)
