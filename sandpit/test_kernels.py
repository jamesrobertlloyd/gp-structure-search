# A test

Experiment(description='An example experiment',
           data_dir='../data/1d_data/01-airline.mat',
           results_dir='../examples/test_kernels',
           max_depth=2,                # How deep to run the search.
           k=1,                        # Keep the k best kernels at every iteration.  1 => greedy search.
           n_rand=2,                   # Number of random restarts.
           iters=10,                   # How long to optimize hyperparameters for.
           base_kernels='SE,Per,Lin,RQ,MT,PP0,PP1,PP2,PP3',
           verbose=True,
           make_predictions=False,
           skip_complete=False,
           use_constraints=True)
