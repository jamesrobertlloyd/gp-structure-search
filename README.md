gp-structure-search
===================

A framework for automatically searching through compositions of covariance functions for Gaussian process regression.

We welcome pull requests and feature suggestions!


To read about some preliminary experiments using this code, see:

Structure Discovery in Nonparametric Regression through Compositional Kernel Search
by David Duvenaud, James Robert Lloyd, Roger Grosse, Joshua B. Tenenbaum, Zoubin Ghahramani
http://arxiv.org/abs/1302.4922


Feel free to email us:
James Lloyd (jrl44@cam.ac.uk)
David Duvenaud (dkd23@cam.ac.uk)
Roger Grosse (rgrosse@mit.edu)


Install instructions:

1. Install the GPML toolkit: http://www.gaussianprocess.org/gpml/code/matlab/doc/

2. Copy the custom covariances in source/matlab/custom_cov/ to the cov/ directory in your GPML install.

3. create a config.py file, something like:
MATLAB_LOCATION = "/misc/apps/matlab/matlabR2011b/bin/matlab"
GPML_PATH = '/home/user/gpml/'
COLOR_SCHEME = 'dark'
LOCATION = 'local'
LOCAL_TEMP_PATH = '/home/user/git/gp-structure-search/temp'

4. Modify one of the experiments in experiments/ to suit your needs.  Set local_computation=True.

5. In a Python interpreter, import experiment.py, and call run_experiment_file(filename).

6. Automated kernel discovery!

7. The results of the kernel search will be in a text file.  Many helper functions to summarize results are in postprocessing.py.  For example, to produce nice plots of your decomposition, call make_all_1d_figures()


If you have any questions about getting this running on your machine, please let us know.

If describe your problem to us, we'd be happy to give advice about running the method.
