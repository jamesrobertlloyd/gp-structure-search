Gaussian Process Structure Search
===================

A framework for automatically searching through compositions of covariance functions for Gaussian process regression.

We welcome pull requests and feature suggestions!

To read about some preliminary experiments using this code, see:

*Structure Discovery in Nonparametric Regression through Compositional Kernel Search*  
by David Duvenaud, James Robert Lloyd, Roger Grosse, Joshua B. Tenenbaum, Zoubin Ghahramani  
http://arxiv.org/abs/1302.4922


Feel free to email us with any questions:  
[James Lloyd](http://mlg.eng.cam.ac.uk/Lloyd/) (jrl44@cam.ac.uk)  
[David Duvenaud](http://mlg.eng.cam.ac.uk/duvenaud/) (dkd23@cam.ac.uk)  
[Roger Grosse](http://people.csail.mit.edu/rgrosse/) (rgrosse@mit.edu)  


### Instructions:

You'll need Matlab and Python 2.7 with numpy.

You'll also need to create `source/cblparallel/config.py` - follow the format of the example file in the same directory.

To check whether the framework runs, go to the source directory and run `demo.py`.

There are some example experiment scripts `source/examples/`.

Many helper functions to summarize results are in `postprocessing.py`.  For example, to produce nice plots of your decomposition, call `make_all_1d_figures()`


If you have any questions about getting this running on your machine or cluster, please let us know.

If describe your problem to us, we'll also happy to give advice about running the method.

### Known issues:

Windows users will need to make a change

* All strings in config files are not sanitized, therefore backslashes and other special characters should be delimited e.g. `C:\\ProgramFiles\\Matlab`
