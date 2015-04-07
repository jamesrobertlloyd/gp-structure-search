This is part of the [automatic statistician](http://www.automaticstatistician.com/) project
========

Gaussian Process Structure Search
===================

<img src="https://raw.githubusercontent.com/jamesrobertlloyd/gp-structure-search/master/logo.png" width="700">

Code for automatically searching through compositions of covariance functions for Gaussian process regression, described in 

[Structure Discovery in Nonparametric Regression through Compositional Kernel Search](http://arxiv.org/abs/1302.4922)
by David Duvenaud, James Robert Lloyd, Roger Grosse, Joshua B. Tenenbaum, Zoubin Ghahramani  

### Abstract:
Despite its importance, choosing the structural form of the kernel in nonparametric regression remains a black art. We define a space of kernel structures which are built compositionally by adding and multiplying a small number of base kernels. We present a method for searching over this space of structures which mirrors the scientific discovery process. The learned structures can often decompose functions into interpretable components and enable long-range extrapolation on time-series datasets. Our structure search method outperforms many widely used kernels and kernel combination methods on a variety of prediction tasks.

Feel free to email us with any questions:  
[James Lloyd](http://mlg.eng.cam.ac.uk/Lloyd/) (jrl44@cam.ac.uk)  
[David Duvenaud](http://people.seas.harvard.edu/~dduvenaud/) (dduvenaud@seas.harvard.edu)  
[Roger Grosse](http://www.cs.toronto.edu/~rgrosse/) (rgrosse@cs.toronto.edu)  

We welcome pull requests and feature suggestions!


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


Updated Repository and Paper:
-------------------------------------------

Development on this project has now moved to [github.com/jamesrobertlloyd/gpss-research/](www.github.com/jamesrobertlloyd/gpss-research/).

Further developments are described in the paper:
[Automatic Construction and Natural-Language Description of Nonparametric Regression Models](http://arxiv.org/pdf/1402.4304.pdf)
by James Robert Lloyd, David Duvenaud, Roger Grosse, Joshua B. Tenenbaum and Zoubin Ghahramani,
appearing in [AAAI 2014](http://www.aaai.org/Conferences/AAAI/aaai14.php).


### Abstract:
This paper presents the beginnings of an automatic statistician, focusing on regression problems. Our system explores an open-ended space of statistical models to discover a good explanation of a data set, and then produces a detailed report with figures and natural-language text. Our approach treats unknown regression functions nonparametrically using Gaussian processes, which has two important consequences. First, Gaussian processes can model functions in terms of high-level properties (e.g. smoothness, trends, periodicity, changepoints). Taken together with the compositional structure of our language of models this allows us to automatically describe functions in simple terms. Second, the use of flexible nonparametric models and a rich language for composing them in an open-ended manner also results in state-of-the-art extrapolation performance evaluated over 13 real time series data sets from various domains.
