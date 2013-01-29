load '/tmp/tmpr5H6B6.mat'  % Load the data, it should contain X and y.

addpath(genpath('/home/jrl44/Documents/MATLAB/GPML'));
addpath(genpath('/scratch/home/Research/GPs/gp-structure-search/source/matlab'));

plot_decomp(X, y, {@covProd, {{@covMask, {[1], {@covPeriodic}}}, {@covMask, {[1], {@covRQiso}}}}}, [ 1.752912 -0.000229 -0.100783 2.389821 3.094452 -3.307324 ], { {@covProd, {{@covMask, {[1], {@covPeriodic}}}, {@covMask, {[1], {@covRQiso}}}}} }, { [ 1.752912 -0.000229 -0.100783 2.389821 3.094452 -3.307324 ] }, [-1.48313956], '/scratch/home/Research/GPs/gp-structure-search/figures/decomposition/test', { ' PE_{0} \times RQ_{0} ' })
