"""
Contains helper functions to create figures and tables, based on the results of experiments.

@authors: David Duvenaud (dkd23@cam.ac.uk)
          James Robert Lloyd (jrl44@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
          
February 2013          
"""

import numpy as np
nax = np.newaxis
import os
import random
import scipy.io

import config
import experiment as exp
import flexiblekernel as fk
import gpml
import utils.latex
import re


def parse_all_results(folder=config.D1_RESULTS_PATH, save_file='kernels.tex', one_d=False):
    """
    Creates a list of results, then sends them to be formatted into latex.
    """
    entries = [];
    rownames = [];
    
    colnames = ['Dataset', 'NLL', 'Kernel' ]
    for rt in gen_all_results(folder):
        print "dataset: %s kernel: %s\n" % (rt[0], rt[-1].pretty_print())
        if not one_d:
            entries.append([' %4.1f' % rt[-1].nll, ' $ %s $ ' % rt[-1].latex_print()])
        else:
            # Remove any underscored dimensions
            entries.append([' %4.1f' % rt[-1].nll, ' $ %s $ ' % re.sub('_{[0-9]+}', '', rt[-1].latex_print())])
        rownames.append(rt[0])
    
    utils.latex.table(''.join(['../latex/tables/', save_file]), rownames, colnames, entries)


def gen_all_results(folder=config.RESULTS_PATH):
    """Look through all the files in the results directory"""
    file_list = sorted([f for (r,d,f) in os.walk(folder)][0])
    #for r,d,f in os.walk(folder):
    for files in file_list:
        if files.endswith(".txt"):
            results_filename = os.path.join(folder,files)#r
            best_tuple = exp.parse_results( results_filename )
            yield files.split('.')[-2], best_tuple
                

def make_all_1d_figures(folder=config.D1_RESULTS_PATH, max_level=None):
    data_sets = list(exp.gen_all_datasets("../data/1d_data_rescaled/"))
    for r, file in data_sets:
        results_file = os.path.join(folder, file + "_result.txt")
        # Is the experiment complete
        if os.path.isfile(results_file):
            # Find best kernel and produce plots
            X, y, D = gpml.load_mat(os.path.join(r,file + ".mat"))
            best_kernel = exp.parse_results(os.path.join(folder, file + "_result.txt"), max_level=max_level)
            stripped_kernel = fk.strip_masks(best_kernel.k_opt)
            if not max_level is None:
                fig_folder = os.path.join('../figures/decomposition/', (file + '_max_level_%d' % max_level))
            else:
                fig_folder = os.path.join('../figures/decomposition/', file)
            if not os.path.exists(fig_folder):
                os.makedirs(fig_folder)
            gpml.plot_decomposition(stripped_kernel, X, y, os.path.join(fig_folder, file), noise=best_kernel.noise)
            
def make_all_1d_figures_all_depths(folder=config.D1_RESULTS_PATH, max_depth=8):
    make_all_1d_figures(folder=folder)
    for level in range(max_depth+1):
        make_all_1d_figures(folder=folder, max_level=level)
        
def make_kernel_description_table():
    '''A helper to generate a latex table listing all the kernels used, and their descriptions.'''
    entries = [];
    rownames = [];
    
    colnames = ['', 'Description', 'Parameters' ]
    for k in fk.base_kernel_families():
        # print "dataset: %s kernel: %s\n" % (rt[0], rt[-1].pretty_print())
        rownames.append( k.latex_print() )
        entries.append([ k.family().description(), k.family().params_description()])
    
    utils.latex.table('../latex/tables/kernel_descriptions.tex', rownames, colnames, entries, 'kernel_descriptions')
