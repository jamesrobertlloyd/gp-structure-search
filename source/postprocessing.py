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


def parse_all_results(folder, save_file='kernels.tex', one_d=False):
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


def gen_all_results(folder):
    """Look through all the files in the results directory"""
    file_list = sorted([f for (r,d,f) in os.walk(folder)][0])
    #for r,d,f in os.walk(folder):
    for files in file_list:
        if files.endswith(".txt"):
            results_filename = os.path.join(folder,files)#r
            best_tuple = exp.parse_results( results_filename )
            yield files.split('.')[-2], best_tuple
                

def make_all_1d_figures(folder, save_folder='../figures/decomposition/', max_level=None, prefix='', rescale=True, data_folder=None):
    """Crawls the results directory, and makes decomposition plots for each file.
    
    prefix is an optional string prepended to the output directory
    """
    #### Quick fix to axis scaling
    #### TODO - Ultimately this and the shunt below should be removed / made elegant
    if rescale:
        data_sets = list(exp.gen_all_datasets("../data/1d_data_rescaled/"))
    else:
        if data_folder is None:
            data_sets = list(exp.gen_all_datasets("../data/1d_data/"))
        else:
            data_sets = list(exp.gen_all_datasets(data_folder))
    for r, file in data_sets:
        results_file = os.path.join(folder, file + "_result.txt")
        # Is the experiment complete
        if os.path.isfile(results_file):
            # Find best kernel and produce plots
            datafile = os.path.join(r,file + ".mat")
            X, y, D = gpml.load_mat(datafile)
            if rescale:
                # Load unscaled data to remove scaling later
                unscaled_file = os.path.join('../data/1d_data/', re.sub('-s$', '', file) + '.mat')
                data = gpml.load_mat(unscaled_file)
                (X_unscaled, y_unscaled) = (data[0], data[1])
                (X_mean, X_scale) = (X_unscaled.mean(), X_unscaled.std())
                (y_mean, y_scale) = (y_unscaled.mean(), y_unscaled.std())
            else:
                (X_mean, X_scale, y_mean, y_scale) = (0,1,0,1)
                
            # A shunt to deal with a legacy issue.
            if datafile == '../data/1d_data/01-airline-months.mat':
                # Scaling should turn months starting at zero into years starting at 1949
                print "Special rescaling for airline months data"
                X_mean = X_mean + 1949
                X_scale = 1.0/12.0
                                
            best_kernel = exp.parse_results(os.path.join(folder, file + "_result.txt"), max_level=max_level)
            stripped_kernel = fk.strip_masks(best_kernel.k_opt)
            if not max_level is None:
                fig_folder = os.path.join(save_folder, (prefix + file + '_max_level_%d' % max_level))
            else:
                fig_folder = os.path.join(save_folder, (prefix + file))
            if not os.path.exists(fig_folder):
                os.makedirs(fig_folder)
            gpml.plot_decomposition(stripped_kernel, X, y, os.path.join(fig_folder, file), best_kernel.noise, X_mean, X_scale, y_mean, y_scale)
        else:
            print "Cannnot find file %s" % results_file
            
def make_all_1d_figures_all_depths(folder, max_depth=10, prefix=''):
    make_all_1d_figures(folder=folder)
    for level in range(max_depth+1):
        make_all_1d_figures(folder=folder, max_level=level, prefix=prefix)
        
def compare_1d_decompositions():
    '''Produces the decomposition for all the files in the listed directories - to see which one to pick'''
    folders = ['../results/4-Feb-1d', 
               '../results/5-Feb-1d-NewLin', 
               '../results/5-Feb-1d-OldLin', 
               '../results/6-Feb-1d-More-Restarts', 
               '../results/6-Feb-1d-Even-More-Restarts', 
               '../results/8-Feb-1d-collated', 
               '../results/9-Feb-1d', 
               '../results/10-Feb-1d']
    for folder in folders:
        make_all_1d_figures(folder=folder, save_folder='../temp_figures/' + folder.split('/')[-1])
        
def collate_decompositions(top_folder, tex):
    '''Produces a LaTeX document with all decompositions displayed'''
    latex_header = '''
\documentclass[twoside]{article}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{amssymb,amsmath,amsthm}
\usepackage{graphicx}
\usepackage{preamble}
\usepackage{natbib}
%%%% REMEMBER ME!
%\usepackage[draft]{hyperref}
\usepackage{hyperref}
\usepackage{color}
\usepackage{wasysym}
\usepackage{subfigure}
\usepackage{tabularx}
\usepackage{booktabs}
\usepackage{bm}
\\newcommand{\\theHalgorithm}{\\arabic{algorithm}}
\definecolor{mydarkblue}{rgb}{0,0.08,0.45}
\hypersetup{ %
    pdftitle={},
    pdfauthor={},
    pdfsubject={},
    pdfkeywords={},
    pdfborder=0 0 0,
    pdfpagemode=UseNone,
    colorlinks=true,
    linkcolor=mydarkblue,
    citecolor=mydarkblue,
    filecolor=mydarkblue,
    urlcolor=mydarkblue,
    pdfview=FitH}

\\newcolumntype{x}[1]{>{\centering\\arraybackslash\hspace{0pt}}m{#1}}
\\newcommand{\\tabbox}[1]{#1}

\setlength{\marginparwidth}{0.6in}
\input{include/commenting.tex}

\\newif\ifarXiv
%\\arXivtrue

\ifarXiv
	\usepackage[arxiv]{format/icml2013}
\else
	\usepackage[accepted]{format/icml2013}
\\fi
%\usepackage[left=1.00in,right=1.00in,bottom=0.25in,top=0.25in]{geometry} %In case we want larger margins for commenting purposes

%% For submission, make all render blank.
%\\renewcommand{\LATER}[1]{}
%\\renewcommand{\\fLATER}[1]{}
%\\renewcommand{\TBD}[1]{}
%\\renewcommand{\\fTBD}[1]{}
%\\renewcommand{\PROBLEM}[1]{}
%\\renewcommand{\\fPROBLEM}[1]{}
%\\renewcommand{\NA}[1]{#1}  %% Note, NA's pass through!

    
\\begin{document}

%\\renewcommand{\\baselinestretch}{0.99}

\\twocolumn[
\icmltitle{Structure Discovery in Nonparametric Regression through Compositional Kernel Search - Automatic Decompositions}

\icmlauthor{David Duvenaud$^{\dagger}$}{dkd23@cam.ac.uk}
%\icmladdress{University of Cambridge}
\icmlauthor{James Robert Lloyd$^{\dagger}$}{jrl44@cam.ac.uk}
%\icmladdress{University of Cambridge}
\icmlauthor{Roger Grosse}{rgrosse@mit.edu}
%\icmladdress{Massachussets Institute of Technology}
\icmlauthor{Joshua B. Tenenbaum}{jbt@mit.edu}
%\icmladdress{Massachussets Institute of Technology}
\icmlauthor{Zoubin Ghahramani}{zoubin@eng.cam.ac.uk}
%\icmladdress{University of Cambridge}
%\icmladdress{Brain and Cognitive Sciences, Massachusetts Institute of Technology}    
            
\icmlkeywords{nonparametrics, gaussian process, machine learning, ICML, structure learning, extrapolation, regression, kernel learning, equation learning, supervised learning, time series}
\\vskip 0.3in
]
'''

    latex_footer = '''

\end{document}    
'''

    latex_body = '''
\section{%(folder)s}

\input{figures/%(folder)s/decomp.tex}    
'''

    decomp_header = '''
\\begin{figure}[H]
\\newcommand{\wmgd}{1\columnwidth}
\\newcommand{\hmgd}{3.0cm}
\\newcommand{\mdrd}{figures/%(folder)s}
\\newcommand{\mbm}{\hspace{-0.3cm}}
\\begin{tabular}{c}
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(folder)s_all} \\\\ = \\\\
'''

    decomp_footer = '''
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(folder)s_resid}
\end{tabular}
\end{figure}
'''

    decomp_body = '''
\mbm \includegraphics[width=\wmgd,height=\hmgd]{\mdrd/%(folder)s_%(number)d} \\\\ + \\\\
'''
    
    latex = latex_header
    for folder in [adir for adir in sorted(os.listdir(top_folder)) if os.path.isdir(os.path.join(top_folder, adir))]:
        decomp_text = decomp_header % {'folder' : folder}
        i = 1
        while os.path.isfile(os.path.join(top_folder, folder, '%s_%d.pdf' % (folder, i))):
            decomp_text = decomp_text + decomp_body % {'folder' : folder, 'number' : i}
            i += 1
        decomp_text = decomp_text + decomp_footer % {'folder' : folder}
        
        with open(os.path.join(top_folder, folder, 'decomp.tex'), 'w') as decomp_file:
            decomp_file.write(decomp_text)
        
        latex = latex + latex_body % {'folder' : folder}
    latex = latex + latex_footer
    
    with open(tex, 'w') as latex_file:
        latex_file.write(latex)
    
                
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
