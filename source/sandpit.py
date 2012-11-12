'''
Created on Nov 2012

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          David Duvenaud (dkd23@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
'''

import flexiblekernel as fk
import grammar
import gpml
import structure_search

import numpy as np
import pylab
import scipy.io
import sys
import os

def kernel_test():
    k = fk.MaskKernel(4, 3, fk.SqExpKernel(0, 0))
    print k.gpml_kernel_expression()
    print k.pretty_print()
    print '[%s]' % k.param_vector()
    print 'kernel_test complete'
    
def base_kernel_test():
    print [k.pretty_print() for k in fk.base_kernels(1)]
    print 'base_kernel_test complete'
    
def expand_test():  
    k1 = fk.SqExpKernel(1, 1)
    k2 = fk.SqExpPeriodicKernel(2, 2, 2)
    e = fk.SumKernel([k1, k2])
    
    g = grammar.OneDGrammar()
    
    print ''
    for f in grammar.expand(e, g):
        #print f
        print f.pretty_print()
        print grammar.canonical(f).pretty_print()
        print
        
    print '   ***** duplicates removed  *****'
    print
    
    kernels = grammar.expand(e, g)
    for f in grammar.remove_duplicates(kernels):
        print f.pretty_print()
        print
        
    print '%d originally, %d without duplicates' % (len(kernels), len(grammar.remove_duplicates(kernels)))
        
    print 'expand_test complete'
    
def expand_test2():    
    k1 = fk.MaskKernel(2, 0, fk.SqExpKernel(1, 1))
    k2 = fk.MaskKernel(2, 1, fk.SqExpPeriodicKernel(2, 2, 2))
    e = fk.SumKernel([k1, k2])
    
    g = grammar.MultiDGrammar(2)
    
    print ''
    for f in grammar.expand(e, g):
        print f.pretty_print()
        print grammar.canonical(f).pretty_print()
        print
        
    print '   ***** duplicates removed  *****'
    print
    
    kernels = grammar.expand(e, g)
    for f in grammar.remove_duplicates(kernels):
        print f.pretty_print()
        print
        
    print '%d originally, %d without duplicates' % (len(kernels), len(grammar.remove_duplicates(kernels)))
        
    print 'expand_test complete'
    

def load_mauna():
    '''2011 Mauna dataset.'''
    
    data_file = '../data/mauna.mat'
    data = scipy.io.loadmat(data_file)
    return data['X'], data['y']

def load_mauna_original():
    """
    Original Mauna dataset made to match the experiments from Carl's book.
    For details, see data/preprocess_mauna_2004.m
    """
    
    data_file = '../data/mauna2003.mat'
    data = scipy.io.loadmat(data_file)
    return data['X'], data['y']


def call_gpml_test():
    
    np.random.seed(0)
    
    k = fk.SumKernel([fk.SqExpKernel(0, 0), fk.SqExpKernel(0, 0)])
    print k.gpml_kernel_expression()
    print k.pretty_print()
    print '[%s]' % k.param_vector()

    X, y = load_mauna()
    
    N_orig = X.shape[0]
    X = X[:N_orig//3, :]
    y = y[:N_orig//3, :]

    results = []

    pylab.figure()
    for i in range(15):
        init_params = np.random.normal(size=k.param_vector().size)
        #kernel_hypers, nll, nlls = gpml.optimize_params(k.gpml_kernel_expression(), k.param_vector(), X, y, return_all=True)
        kernel_hypers, nll, nlls = gpml.optimize_params(k.gpml_kernel_expression(), init_params, X, y, return_all=True)
    
        print "kernel_hypers =", kernel_hypers
        print "nll =", nll
        
        k_opt = k.family().from_param_vector(kernel_hypers)
        print k_opt.gpml_kernel_expression()
        print k_opt.pretty_print()
        print '[%s]' % k_opt.param_vector()
        
        pylab.semilogx(range(1, nlls.size+1), nlls)
        
        results.append((kernel_hypers, nll))
        
        pylab.draw()
    
    print
    print
    results = sorted(results, key=lambda p: p[1])
    for kernel_hypers, nll in results:
        print nll, kernel_hypers
        
    print "done"
        



def sample_mauna_best():
    # This kernel was chosen from a run of Mauna datapoints.
    kernel = ( fk.SqExpKernel(-0.7, -1.3) + fk.SqExpKernel(4.8, 2.3) ) * \
             ( fk.SqExpKernel(3.0, 0.5) + fk.SqExpPeriodicKernel(0.4, -0.0, -0.9) ) 
        
    X = np.linspace(0,50,500)
    
    # Todo: set random seed.
    sample = gpml.sample_from_gp_prior(kernel, X)
    
    pylab.figure()
    pylab.plot(X, sample)
    pylab.title('( SqExp(ell=-0.7, sf=-1.3) + SqExp(ell=4.8, sf=2.3) ) \n x ( SqExp(ell=3.0, sf=0.5) + Periodic(ell=0.4, p=-0.0, sf=-0.9) )')
    
    

    
    
def sample_Carls_kernel():
    kernel = fk.Carls_Mauna_kernel()
        
    X = np.linspace(0,50,500)
    
    # Todo: set random seed.
    sample = gpml.sample_from_gp_prior(kernel, X)
    
    pylab.figure()
    pylab.plot(X, sample)    
    pylab.title('Carl''s kernel');


def compare_kernels_experiment():
    kernel1 = fk.Carls_Mauna_kernel()
    kernel2  = ( fk.SqExpKernel(-0.7, -1.3) + fk.SqExpKernel(4.8, 2.3) ) * \
               ( fk.SqExpKernel(3.0, 0.5) + fk.SqExpPeriodicKernel(0.4, -0.0, -0.9) )
    #kernel2 = ( SqExp(ell=-0.8, sf=-1.4) + Periodic(ell=0.5, p=-0.3, sf=-1.1) + RQ(ell=1.9, sf=1.6, a=0.2) + ( SqExp(ell=4.5, sf=1.0) x Periodic(ell=0.6, p=-0.0, sf=0.1) )  )  
             
    X, y = load_mauna_original()
    N_orig = X.shape[0]  # subsample data.
    X = X[:N_orig//5, :]
    y = y[:N_orig//5, :]     
     
    print "Carl's kernel"
    print kernel1.pretty_print()
    kernel_hypers1, nll1 = gpml.optimize_params(kernel1.gpml_kernel_expression(), kernel1.param_vector(), \
                                                X, y, noise=np.log(0.19), iters=100 )
    k1_opt = kernel1.family().from_param_vector(kernel_hypers1)
    print k1_opt.pretty_print()   
    print "Carl's NLL =", nll1 
    
    print "Our kernel"
    print kernel2.pretty_print()
    kernel_hypers2, nll2 = gpml.optimize_params(kernel2.gpml_kernel_expression(), kernel2.param_vector(), \
                                                X, y, noise=np.log(0.19), iters=100)
    k2_opt = kernel2.family().from_param_vector(kernel_hypers2)
    print k2_opt.pretty_print()            

    print "Our NLL =", nll2
    
         

def simple_mauna_experiment():
    '''A first version of an experiment learning kernels'''
    
    seed_kernels = [fk.SqExpKernel(0, 0)]
    
    X, y = load_mauna_original()
    N_orig = X.shape[0]  # subsample data.
    X = X[:N_orig//3, :]
    y = y[:N_orig//3, :] 
    
    max_depth = 4
    k = 4    # Expand k best
    nll_key = 1
    laplace_key = 2
    
    results = []
    for dummy in range(max_depth):     
        new_results = structure_search.try_expanded_kernels(X, y, D=2, seed_kernels=seed_kernels, verbose=False)
        results = results + new_results
        
        print
        results = sorted(results, key=lambda p: p[nll_key], reverse=True)
        for kernel, nll, laplace in results:
            print nll, laplace, kernel.pretty_print()
            
        seed_kernels = [r[0] for r in sorted(new_results, key=lambda p: p[nll_key])[0:k]]

    
def plot_Carls_kernel():
    kernel = fk.Carls_Mauna_kernel()
        
    X = np.linspace(0,10,1000)
    sigma = gpml.plot_kernel(kernel, X)
    
    pylab.figure()
    pylab.plot(X, sigma)    
    pylab.title('Carl''s kernel');
    
    
def plot_our_kernel():
    kernel  = ( fk.SqExpKernel(-0.7, -1.3) + fk.SqExpKernel(4.8, 2.3) ) * \
              ( fk.SqExpKernel(3.0, 0.5) + fk.SqExpPeriodicKernel(0.4, -0.0, -0.9) ) 
        
    X = np.linspace(0,10,1000)
    sigma = gpml.plot_kernel(kernel, X)
    
    pylab.figure()
    pylab.plot(X, sigma)    
    pylab.title('Our kernel');  
    
def load_simple_gef_load():
    '''Zone 1 and temperature station 2'''
    
    data_file = '../data/gef_load_simple.mat'
    data = scipy.io.loadmat(data_file)
    return data['X'], data['y'] 
    
def load_full_gef_load():
    '''20 Zones in y, time and 11 temp stations in X'''
    
    data_file = '../data/gef_load_full_Xy.mat'
    data = scipy.io.loadmat(data_file)
    return data['X'], data['y']  
    
def simple_gef_load_experiment(verbose=True):
    '''A first version of an experiment learning kernels'''
    
    seed_kernels = [fk.MaskKernel(2, 0, fk.SqExpKernel(0, 0)),
                    fk.MaskKernel(2, 1, fk.SqExpKernel(0, 0))]
    
    X, y = load_simple_gef_load()
    # subsample data.
    X = X[0:99, :]
    y = y[0:99, :] 
    
    max_depth = 5
    k = 2    # Expand k best
    nll_key = 1
    BIC_key = 2
    active_key = BIC_key
    
    
    results = []
    for dummy in range(max_depth):     
        new_results = structure_search.try_expanded_kernels(X, y, D=2, seed_kernels=seed_kernels, verbose=verbose)
        results = results + new_results
        
        print
        results = sorted(results, key=lambda p: p[active_key], reverse=True)
        for kernel, nll, BIC in results:
            print nll, BIC, kernel.pretty_print()
            
        seed_kernels = [r[0] for r in sorted(new_results, key=lambda p: p[active_key])[0:k]] 
    
def full_gef_load_experiment(zone=1, max_depth=5, verbose=True):
    '''Round 2'''
    
#    seed_kernels = [fk.MaskKernel(2, 0, fk.SqExpKernel(0, 0)),
#                    fk.MaskKernel(2, 1, fk.SqExpKernel(0, 0))]

    seed_kernels = [fk.MaskKernel(12, i, fk.SqExpKernel(0., 0.))  for i in range(12)] + \
                   [fk.MaskKernel(12, i, fk.SqExpPeriodicKernel(0., 0., 0.))  for i in range(12)] + \
                   [fk.MaskKernel(12, i, fk.RQKernel(0., 0., 0.))  for i in range(12)]
    
    X, y = load_full_gef_load()
    # subsample data.
    X = X[0:299, :]
    y = y[0:299, zone-1] 
    
#    max_depth = 5
    k = 2    # Expand k best
    nll_key = 1
    BIC_key = 2
    active_key = BIC_key
    
    
    results = []
    for i in range(max_depth):     
        if i:
            expand = True
        else:
            expand = False
        new_results = structure_search.try_expanded_kernels(X, y, D=12, seed_kernels=seed_kernels, expand=expand, verbose=verbose)
        results = results + new_results
        
        print
        results = sorted(results, key=lambda p: p[active_key], reverse=True)
        for kernel, nll, BIC in results:
            print nll, BIC, kernel.pretty_print()
            
        seed_kernels = [r[0] for r in sorted(new_results, key=lambda p: p[active_key])[0:k]]
        

        #os.system(command_str)
        
#### Attempt at sending individual jobs to the cluster
        
import pysftp, tempfile, config, subprocess, config, time
nax = np.newaxis

def mkstemp_safe(directory, suffix):
    (os_file_handle, file_name) = tempfile.mkstemp(dir=directory, suffix=suffix)
    os.close(os_file_handle)
    return file_name

def fear_connect():
    return pysftp.Connection('fear', username=config.USERNAME, password=config.PASSWORD)

def fear_command(cmd, fear=None):
    if not fear is None:
        srv = fear
    else:
        srv = fear_connect()
    output =  srv.execute(cmd)
    if fear is None:
        srv.close()
    return output
    
def copy_to_fear(local_path, remote_path, fear=None):
    if not fear is None:
        srv = fear
    else:
        srv = fear_connect()
    srv.put(local_path, remote_path)
    if fear is None:
        srv.close()
    
def copy_from_fear(remote_path, local_path, fear=None):
    if not fear is None:
        srv = fear
    else:
        srv = fear_connect()
    srv.get(remote_path, local_path)
    if fear is None:
        srv.close()
    
def fear_rm(remote_path, fear=None):
    if not fear is None:
        srv = fear
    else:
        srv = fear_connect()
    output =  srv.execute('rm %s' % remote_path)
    if fear is None:
        srv.close()
    return output

def fear_file_exists(remote_path, fear=None):
    if not fear is None:
        srv = fear
    else:
        srv = fear_connect()
    response = srv.execute('if [ -e %s ] \nthen \necho ''exists'' \nfi' % remote_path)
    if fear is None:
        srv.close()
    return response == ['exists\n']

def fear_qdel_all(fear=None):
    if not fear is None:
        srv = fear
    else:
        srv = fear_connect()
    output = srv.execute('. /usr/local/grid/divf2/common/settings.sh; qdel -u %s' % config.USERNAME)
    if fear is None:
        srv.close()
    return output

def qsub_matlab_code(code, verbose=True, local_dir ='../temp/', remote_dir ='./temp/', fear=None):
    # Write to a temp script
    script_file = mkstemp_safe(local_dir, '.m')
    shell_file = mkstemp_safe(local_dir, '.sh')
    
    f = open(script_file, 'w')
    f.write(code)
    f.close()
    
    #### Local file reference without extension - MATLAB fails silently otherwise
    f = open(shell_file, 'w')
    f.write('/usr/local/apps/matlab/matlabR2011b/bin/matlab -nosplash -nojvm -nodisplay -singleCompThread -r ' + script_file.split('/')[-1].split('.')[0] + '\n')
    f.close()
    
    # Copy this to fear
    
    copy_to_fear(script_file, remote_dir + script_file.split('/')[-1], fear)
    copy_to_fear(shell_file, remote_dir + shell_file.split('/')[-1], fear)
    
    # Create fear call
    
    #### WARNING - hardcoded path 'temp'

    fear_string = ' '.join(['. /usr/local/grid/divf2/common/settings.sh;',
                            'cd temp;'
                            'chmod +x %s;' % shell_file.split('/')[-1],
                            'qsub -l lr=0',
                            shell_file.split('/')[-1] + ';',
                            'cd ..'])

    if verbose:
        print 'Submitting : %s' % fear_string
    
    # Send this command to fear
    
    fear_command(fear_string, fear)
    
    # Tell the caller where the script file was written
    return script_file, shell_file

def re_qsub(shell_file, verbose=True, fear=None):

    # Create fear call
    
    #### WARNING - hardcoded path 'temp'

    fear_string = ' '.join(['. /usr/local/grid/divf2/common/settings.sh;',
                            'cd temp;'
                            'chmod +x %s;' % shell_file.split('/')[-1],
                            'qsub -l lr=0',
                            shell_file.split('/')[-1] + ';',
                            'cd ..'])

    if verbose:
        print 'Re-submitting : %s' % fear_string
    
    # Send this command to fear
    
    fear_command(fear_string, fear)
    

# Matlab code to optimise hyper-parameters on one file, given one kernel.
OPTIMIZE_KERNEL_CODE = r"""
%% Load the data, it should contain X and y.
a = 'trying to load data files'
load '%(datafile)s'
a = 'loaded data files'

%% Load GPML
addpath(genpath('%(gpml_path)s'));
a = 'loaded GPML'

%% Set up model.
meanfunc = {@meanConst}
hyp.mean = mean(y)

covfunc = %(kernel_family)s
hyp.cov = %(kernel_params)s

likfunc = @likGauss
hyp.lik = %(noise)s

[hyp_opt, nlls] = minimize(hyp, @gp, -%(iters)s, @infExact, meanfunc, covfunc, likfunc, X, y);
best_nll = nlls(end)

laplace_nle = best_nll %% HACK HACK HACK

save( '%(writefile)s', 'hyp_opt', 'best_nll', 'nlls', 'laplace_nle' );
a = 'Goodbye, World!'
exit();
"""

def fear_run_experiments(kernels, X, y, return_all=False, verbose=True, noise=None, iters=300, local_dir ='../temp/', remote_dir ='./temp/', \
                         sleep_time=10, n_sleep_timeout=6, re_submit_wait=60):
    '''
    Sends jobs to fear, waits for them, returns the results
    '''
    # Not sure what this is for
    
    if X.ndim == 1:
        X = X[:, nax]
    if y.ndim == 1:
        y = y[:, nax]
        
    if noise is None:
        noise = np.log(np.var(y)/10)   #### Just a heuristic.
        
    data = {'X': X, 'y': y}
    
    # Setup the connection to fear
    
    fear = fear_connect()
    
    # Submit all the jobs and remember where we put them
    
    data_files = []
    write_files = []
    script_files = []
    shell_files = []
    
    for kernel in kernels:
        
        # Create data file and results file
    
        data_files.append(mkstemp_safe(local_dir, '.mat'))
        write_files.append(mkstemp_safe(local_dir, '.mat'))
        
        # Save data
        
        scipy.io.savemat(data_files[-1], data)
        
        # Copy files to fear
   
        copy_to_fear(data_files[-1], remote_dir + data_files[-1].split('/')[-1], fear)
#        copy_to_fear(write_files[-1], remote_dir + write_files[-1].split('/')[-1])
        
        # Create MATLAB code
    
        code = OPTIMIZE_KERNEL_CODE % {'datafile': data_files[-1].split('/')[-1],
                                       'writefile': write_files[-1].split('/')[-1],
                                       'gpml_path': config.FEAR_GPML_PATH,
                                       'kernel_family': kernel.gpml_kernel_expression(),
                                       'kernel_params': '[ %s ]' % ' '.join(str(p) for p in kernel.param_vector()),
                                       'noise': str(noise),
                                       'iters': str(iters)}
        
        # Submit this to fear and save the file names
        
        script_file, shell_file = qsub_matlab_code(code=code, verbose=verbose, local_dir=local_dir, remote_dir=remote_dir, fear=fear)
        script_files.append(script_file)
        shell_files.append(shell_file)
        
    # Let the scripts run
    
#    if verbose:
#        print 'Giving the jobs some time to run'
#    time.sleep(re_submit_wait)
        
    # Wait for and read in results
    
    fear_finished = False
    job_finished = [False] * len(write_files)
    results = [None] * len(write_files)
    sleep_count = 0
    
    while not fear_finished:
        for (i, write_file) in enumerate(write_files):
            if not job_finished[i]:
                if fear_file_exists(remote_dir + write_file.split('/')[-1], fear):
                    # Another job has finished
                    job_finished[i] = True
                    sleep_count = 0
                    # Copy files
                    os.remove(write_file)
                    copy_from_fear(remote_dir + write_file.split('/')[-1], write_file, fear)
                    # Read results
                    gpml_result = scipy.io.loadmat(write_file)
                    optimized_hypers = gpml_result['hyp_opt']
                    nll = gpml_result['best_nll'][0, 0]
#                    nlls = gpml_result['nlls'].ravel()
                    laplace_nle = gpml_result['laplace_nle'][0, 0]
                    kernel_hypers = optimized_hypers['cov'][0, 0].ravel()
                    k_opt = kernels[i].family().from_param_vector(kernel_hypers)
                    BIC = 2 * nll + len(kernel_hypers) * np.log(y.shape[0])
                    results[i] = (k_opt, nll, laplace_nle, BIC)
                    # Tidy up
                    fear_rm(remote_dir + data_files[i].split('/')[-1], fear)
                    fear_rm(remote_dir + write_files[i].split('/')[-1], fear)
                    fear_rm(remote_dir + script_files[i].split('/')[-1], fear)
                    fear_rm(remote_dir + shell_files[i].split('/')[-1], fear)
                    fear_rm(remote_dir + shell_files[i].split('/')[-1] + '*', fear)
                    os.remove(data_files[i])
                    os.remove(write_files[i])
                    os.remove(script_files[i])
                    os.remove(shell_files[i])
                    # Tell the world
                    if verbose:
                        print '%d / %d jobs complete' % (sum(job_finished), len(job_finished))
        
        if sum(job_finished) == len(job_finished):
            fear_finished = True    
        if not fear_finished:
            if verbose:
                print 'Sleeping'
            sleep_count += 1
            if sleep_count < n_sleep_timeout:
                time.sleep(sleep_time)
            else:
                # Jobs taking too long - assume failure - resubmit
                fear_qdel_all(fear)
                for (i, shell_file) in enumerate(shell_files):
                    if not job_finished[i]:
                        re_qsub(shell_file, verbose=verbose, fear=fear)
                if verbose:
                    print 'Giving the jobs some time to run'
                time.sleep(re_submit_wait)
                sleep_count = 0
            
    fear.close()
    
    return results

def fear_load_mat(data_file, y_dim=1):
    '''Load a Matlab file'''
    data = scipy.io.loadmat(data_file)
    return data['X'], data['y'][:,y_dim-1], np.shape(data['X'])[1]

def fear_expand_kernels(D, seed_kernels, verbose=False):    
    '''
    Just expands
    '''
       
    g = grammar.MultiDGrammar(D)
    print 'Seed kernels :'
    for k in seed_kernels:
        print k.pretty_print()
    kernels = []
    for k in seed_kernels:
        kernels = kernels + grammar.expand(k, g)
    kernels = grammar.remove_duplicates(kernels)
    print 'Expanded kernels :'
    for k in kernels:
        print k.pretty_print()
            
    return (kernels)


def fear_experiment(data_file, results_filename, y_dim=1, subset=None, max_depth=2, k=2, verbose=True, sleep_time=60, n_sleep_timeout=20, re_submit_wait=60, \
                    description=''):
    '''Recursively search for the best kernel'''

    X, y, D = fear_load_mat(data_file, y_dim)
    
    # Subset if necessary
    if not subset is None:
        X = X[subset, :]
        y = y[subset] 
    
    ##### This should be abstracted
    seed_kernels = [fk.MaskKernel(D, i, fk.SqExpKernel(0., 0.))  for i in range(D)] + \
                   [fk.MaskKernel(D, i, fk.SqExpPeriodicKernel(0., 0., 0.))  for i in range(D)] + \
                   [fk.MaskKernel(D, i, fk.RQKernel(0., 0., 0.))  for i in range(D)]
    
    nll_key = 1
    laplace_key = 2
    BIC_key = 3
    active_key = BIC_key
        
    results = []
    results_sequence = []
    for r in range(max_depth):   
        if r == 0:  
            new_results = fear_run_experiments(seed_kernels, X, y, verbose=verbose, \
                                               sleep_time=sleep_time, n_sleep_timeout=n_sleep_timeout, re_submit_wait=re_submit_wait)
        else:
            new_results = fear_run_experiments(fear_expand_kernels(D, seed_kernels, verbose=verbose), X, y, verbose=verbose, \
                                               sleep_time=sleep_time, n_sleep_timeout=n_sleep_timeout, re_submit_wait=re_submit_wait)
            
        results = results + new_results
        
        print
        results = sorted(results, key=lambda p: p[active_key], reverse=True)
        for kernel, nll, laplace, BIC in results:
            print nll, laplace, BIC, kernel.pretty_print()
            
        seed_kernels = [r[0] for r in sorted(new_results, key=lambda p: p[active_key])[0:k]]
        
        results_sequence.append(results)

    # Write results to a file
    results = sorted(results, key=lambda p: p[active_key], reverse=True)
    with open(results_filename, 'w') as outfile:
        outfile.write('Experiment results for\n datafile = %s\n y_dim = %d\n subset = %s\n max_depth = %f\n k = %f\n Description = %s\n\n' % (data_file, y_dim, subset, max_depth, k, description)) 
        for (i, results) in enumerate(results_sequence):
            outfile.write('\n%%%%%%%%%% Level %d %%%%%%%%%%\n\n' % i)
            for kernel, nll, laplace, BIC in results:
                outfile.write( 'nll=%f, laplace=%f, BIC=%f, kernel=%s\n' % (nll, laplace, BIC, kernel.__repr__()))
                
def plot_gef_load_Z01():
    # This kernel was chosen from a run of gef_load datapoints.
#    kernel = eval(ProductKernel([ covMask(ndim=12, active_dimension=0, base_kernel=RQKernel(lengthscale=0.268353, output_variance=-0.104149, alpha=-2.105742)), covMask(ndim=12, active_dimension=9, base_kernel=SqExpKernel(lengthscale=1.160242, output_variance=0.004344)), SumKernel([ covMask(ndim=12, active_dimension=0, base_kernel=SqExpPeriodicKernel(lengthscale=-0.823413, period=0.000198, output_variance=-0.917064)), covMask(ndim=12, active_dimension=0, base_kernel=RQKernel(lengthscale=-0.459219, output_variance=-0.077250, alpha=-2.212718)) ]) ]))

    X, y, D = fear_load_mat('../data/gef_load_full_Xy.mat', 1)
    
    kernel = fk.MaskKernel(D, 0, fk.RQKernel(0.268353, -0.104149, -2.105742)) * fk.MaskKernel(D, 9, fk.SqExpKernel(1.160242, 0.004344)) * \
             (fk.MaskKernel(D, 0, fk.SqExpPeriodicKernel(-0.823413, 0.000198, -0.917064)) + fk.MaskKernel(D, 0, fk.RQKernel(-0.459219, -0.077250, -2.212718)))
    
    # Todo: set random seed.
    sample = gpml.sample_from_gp_prior(kernel, X[0:499,:])
    
    pylab.figure()
    pylab.plot(X[0:499,0], y[0:499])
    pylab.title('GEF load Z01 - first 500 data points')
    pylab.xlabel('Time')
    pylab.ylabel('Load')
    
    pylab.figure()
    pylab.plot(X[0:499,0], sample)
    pylab.title('GEF load Z01 - a sample from the learnt kernel')
    pylab.xlabel('Time')
    pylab.ylabel('Load')
    
    kernel_1 = fk.MaskKernel(D, 0, fk.RQKernel(0.268353, -0.104149, -2.105742)) * fk.MaskKernel(D, 9, fk.SqExpKernel(1.160242, 0.004344)) * \
               fk.MaskKernel(D, 0, fk.SqExpPeriodicKernel(-0.823413, 0.000198, -0.917064))
               
    posterior_mean_1 = gpml.posterior_mean(kernel, kernel_1, X[0:499,:], y[0:499])
    
    pylab.figure()
    pylab.plot(X[0:499,0], posterior_mean_1)
    pylab.title('GEF load Z01 - periodic posterior mean component')
    pylab.xlabel('Time')
    pylab.ylabel('Load')
    
    kernel_2 = fk.MaskKernel(D, 0, fk.RQKernel(0.268353, -0.104149, -2.105742)) * fk.MaskKernel(D, 9, fk.SqExpKernel(1.160242, 0.004344)) * \
               fk.MaskKernel(D, 0, fk.RQKernel(-0.459219, -0.077250, -2.212718))
               
    posterior_mean_2 = gpml.posterior_mean(kernel, kernel_2, X[0:499,:], y[0:499])
    
    pylab.figure()
    pylab.plot(X[0:499,0], posterior_mean_2)
    pylab.title('GEF load Z01 - smooth posterior mean component')
    pylab.xlabel('Time')
    pylab.ylabel('Load')

def main():
    # Run everything
#    fear_experiment('../data/abalone_500.mat', '../results/abalone_500_01.txt', max_depth=4, k=3)
    fear_experiment('../data/gef_load_full_Xy.mat', '../results/gef_load_500_Z01_02.txt', max_depth=6, k=5, subset=range(500), y_dim=1, description = 'BIC, 0 init')
    fear_experiment('../data/gef_load_full_Xy.mat', '../results/gef_load_500_Z09_02.txt', max_depth=6, k=5, subset=range(500), y_dim=9, description = 'BIC, 0 init')
    fear_experiment('../data/bach_synth_r_200.mat', '../results/bach_synth_r_200_02.txt', max_depth=6, k=5, description = 'BIC, 0 init')
    fear_experiment('../data/housing.mat', '../results/housing_02.txt', max_depth=6, k=5, description = 'BIC, 0 init')
    fear_experiment('../data/mauna2003.mat', '../results/mauna2003_02.txt', max_depth=6, k=5, description = 'BIC, 0 init')
    fear_experiment('../data/mauna2011.mat', '../results/mauna2011_02.txt', max_depth=6, k=5, description = 'BIC, 0 init')
    fear_experiment('../data/prostate.mat', '../results/prostate_02.txt', max_depth=6, k=5, description = 'BIC, 0 init')
    fear_experiment('../data/pumadyn256.mat', '../results/pumadyn256_02.txt', max_depth=6, k=5, description = 'BIC, 0 init')
    fear_experiment('../data/r_concrete_100.mat', '../results/r_concrete_100_02.txt', max_depth=6, k=5, description = 'BIC, 0 init')
    fear_experiment('../data/r_concrete_500.mat', '../results/r_concrete_500_02.txt', max_depth=6, k=5, description = 'BIC, 0 init')
    fear_experiment('../data/r_solar_500.mat', '../results/r_solar_500_02.txt', max_depth=6, k=5, description = 'BIC, 0 init')
    fear_experiment('../data/unicycle_pitch_angle_400.mat', '../results/unicycle_pitch_angle_400_02.txt', max_depth=6, k=5, description = 'BIC, 0 init')
    fear_experiment('../data/unicycle_pitch_ang_vel_400.mat', '../results/unicycle_pitch_ang_vel_400_02.txt', max_depth=6, k=5, description = 'BIC, 0 init')

if __name__ == '__main__':
    #kernel_test()
    #expression_test()
    #base_kernel_test()
    #expand_test()
    #call_gpml_test()
    #sample_from_gp_prior()
    #if sys.flags.debug or __debug__:
    #    print 'Debug mode'
    #call_cluster()
    #pass
    main()

