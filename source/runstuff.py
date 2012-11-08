'''
Created on 8 Nov 2012

@author: dkd23
'''

import os

  
def call_cluster(k=3, max_depth=2):
    '''Run experiments on all data sets on the cluster'''
    datasets = ['data/mauna2003.mat', \
                'data/abalone_500.mat', \
                'data/bach_synth_r_200.mat', \
                'data/r_concrete_500.mat', \
                'data/housing.mat' \
               ]
    
    #fear_string = '''. /usr/local/grid/divf2/common/settings.sh; qsub -l lr=0 -o "/home/mlg/dkd23/large_results/fear_logs_gpss/gpss_run_log_%(dataset)s.txt" -e "/home/mlg/dkd23/large_results/fear_logs_gpss/gpss_error_log_%(dataset)s.txt"  /home/mlg/dkd23/git/gp-structure-search/source/run_python.sh /home/mlg/dkd23/git/gp-structure-search/%(dataset)s /home/mlg/dkd23/git/gp-structure-search/%(results_filename)s %(depth)s %(k)s'''
    fear_string = '''. /usr/local/grid/divf2/common/settings.sh; qsub -l lr=0 /home/mlg/dkd23/git/gpss_stable/source/run_python.sh /home/mlg/dkd23/git/gpss_stable/%(dataset)s /home/mlg/dkd23/git/gpss_stable/%(results_filename)s %(depth)s %(k)s'''
    
    for data_file in datasets:
        results_filename = 'experiment_output_%s_depth=%d_k=%d' % ( data_file, max_depth, k )
        command_str = fear_string % {'dataset' : data_file, 'results_filename' : results_filename, 'depth' : max_depth, 'k' : k }
        os.system(command_str)
        
if __name__ == '__main__':
    call_cluster()       
