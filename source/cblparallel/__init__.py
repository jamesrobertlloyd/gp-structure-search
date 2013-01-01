"""
A module to make it much easier to send code to the CBL computing cluster.

Contributions encouraged

@authors:
James Robert Lloyd (jrl44@cam.ac.uk)

"""

import pyfear # Is this line necessary?
from pyfear import fear
from util import mkstemp_safe
import os
import psutil, subprocess, sys, time

from config import *

#### WISHLIST
####  - Write setup function / make it possible for people to use this code without lots of hassle
####  - Provide convenience functions to setup MATLAB/python paths
####  - Merge job handling code
####  - Return STDOUT and STDERR from scripts
####  - Allow job handling across gate.eng.cam.ac.uk 
####  - Write unit tests
####  - Write asynchronous job handling
####  - Add support for jobs in arbitrary languages (will require more user input)

def setup():
    '''
    Run an interactive script to setup various preliminaries e.g.
     - RSA key pairs
     - Fear .profile including script that makes qsub, qstat etc available
     - Local directory on fear with python scripts
     - Local directory on machine where temporary files are stored
     - Local directory on fear where temporary files are stored
    '''
    pass
    
#### TODO - Combine these functions - they share much code
####      - Maybe this could be achieved by creating a generic object like fear that (re)moves files etc.
####      - but either does this on fear, on local machine, or on fear via gate.eng.cam.ac.uk

def run_batch_on_fear(scripts, language='python', job_check_sleep=30, file_copy_timeout=120, max_jobs=50, verbose=True):
    '''
    Receives a list of python scripts to run

    Assumes the code has an output file - i.e. %(output_file)s - that will be managed by this function
    
    Returns a list of local file names where the code has presumably stored output
    '''
    # Define some code constants
    #### TODO - this path adding code should accept an optional list of paths
    python_path_code = '''
import sys
sys.path.append('%s')
''' % REMOTE_PYTHON_PATH

    #### This will be deprecated in future MATLAB - hopefully the -singleCompThread command is sufficient
    #### TODO - No longer used? Remove?
    matlab_single_thread = '''
maxNumCompThreads(1);
'''

    matlab_path_code = '''
addpath(genpath('%s'))
''' % REMOTE_MATLAB_PATH
    
    #### TODO - allow port forwarding / tunneling - copy to machine on network with more disk space than fear, then copy from that machine?
    python_transfer_code = '''
from util import timeoutCommand
print "Moving output file"
if not timeoutCommand(cmd='scp -i %(rsa_key)s %(output_file)s %(username)s@%(local_host)s:%(local_temp_path)s; rm %(output_file)s').run(timeout=%(timeout)d)[0]:
    raise RuntimeError('Copying output raised error or timed out')
''' % {'rsa_key' : REMOTE_TO_LOCAL_KEY_FILE,
       'output_file' : '%(output_file)s',
       'username' : USERNAME,
       'local_host' : LOCAL_HOST,
       'local_temp_path' : LOCAL_TEMP_PATH,
       'timeout' : file_copy_timeout}
    
    #### TODO - does this suffer from the instabilities that lead to the verbosity of the python command above
    matlab_transfer_code = '''
system('scp -i %(rsa_key)s %(output_file)s %(username)s@%(local_host)s:%(local_temp_path)s; rm %(output_file)s')
''' % {'rsa_key' : REMOTE_TO_LOCAL_KEY_FILE,
       'output_file' : '%(output_file)s',
       'username' : USERNAME,
       'local_host' : LOCAL_HOST,
       'local_temp_path' : LOCAL_TEMP_PATH}
       
    python_completion_code = '''
print 'Writing completion flag'
with open('%(flag_file)s', 'w') as f:
    f.write('Goodbye, World')
print "I'll bite your legs off!"
quit()
'''
  
    #### TODO - Is this completely stable       
    matlab_completion_code = '''
fprintf('\\nWriting completion flag\\n');
ID = fopen('%(flag_file)s', 'w');
fprintf(ID, 'Goodbye, world');
fclose(ID);
fprintf('\\nGoodbye, World\\n');
quit()
'''
    
    # Open a connection to fear as a with block - ensures connection is closed
    with pyfear.fear() as fear:
    
        # Initialise lists of file locations job ids
        shell_files = [None] * len(scripts)
        script_files = [None] * len(scripts)
        output_files = [None] * len(scripts)
        stdout_files = [None] * len(scripts)
        flag_files = [None] * len(scripts)
        job_ids = [None] * len(scripts) 
        fear_finished = False
        job_finished = [False] * len(scripts)  
        should_sleep = False 

        # Loop through jobs, submitting jobs whenever fear usage low enough, re-submitting failed jobs
        while not fear_finished:
            for (i, code) in enumerate(scripts):
                if (not job_finished[i]) and (job_ids[i] is None):
                    # This script has not been run - check number of jobs and potentially run
                    if fear.jobs_alive() <= max_jobs:
                        # Jobs can run, continue looping
                        should_sleep = False
                        # Create necessary files in local path (avoids collisions)
                        if language == 'python':
                            script_files.append(mkstemp_safe(LOCAL_TEMP_PATH, '.py'))
                        elif language == 'matlab':
                            script_files.append(mkstemp_safe(LOCAL_TEMP_PATH, '.mat'))
                        shell_files.append(mkstemp_safe(LOCAL_TEMP_PATH, '.sh'))
                        output_files.append(mkstemp_safe(LOCAL_TEMP_PATH, '.out'))
                        flag_files.append(mkstemp_safe(LOCAL_TEMP_PATH, '.flag'))
                        # Customise code (path, transfer of output back to local host, flag file writing)
                        #### TODO - make path and output_transfer optional
                        if language == 'python':
                            code = python_path_code + code + python_transfer_code + python_completion_code
                        elif language == 'matlab':
                            code = matlab_path_code + code + matlab_transfer_code + matlab_completion_code
                        code = code % {'output_file': os.path.join(REMOTE_TEMP_PATH, os.path.split(output_files[i])[-1]),
                                       'flag_file' : os.path.join(REMOTE_TEMP_PATH, os.path.split(flag_files[i])[-1])}
                        # Write code and shell file
                        with open(script_files[i], 'w') as f:
                            f.write(code)
                        with open(shell_files[i], 'w') as f:
                            #### TODO - is os.path.join always correct - what happens if this program is being run on windows?
                            if language == 'python':
                                f.write('python ' + os.path.join(REMOTE_TEMP_PATH, os.path.split(script_files[i])[-1]) + '\n')
                            elif language == 'matlab':
                                f.write('cd ' + REMOTE_TEMP_PATH + ';\n' + REMOTE_MATLAB + ' -nosplash -nojvm -nodisplay -singleCompThread -r ' + \
                                        os.path.split(script_files[i])[-1].split('.')[0] + '\n')
                        # Transfer files to fear
                        fear.copy_to(script_files[i], os.path.join(REMOTE_TEMP_PATH, os.path.split(script_files[i])[-1]))
                        fear.copy_to(shell_files[i], os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]))
                        # Submit the job to fear
                        print 'Submitting job %d of %d' % (i + 1, len(scripts))
                        job_ids[i] = fear.qsub(os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]), verbose=verbose)
                    else:
                        # If this is the last thing to happen in the loop then might as well sleep for a bit to wait for stack to free up
                        should_sleep = True
                elif (not job_finished[i]) and (not job_ids[i] is None):
                    # Has the process terminated?
                    if fear.job_terminated(job_ids[i], update=True):
                        if not fear.file_exists(os.path.join(REMOTE_TEMP_PATH, os.path.split(flag_files[i])[-1])):
                            # Job has finished but no output - resubmit later
                            #### TODO - Record the output and error file - and potentially remove them
                            print 'Shell script %s job_id %s failed' % (os.path.split(shell_files[i])[-1], job_ids[i])
                            job_ids[i] = None
                        else:
                            # Job has finished successfully
                            job_finished[i] = True
                            # Tell the world
                            if verbose:
                                print '%d / %d jobs complete' % (sum(job_finished), len(job_finished))
                        # Tidy up fear
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(script_files[i])[-1]))
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]))
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(flag_files[i])[-1]))
                        #### TODO - record the output and error files for future reference
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]) + '*') # Kills temporary output files
                        # Tidy up local temporary directory
                        os.remove(script_files[i])
                        os.remove(shell_files[i])
                        os.remove(flag_files[i])    
                        job_ids[i] = None                     
                            
                    elif not (fear.job_queued(job_ids[i]) or fear.job_running(job_ids[i]) \
                              or fear.job_loading(job_ids[i])):
                        # Job has some status other than running, queuing or loading - something is wrong, delete it
                        #### TODO - Record the output and error file - and potentially remove them
                        fear.qdel(job_ids[i])
                        print 'Shell script %s job_id %s stuck, deleting' % (os.path.split(shell_files[i])[-1], job_ids[i])
                        #### TODO - remove this code duplication
                        # Tidy up fear
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(script_files[i])[-1]))
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]))
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(flag_files[i])[-1]))
                        #### TODO - record the output and error files for future reference
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]) + '*') # Kills temporary output files
                        # Tidy up local temporary directory
                        os.remove(script_files[i])
                        os.remove(shell_files[i])
                        os.remove(flag_files[i])    
                        job_ids[i] = None    
                    else:
                        should_sleep = True
            if all(job_finished):
                fear_finished = True    
            elif should_sleep:
                if verbose:
                    fear.qstat()
                    print '%d of %d jobs complete' % (sum(job_finished), len(job_finished))
                    print '%d jobs running' % fear.jobs_running(update=False)
                    print '%d jobs loading' % fear.jobs_loading(update=False)
                    print '%d jobs queued' % fear.jobs_queued(update=False)
                    print 'Sleeping for %d seconds' % job_check_sleep
                    time.sleep(job_check_sleep)
                should_sleep = False

    #### TODO - return job output and error files as applicable (e.g. there may be multiple error files associated with one script)
    return output_files
    
def run_batch_locally(scripts, language='python', paths=[], max_cpu=0.9, max_mem=0.9, submit_sleep=1, job_check_sleep=30, \
                      verbose=True, max_files_open=100):
    '''
    Receives a list of python scripts to run

    Assumes the code has an output file that will be managed by this function
    
    Returns a list of local file names where the code has presumably stored output
    '''
    # Define some code constants
    #### Do we need to set paths explicitly?

    #### This will be deprecated in future MATLAB - hopefully the -singleCompThread command is sufficient
    matlab_single_thread = '''
maxNumCompThreads(1);
'''

    python_path_code = '''
import sys
sys.path.append('%s')
'''

    matlab_path_code = '''
addpath(genpath('%s'))
'''
       
    python_completion_code = '''
print 'Writing completion flag'
with open('%(flag_file)s', 'w') as f:
    f.write('Goodbye, World')
print "Goodbye, World"
quit()
'''
  
    #### TODO - Is this completely stable       
    matlab_completion_code = '''
fprintf('\\nWriting completion flag\\n');
ID = fopen('%(flag_file)s', 'w');
fprintf(ID, 'Goodbye, world');
fclose(ID);
fprintf('\\nGoodbye, World\\n');
quit()
'''
    
    # Initialise lists of file locations job ids
    shell_files = [None] * len(scripts)
    script_files = [None] * len(scripts)
    output_files = [None] * len(scripts)
    stdout_files = [None] * len(scripts)
    stdout_file_handles = [None] * len(scripts)
    flag_files = [None] * len(scripts)
    processes = [None] * len(scripts)
    fear_finished = False
    job_finished = [False] * len(scripts)  
    should_sleep = False 
    
    files_open = 0

    # Loop through jobs, submitting jobs whenever CPU usage low enough, re-submitting failed jobs
    while not fear_finished:
        for (i, code) in enumerate(scripts):
            if (not job_finished[i]) and (processes[i] is None) and (files_open < max_files_open):
                # This script has not been run - check CPU and potentially run
                if (psutil.cpu_percent() < max_cpu * 100) and (psutil.virtual_memory().percent < max_mem * 100):
                    # Jobs can run, continue looping
                    should_sleep = False
                    # Get the job ready
                    if language == 'python':
                        script_files[i] = (mkstemp_safe(LOCAL_TEMP_PATH, '.py'))
                    elif language == 'matlab':
                        script_files[i] = (mkstemp_safe(LOCAL_TEMP_PATH, '.m'))
                    # Create necessary files in local path
                    shell_files[i] = (mkstemp_safe(LOCAL_TEMP_PATH, '.sh'))
                    output_files[i] = (mkstemp_safe(LOCAL_TEMP_PATH, '.out'))
                    stdout_files[i] = (mkstemp_safe(LOCAL_TEMP_PATH, '.o'))
                    flag_files[i] = (mkstemp_safe(LOCAL_TEMP_PATH, '.flag'))
                    # Customise code
                    #### TODO - make path and output_transfer optional
                    if language == 'python':
                        code = code + python_completion_code
                        for path in paths:
                            code = (python_path_code % path) + code
                    elif language == 'matlab':
                        code = code + matlab_completion_code
                        for path in paths:
                            code = (matlab_path_code % path) + code
                    code = code % {'output_file': output_files[i],
                                   'flag_file' : flag_files[i]}
                    # Write code and shell file
                    with open(script_files[i], 'w') as f:
                        f.write(code)
                    with open(shell_files[i], 'w') as f:
                        #### TODO - is os.path.join always correct - what happens if this program is being run on windows?
                        if language == 'python':
                            f.write('python ' + script_files[i] + '\n')
                        elif language == 'matlab':
                            f.write('cd ' + os.path.split(script_files[i])[0] + ';\n' + LOCAL_MATLAB + ' -nosplash -nojvm -nodisplay -singleCompThread -r ' + \
                                    os.path.split(script_files[i])[-1].split('.')[0] + '\n')
                    # Start running the job
                    print 'Submitting job %d of %d' % (i + 1, len(scripts))
                    stdout_file_handles[i] = open(stdout_files[i], 'w')
                    files_open = files_open + 1
                    processes[i] = subprocess.Popen(['sh', shell_files[i]], stdout = stdout_file_handles[i]);
                    # Sleep for a bit so the process can kick in (prevents 100s of jobs being sent to processor)
                    time.sleep(submit_sleep)
                else:
                    # If this is the last thing to happen in the loop then might as well sleep for a bit
                    should_sleep = True
            elif (not job_finished[i]) and (not processes[i] is None):
                # Ask the process how its doing
                processes[i].poll()
                # Check to see if the process has completed
                if not processes[i].returncode is None:
                    if os.path.isfile(flag_files[i]):
                        job_finished[i] = True
                        print 'Job %d of %d has completed' % (i + 1, len(scripts))
                    else:
                        print 'Job %d has failed - will try again later' % i + 1
                        processes[i] = None
                    # Tidy up temp files
                    os.remove(script_files[i])
                    os.remove(shell_files[i])
                    stdout_file_handles[i].close()
                    files_open = files_open - 1
                    os.remove(stdout_files[i])
                    os.remove(flag_files[i])
                    processes[i] = None
                else:
                    should_sleep = True
        if all(job_finished):
            fear_finished = True    
        elif should_sleep:
            # Count how many jobs are queued
            n_queued = 0
            # Count how many jobs are running
            n_running = 0
            if verbose:
                # print '%d jobs running' % n_running
                # print '%d jobs queued' % n_queued
                print 'Sleeping for %d seconds' % job_check_sleep
                time.sleep(job_check_sleep)
            should_sleep = False

    #### TODO - return job output and error files as applicable (e.g. there may be multiple error files associated with one script)
    return output_files

