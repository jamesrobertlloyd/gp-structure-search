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
from counter import Progress

from config import *

#### WISHLIST
####  - Limit number of active local jobs
####  - Display progress
####  - Loop more frequently
####  - Write setup function / make it possible for people to use this code without lots of hassle
####  - Provide convenience functions to setup MATLAB/python paths
####  - Merge job handling code
####  - Return STDOUT and STDERR from scripts
####  - Tidy up port forwarding
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
    
def start_port_forwarding():
    #### TODO - Make me nicer!
    cmd = 'ssh -N -f -L %d:fear:22 %s@gate.eng.cam.ac.uk' % (HOME_TO_REMOTE_PORT, USERNAME)
    subprocess.call(cmd.split(' '))
    cmd = 'ssh -N -f -R %d:localhost:22 -p %d %s@localhost' % (REMOTE_TO_HOME_PORT, HOME_TO_REMOTE_PORT, USERNAME)
    subprocess.call(cmd.split(' '))
    
#### TODO - Combine these functions? - they share much code
####      - Maybe this could be achieved by creating a generic object like fear that (re)moves files etc.
####      - but either does this on fear, on local machine, or on fear via gate.eng.cam.ac.uk

def run_batch_on_fear(scripts, language='python', job_check_sleep=30, file_copy_timeout=120, max_jobs=500, verbose=True):
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
    #### FIXME - Now does port forwarding but with fixed port numbers = bad
    if LOCATION == 'home':
        python_transfer_code = '''
from subprocess_timeout import timeoutCommand
print "Setting up port forwarding"
timeoutCommand(cmd='ssh -i %(rsa_remote)s -N -f -L %(r2rport)d:localhost:%(r2hport)d %(username)s@fear').run(timeout=%(timeout)d)
print "Moving output file"
if not timeoutCommand(cmd='scp -P %(r2rport)d -i %(rsa_home)s %(output_file)s %(home_user)s@localhost:%(local_temp_path)s; rm %(output_file)s').run(timeout=%(timeout)d)[0]:
    raise RuntimeError('Copying output raised error or timed out')
''' % {'rsa_remote' : REMOTE_TO_REMOTE_KEY_FILE,
       'r2rport' : REMOTE_TO_REMOTE_PORT,
       'r2hport' : REMOTE_TO_HOME_PORT,
       'username' : USERNAME,
       'timeout' : file_copy_timeout,
       'rsa_home' : REMOTE_TO_HOME_KEY_FILE,
       'output_file' : '%(output_file)s',
       'home_user' : HOME_USERNAME,
       'local_temp_path' : HOME_TEMP_PATH}
    else:    
        python_transfer_code = '''
#from util import timeoutCommand
from subprocess_timeout import timeoutCommand
print "Moving output file"
if not timeoutCommand(cmd='scp -i %(rsa_key)s %(output_file)s %(username)s@%(local_host)s:%(local_temp_path)s; rm %(output_file)s').run(timeout=%(timeout)d)[0]:
    raise RuntimeError('Copying output raised error or timed out')
''' % {'rsa_key' : REMOTE_TO_LOCAL_KEY_FILE,
       'output_file' : '%(output_file)s',
       'username' : USERNAME,
       'local_host' : LOCAL_HOST,
       'local_temp_path' : LOCAL_TEMP_PATH,
       'timeout' : file_copy_timeout}
    
    #### TODO - make this location independent
    #### TODO - does this suffer from the instabilities that lead to the verbosity of the python command above
    if LOCATION == 'home':
        matlab_transfer_code = '''
system('ssh -i %(rsa_remote)s -N -f -L %(r2rport)d:localhost:%(r2hport)d %(username)s@fear')
system('scp -P %(r2rport)d -i %(rsa_home)s %(output_file)s %(home_user)s@localhost:%(local_temp_path)s; rm %(output_file)s')
''' % {'rsa_remote' : REMOTE_TO_REMOTE_KEY_FILE,
       'r2rport' : REMOTE_TO_REMOTE_PORT,
       'r2hport' : REMOTE_TO_HOME_PORT,
       'username' : USERNAME,
       'rsa_home' : REMOTE_TO_HOME_KEY_FILE,
       'output_file' : '%(output_file)s',
       'home_user' : HOME_USERNAME,
       'local_temp_path' : HOME_TEMP_PATH}
    else:
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
    with pyfear.fear(via_gate=(LOCATION=='home')) as fear:
    
        # Initialise lists of file locations job ids
        shell_files = [None] * len(scripts)
        script_files = [None] * len(scripts)
        output_files = [None] * len(scripts)
        stdout_files = [None] * len(scripts)
        flag_files = [None] * len(scripts)
        job_ids = [None] * len(scripts) 
        fear_finished = False
        job_finished = [False] * len(scripts) 
        
        # Modify all scripts and create local temporary files
        #### TODO - Writing to the network can be slow
        ####      - Perhaps it would be better writing to local disk followed by a block file transfer (is this possible?)
        ####      - Would make sense to do this when reading output files as well - perhaps this should be provided in this module?
        
        for (i, code) in enumerate(scripts):
            print 'Writing temp files for job %d of %d' % (i + 1, len(scripts))
            if LOCATION == 'local':
                temp_dir = LOCAL_TEMP_PATH
            else:
                temp_dir = HOME_TEMP_PATH
            if language == 'python':
                script_files[i] = mkstemp_safe(temp_dir, '.py')
            elif language == 'matlab':
                script_files[i] = mkstemp_safe(temp_dir, '.m')
            shell_files[i] = mkstemp_safe(temp_dir, '.sh')
            output_files[i] = mkstemp_safe(temp_dir, '.out')
            flag_files[i] = mkstemp_safe(temp_dir, '.flg')
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
            

        # Loop through jobs, submitting jobs whenever fear usage low enough, re-submitting failed jobs
        while not fear_finished:
            # Update knowledge of fear - trying to limit communication
            fear.qstat()
            jobs_alive = fear.jobs_alive(update=False)
            # Sleep unless anything happens
            should_sleep = True
            for i in range(len(scripts)): # Make me more pythonic with zipping
                # Does the job need to be run and can we run it?
                if (not job_finished[i]) and (job_ids[i] is None) and (jobs_alive <= max_jobs):
                    # Something has happened
                    should_sleep = False
                    # Transfer files to fear
                    fear.copy_to_temp(script_files[i])
                    fear.copy_to_temp(shell_files[i])
                    # Submit the job to fear
                    print 'Submitting job %d of %d' % (i + 1, len(scripts))
                    job_ids[i] = fear.qsub(os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]), verbose=verbose) # Hide path constant
                    # Increment job count
                    jobs_alive += 1
                # Otherwise was it running last we checked?
                elif (not job_finished[i]) and (not job_ids[i] is None):
                    # Has the process terminated?
                    if fear.job_terminated(job_ids[i], update=False):
                        # Decrement job count
                        #jobs_alive -= 1 - would not have been counted earlier
                        should_sleep = False
                        # Has the job failed to write a flag or is the output file empty
                        #### FIXME - If LOCATION=='home' need to check .out file on local is non-empty
                        #### TODO - Can likely increase speed by checking status of files (on fear and local) in one block - not sure how to do it though
                        ####      - In particular checking for file existence should probably be done with a SFTP client rather than SSH - might be faster?
                        if (not fear.file_exists(os.path.join(REMOTE_TEMP_PATH, os.path.split(flag_files[i])[-1]))) or \
                           (os.stat(output_files[i]).st_size == 0):
                            # Job has finished but missing output - resubmit later
                            print 'Shell script %s job_id %s failed' % (os.path.split(shell_files[i])[-1], job_ids[i])
                            # Save job id for file deletion
                            old_job_id = job_ids[i]
                            job_ids[i] = None
                        else:
                            # Job has finished successfully
                            job_finished[i] = True
                            # Save job id for file deletion
                            old_job_id = job_ids[i]
                            # Move files if necessary
                            #if LOCATION=='home':
                            #    # Copy the file from local storage machine (and delete it)
                            #    fear.copy_from_localhost(localpath=output_files[i], remotepath=os.path.join(LOCAL_TEMP_PATH, os.path.split(output_files[i])[-1]))
                            # Tell the world
                            if verbose:
                                print '%d / %d jobs complete' % (sum(job_finished), len(job_finished))
                            # Tidy up local temporary directory - actually - do this in one batch later
                            #os.remove(script_files[i])
                            #os.remove(shell_files[i])
                            #os.remove(flag_files[i])
                        # Tidy up fear
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(script_files[i])[-1]))
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]))
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(flag_files[i])[-1]))
                        #### TODO - record the output and error files for future reference
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]) + ('.o%s' % old_job_id))
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]) + ('.e%s' % old_job_id))
                        #### TODO - is the following line faster?
                        #fear.command('rm ' + ' ; rm '.join([os.path.join(REMOTE_TEMP_PATH, os.path.split(script_files[i])[-1]), os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]), os.path.join(REMOTE_TEMP_PATH, os.path.split(flag_files[i])[-1]), os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]) + ('.o%s' % old_job_id), os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]) + ('.e%s' % old_job_id)]))
                        # Tidy up local temporary directory
                        #os.remove(script_files[i])
                        #os.remove(shell_files[i])
                        #os.remove(flag_files[i])    
                        job_ids[i] = None                     
                            
                    elif not (fear.job_queued(job_ids[i]) or fear.job_running(job_ids[i]) \
                              or fear.job_loading(job_ids[i])):
                        # Job has some status other than running, queuing or loading - something is wrong, delete it
                        jobs_alive -= 1
                        should_sleep = False
                        old_job_id = job_ids[i]
                        fear.qdel(job_ids[i])
                        print 'Shell script %s job_id %s stuck, deleting' % (os.path.split(shell_files[i])[-1], job_ids[i])
                        #### TODO - remove this code duplication
                        # Tidy up fear
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(script_files[i])[-1]))
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]))
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(flag_files[i])[-1]))
                        #### TODO - record the output and error files for future reference
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]) + ('.o%s' % old_job_id))
                        fear.rm(os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]) + ('.e%s' % old_job_id))
                        #### TODO - is the following line faster?
                        #fear.command('rm ' + ' ; rm '.join([os.path.join(REMOTE_TEMP_PATH, os.path.split(script_files[i])[-1]), os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]), os.path.join(REMOTE_TEMP_PATH, os.path.split(flag_files[i])[-1]), os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]) + ('.o%s' % old_job_id), os.path.join(REMOTE_TEMP_PATH, os.path.split(shell_files[i])[-1]) + ('.e%s' % old_job_id)]))
                        # Tidy up local temporary directory
                        #os.remove(script_files[i])
                        #os.remove(shell_files[i])
                        #os.remove(flag_files[i])    
                        job_ids[i] = None   
            if all(job_finished):
                fear_finished = True    
            elif should_sleep:
                if verbose:
                    #fear.qstat()
                    print '%d of %d jobs complete' % (sum(job_finished), len(job_finished))
                    print '%d jobs running' % fear.jobs_running(update=False)
                    print '%d jobs loading' % fear.jobs_loading(update=False)
                    print '%d jobs queued' % fear.jobs_queued(update=False)
                    print 'Sleeping for %d seconds' % job_check_sleep
                    time.sleep(job_check_sleep)

    # Tidy up temporary directory
    for i in range(len(scripts)):
        print 'Removing temp files for job %d of %d' % (i + 1, len(scripts))
        os.remove(script_files[i])
        os.remove(shell_files[i])
        os.remove(flag_files[i])

    #### TODO - return job output and error files as applicable (e.g. there may be multiple error files associated with one script)
    return output_files
    
def run_batch_locally(scripts, language='python', paths=[], max_cpu=0.9, max_mem=0.9, submit_sleep=1, job_check_sleep=30, \
                      verbose=True, max_files_open=100, max_running_jobs=10):
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
    
    files_open = 0

    # Loop through jobs, submitting jobs whenever CPU usage low enough, re-submitting failed jobs
    if not verbose:
        prog = Progress(len(scripts))
    while not fear_finished:
        should_sleep = True
        for (i, code) in enumerate(scripts):
            if (not job_finished[i]) and (processes[i] is None) and (files_open <= max_files_open) and (len([1 for p in processes if not p is None]) < max_running_jobs):
                # This script has not been run - check CPU and potentially run
                #### FIXME - Merge if statements
                if (psutil.cpu_percent() < max_cpu * 100) and (psutil.virtual_memory().percent < max_mem * 100):
                    # Jobs can run
                    should_sleep = False
                    # Get the job ready
                    if LOCATION == 'local':
                        temp_dir = LOCAL_TEMP_PATH
                    else:
                        temp_dir = HOME_TEMP_PATH
                    if language == 'python':
                        script_files[i] = (mkstemp_safe(temp_dir, '.py'))
                    elif language == 'matlab':
                        script_files[i] = (mkstemp_safe(temp_dir, '.m'))
                    # Create necessary files in local path
                    shell_files[i] = (mkstemp_safe(temp_dir, '.sh'))
                    output_files[i] = (mkstemp_safe(temp_dir, '.out'))
                    stdout_files[i] = (mkstemp_safe(temp_dir, '.o'))
                    flag_files[i] = (mkstemp_safe(temp_dir, '.flg'))
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
                            if LOCATION == 'home':
                                matlab_path = HOME_MATLAB
                            else:
                                matlab_path = LOCAL_MATLAB
                            f.write('cd ' + os.path.split(script_files[i])[0] + ';\n' + matlab_path + ' -nosplash -nojvm -nodisplay -singleCompThread -r ' + \
                                    os.path.split(script_files[i])[-1].split('.')[0] + '\n')
                    # Start running the job
                    if verbose:
                        print 'Submitting job %d of %d' % (i + 1, len(scripts))
                    stdout_file_handles[i] = open(stdout_files[i], 'w')
                    files_open = files_open + 1
                    processes[i] = subprocess.Popen(['sh', shell_files[i]], stdout = stdout_file_handles[i]);
                    # Sleep for a bit so the process can kick in (prevents 100s of jobs being sent to processor)
                    time.sleep(submit_sleep)
            elif (not job_finished[i]) and (not processes[i] is None):
                # Ask the process how its doing
                processes[i].poll()
                # Check to see if the process has completed
                if not processes[i].returncode is None:
                    if os.path.isfile(flag_files[i]):
                        job_finished[i] = True
                        if verbose:
                            print 'Job %d of %d has completed' % (i + 1, len(scripts))
                        else:
                            prog.tick()
                    else:
                        if verbose:
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
                    # Something useful happened
                    should_sleep = False
        if all(job_finished):
            fear_finished = True 
            if not verbose: 
                prog.done()  
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

    #### TODO - return job output and error files as applicable (e.g. there may be multiple error files associated with one script)
    return output_files

