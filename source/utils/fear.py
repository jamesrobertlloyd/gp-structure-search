'''
A set of utilities to maniuplate files on the fear cluster.

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          David Duvenaud (dkd23@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
'''

import pysftp
import config
import re

# TODO:  Make this into a class.

def connect():
    return pysftp.Connection('fear', username=config.USERNAME, password=config.PASSWORD)

def command(cmd, fear=None):
    if not fear is None:
        srv = fear
    else:
        srv = connect()
    output =  srv.execute(cmd)
    if fear is None:
        srv.close()
    return output
    
def copy_to(local_path, remote_path, fear=None):
    if not fear is None:
        srv = fear
    else:
        srv = connect()
    srv.put(local_path, remote_path)
    if fear is None:
        srv.close()
    
def copy_from(remote_path, local_path, fear=None):
    if not fear is None:
        srv = fear
    else:
        srv = connect()
    srv.get(remote_path, local_path)
    if fear is None:
        srv.close()
    
def rm(remote_path, fear=None):
    if not fear is None:
        srv = fear
    else:
        srv = connect()
    output =  srv.execute('rm %s' % remote_path)
    if fear is None:
        srv.close()
    return output

def file_exists(remote_path, fear=None):
    if not fear is None:
        srv = fear
    else:
        srv = connect()
    response = srv.execute('if [ -e %s ] \nthen \necho ''exists'' \nfi' % remote_path)
    if fear is None:
        srv.close()
    return response == ['exists\n']

def qsub(shell_file, verbose=True, fear=None):
    '''Submit a job onto the stack.'''
    
    #### WARNING - hardcoded path 'temp'
    fear_string = ' '.join(['. /usr/local/grid/divf2/common/settings.sh;',
                            'cd %s;' % config.REMOTE_TEMP_PATH,
                            'chmod +x %s;' % shell_file.split('/')[-1],
                            'qsub -l lr=0',
                            shell_file.split('/')[-1] + ';',
                            'cd ..'])

    if verbose:
        print 'Submitting : %s' % fear_string
    output_text = command(fear_string, fear)
    # Return the job id
    return output_text[0].split(' ')[2]

def qdel(job_id, fear=None):
    if not fear is None:
        srv = fear
    else:
        srv = connect()
    output = srv.execute('. /usr/local/grid/divf2/common/settings.sh; qdel %s' % job_id)
    if fear is None:
        srv.close()
    return output

def qdel_all(fear=None):
    if not fear is None:
        srv = fear
    else:
        srv = connect()
    output = srv.execute('. /usr/local/grid/divf2/common/settings.sh; qdel -u %s' % config.USERNAME)
    if fear is None:
        srv.close()
    return output

def qstat_status(fear=None):
    '''Returns a dictionary with (job id, status) pairs'''
    if not fear is None:
        srv = fear
    else:
        srv = connect()
    test_output = srv.execute('. /usr/local/grid/divf2/common/settings.sh; qstat -u %s' % config.USERNAME)
    # Now process this text to turn it into a list of job statuses
    # First remove multiple spaces from the interesting lines
    without_multi_space = [re.sub(' +',' ',line) for line in test_output[2:]]
    # Now create a dictionary of job ids and statuses
    status = {key: value for (key, value) in zip([line.split(' ')[0] for line in without_multi_space], \
                                                 [line.split(' ')[4] for line in without_multi_space])}
    if fear is None:
        srv.close()
    return status 

def job_terminated(job_id, status=None, fear=None):
    '''Returns true if job not listed by qstat'''
    if status is None:
        status = qstat_status(fear)
    return not status.has_key(job_id)

def job_running(job_id, status=None, fear=None):
    if status is None:
        status = qstat_status(fear)
    if status.has_key(job_id):
        return status[job_id] == 'r'
    else:
        return False

def job_queued(job_id, status=None, fear=None):
    if status is None:
        status = qstat_status(fear)
    if status.has_key(job_id):
        return status[job_id] == 'qw'
    else:
        return False

def job_loading(job_id, status=None, fear=None):
    if status is None:
        status = qstat_status(fear)
    if status.has_key(job_id):
        return status[job_id] == 't'
    else:
        return False
