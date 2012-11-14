'''
A set of utilities to maniuplate files on the fear cluster.

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          David Duvenaud (dkd23@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
'''

import pysftp

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

def qdel_all(fear=None):
    if not fear is None:
        srv = fear
    else:
        srv = connect()
    output = srv.execute('. /usr/local/grid/divf2/common/settings.sh; qdel -u %s' % config.USERNAME)
    if fear is None:
        srv.close()
    return output
