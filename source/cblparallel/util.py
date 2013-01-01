'''
Miscellaneous utilities for cblparallel
'''

import tempfile
import os

def mkstemp_safe(directory, suffix):
    (os_file_handle, file_name) = tempfile.mkstemp(dir=directory, suffix=suffix)
    os.close(os_file_handle)
    return file_name
    
import subprocess, threading

#### To think about: Could I just run a timeout function in a separate thread rather than starting a new process?

class timeoutCommand(object):
    def __init__(self, cmd, verbose=False):
        self.cmd = cmd
        self.verbose = verbose
        self.process = None

    def run(self, timeout):
        def target():
            if self.verbose:
                print 'Thread started'
            self.process = subprocess.Popen(self.cmd, shell=True)
            self.process.communicate()
            if self.verbose:
                print 'Thread finished'

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        
        if thread.is_alive():
            if self.verbose:
                print 'Terminating process'
            try:
                self.process.terminate()
                thread.join()
                if self.verbose:    
                    print self.process.returncode                
                return (False, self.process.returncode)
            except:
                print 'Could not terminate process - maybe there was a race'
                return (False, 0)
        else:
            if self.verbose:    
                print self.process.returncode
            return (True, self.process.returncode)
