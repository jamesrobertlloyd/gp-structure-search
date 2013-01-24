import glob
import os
import smtplib
import socket
import subprocess
import sys

import config

def _status_path(key):
    return os.path.join(config.JOBS_PATH, key)

def _status_file(key, host=None):
    if host is not None:
        return os.path.join(_status_path(key), 'status-%s.txt' % host)
    else:
        return os.path.join(_status_path(key), 'status.txt')

def _run_job(script, key, args):
    if key != 'None':
        outstr = open(_status_file(key, socket.gethostname()), 'a')
        print >> outstr, 'running:', args
        outstr.close()
        
    ret = subprocess.call('python %s %s' % (script, args), shell=True)

    if key != 'None':
        outstr = open(_status_file(key, socket.gethostname()), 'a')
        if ret == 0:
            print >> outstr, 'finished:', args
        else:
            print >> outstr, 'failed:', args
        outstr.close()

    

def run_command(command, jobs, machines=None, chdir=None):
    args = ['parallel']
    if machines is not None:
        for m in machines:
            args += ['--sshlogin', m]

    if chdir is not None:
        command = 'cd %s; %s' % (chdir, command)
    args += [command]

    p = subprocess.Popen(args, shell=False, stdin=subprocess.PIPE)
    p.communicate('\n'.join(jobs))

def run(script, jobs, machines=None, key=None, email=True, rm_status=True):
    if key is not None:
        if not os.path.exists(_status_path(key)):
            os.mkdir(_status_path(key))
            
        outstr = open(_status_file(key), 'w')
        for job in jobs:
            print >> outstr, 'queued:', job
        outstr.close()

        if rm_status:
            subprocess.call('cd %s; rm status-*.txt' % _status_path(key), shell=True)
        
    command = 'python utils/parallel.py %s %s' % (key, script)
    run_command(command, jobs, machines=machines, chdir=config.CODE_PATH)

    if email:
        if key is not None:
            subject = '%s jobs finished' % key
            p = subprocess.Popen(['check_status', key], stdout=subprocess.PIPE)
            body, _ = p.communicate()
        else:
            subject = 'jobs finished'
            body = ''

        msg = '\r\n'.join(['From: %s' % config.EMAIL,
                           'To: %s' % config.EMAIL,
                           'Subject: %s' % subject,
                           '',
                           body])
        
        s = smtplib.SMTP('localhost')
        s.sendmail(config.EMAIL, [config.EMAIL], msg)
        s.quit()

def isint(p):
    try:
        int(p)
        return True
    except:
        return False

def parse_machines(s, njobs):
    if s is None:
        return s
    parts = s.split(',')
    result = []
    for p in parts:
        if p == ':':
            result.append('%d/:' % njobs)
        elif p.find(':') != -1:
            lower_str, upper_str = p.split(':')
            for i in range(int(lower_str), int(upper_str) + 1):
                result.append('%d/vision%02d' % (njobs, i))
        elif isint(p):
            result.append('%d/vision%02d' % (njobs, int(p)))
        else:
            result.append('%d/%s' % (njobs, p))
    return result

def list_jobs(key, status_val):
    status_files = [os.path.join(_status_path(key), 'status.txt')]
    status_files += glob.glob('%s/status-*.txt' % _status_path(key))

    status = {}
    for fname in status_files:
        for line_ in open(fname).readlines():
            line = line_.strip()
            sv, args = line.split(':')
            args = args.strip()
            status[args] = sv

    return [k for k, v in status.items() if v == status_val]


if __name__ == '__main__':
    assert len(sys.argv) == 4
    key = sys.argv[1]
    script = sys.argv[2]
    args = sys.argv[3]
    _run_job(script, key, args)
