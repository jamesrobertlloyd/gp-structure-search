import numpy as np
import os
import pylab
import subprocess

import config

ANNO_EXTENSIONS = {'fantasia': 'ecg',
                   'ltstdb': 'atr',
                   'mghdb': 'ari',
                   'mitdb': 'atr',
                   }

def datset_url(dataset):
    return 'http://www.physionet.org/physiobank/database/%s' % dataset

def record_url(dataset, record, extension):
    return 'http://www.physionet.org/physiobank/database/%s/%s.%s' % (dataset, record, extension)

def temp_dir():
    return config.TEMP_PATH

def dest_file(dataset, record, extension):
    return os.path.join(temp_dir(), '%s.%s' % (record, extension))

def load_data(dataset, record, freq=2.):
    anno_ext = ANNO_EXTENSIONS[dataset]
    for ext in ['dat', 'hea', anno_ext]:
        dest = dest_file(dataset, record, ext)
        if not os.path.exists(dest):
            ret = subprocess.call(['wget', record_url(dataset, record, ext), '-O', dest])
            assert ret == 0

    os.environ['WFDB'] = temp_dir()
    outfile = os.path.join(temp_dir(), 'heartrate.txt')
    ret = subprocess.call(['tach', '-r', record, '-a', anno_ext, '-F', str(freq)],
                          stdout=open(outfile, 'w'))
    assert ret == 0

    return map(float, open(outfile).readlines())
    


def plot_examples():
    #DATASET = 'fantasia'
    #RECORDS = ['f1y01', 'f1y02', 'f1y03', 'f1y04', 'f1y05']

    DATASET = 'mghdb'
    #RECORDS = ['mgh156', 'mgh241', 'mgh001', 'mgh002']
    #RECORDS = ['mgh005', 'mgh009', 'mgh016', 'mgh019']
    RECORDS = ['mgh019', 'mgh023', 'mgh027']

    #DATASET = 'ltstdb'
    #RECORDS = ['s20011', 's20021', 's20031', 's20041']
    
    for rec in RECORDS:
        hr = load_data(DATASET, rec, freq=0.1)
        pylab.figure()
        pylab.plot(hr)
        pylab.title(rec)

def save_data():
    VERSIONS = [('fantasia', 'f1y01', 'healthy'),
                ('mghdb', 'mgh156', 'congestive'),
                ('mghdb', 'mgh019', 'atrial-fib'),
                ]
    for dataset, record, name in VERSIONS:
        hr = load_data(dataset, record, 0.1)
        outfile = open('../data/ekg_hr_%s.txt' % name, 'w')
        for elt in hr:
            print >> outfile, elt
        outfile.close()


def get_X_y(name):
    assert name in ['healthy', 'congestive', 'atrial-fib']
    y = map(float, open('../data/ekg_hr_%s.txt' % name).readlines())
    X = np.arange(len(y), dtype=float)
    return X, y

