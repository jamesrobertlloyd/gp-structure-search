import datetime
import numpy as np
nax = np.newaxis


"""Methane data from the NOAA. See ftp://ftp.cmdl.noaa.gov/ccg/ch4/flask/README_surface_flask_ch4.html"""

def datafile():
    return '../data/ch4_brw_surface-flask_1_ccgg_event.txt'

def read_data(fname=None):
    """Returns a Tx1 array X which gives time (in seconds since the epoch) and a length-T
    vector giving the CH4 measurement at that time."""
    if fname is None:
        fname = datafile()

    epoch = datetime.datetime.utcfromtimestamp(0)
        
    x_list = []
    y_list = []
    for line_ in open(fname):
        line = line_.strip()
        if line[0] == '#':
            continue

        parts = line.split()
        year = int(parts[1])
        month = int(parts[2])
        day = int(parts[3])
        hour = int(parts[4])
        minute = int(parts[5])
        second = int(parts[6])
        value = float(parts[11])
        flags = parts[13]

        # ignore anything that's flagged
        if flags != '...':
            continue

        dtime = datetime.datetime(year, month, day, hour, minute, second)
        delta = dtime - epoch
        x_list.append(delta.total_seconds())
        y_list.append(value)

    X = np.array(x_list)[:, nax]
    y = np.array(y_list)
    return X, y



