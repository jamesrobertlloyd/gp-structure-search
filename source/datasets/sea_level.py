import numpy as np
import scipy.io

"""Dataset of sea levels collected by the PSMSL. The original datasets were downloaded
from http://www.psmsl.org/data/obtaining/complete.php. We ran the provided scripts
to generate the two .mat files for the data directory."""


class MonthlyStationInfo:
    def __init__(self, station_id, latitude, longitude, name, stationflag,
                 year, month, time, height, missing, dataflag):
        self.station_id = station_id
        self.latitude = latitude
        self.longitude = longitude
        self.name = name
        self.stationflag = stationflag
        self.year = year
        self.month = month
        self.time = time
        self.height = height
        self.missing = missing
        self.dataflag = dataflag

class AnnualStationInfo:
    def __init__(self, station_id, latitude, longitude, name, stationflag,
                 year, height, interpolated, dataflag):
        self.station_id = station_id
        self.latitude = latitude
        self.longitude = longitude
        self.name = name
        self.stationflag = stationflag
        self.year = year
        self.height = height
        self.interpolated = interpolated
        self.dataflag = dataflag

def monthly_matfile():
    return '../data/sea_level_monthly.mat'

def annual_matfile():
    return '../data/sea_level_annual.mat'

def read_monthly_data(matfile):
    data = scipy.io.loadmat(matfile)['data']
    num_stations = data.shape[1]

    all_station_info = []
    for st in range(num_stations):
        curr_data = data[0, st]
        station_id = curr_data['id'][0, 0]
        latitude, longitude = curr_data['latitude'][0, 0], curr_data['longitude'][0, 0]
        name = curr_data['name'][0]
        stationflag = curr_data['stationflag'][0, 0]
        year = curr_data['year'].ravel()
        month = curr_data['month'].ravel()
        time = curr_data['time'].ravel()
        height = curr_data['height'].ravel()
        missing = curr_data['missing'].ravel()
        dataflag = curr_data['dataflag'].ravel()
        info = MonthlyStationInfo(station_id, latitude, longitude, name, stationflag,
                                  year, month, time, height, missing, dataflag)
        all_station_info.append(info)

    return all_station_info

def read_annual_data(matfile):
    data = scipy.io.loadmat(matfile)['data']
    num_stations = data.shape[1]

    all_station_info = []
    for st in range(num_stations):
        curr_data = data[0, st]
        station_id = curr_data['id'][0, 0]
        latitude, longitude = curr_data['latitude'][0, 0], curr_data['longitude'][0, 0]
        name = curr_data['name'][0]
        stationflag = curr_data['stationflag'][0, 0]
        year = curr_data['year'].ravel()
        height = curr_data['height'].ravel()
        interpolated = curr_data['interpolated'].ravel()
        dataflag = curr_data['dataflag'].ravel()
        info = AnnualStationInfo(station_id, latitude, longitude, name, stationflag,
                                 year, height, interpolated, dataflag)
        all_station_info.append(info)

    return all_station_info


def get_X_y(version):
    """Return a data matrix X where the three columns represent latitude, longitude,
    and time, and a vector y representing height readings."""
    if version == 'annual':
        all_station_info = read_annual_data(annual_matfile())
    elif version == 'monthly':
        all_station_info = read_monthly_data(monthly_matfile())
    else:
        raise RuntimeError('Unknown version: %s' % version)

    
    # eliminate flagged stations
    all_station_info = [si for si in all_station_info if not si.stationflag]

    X_list = []
    y_list = []
    for si in all_station_info:
        if version == 'annual':
            time = si.year.astype(float)
            flag = si.interpolated + si.dataflag
        else:
            time = si.time.astype(float)
            flag = si.missing + si.dataflag
            
        for t, h, f in zip(time, si.height, flag):
            if not f:  # ignore flagged datapoints
                X_list.append((si.latitude, si.longitude, t))
                y_list.append(h)

    X = np.array(X_list)
    y = np.array(y_list)

    # for some reason, there's missing data (marked as NaNs), even beyond
    # what is flagged
    idxs = np.where(-np.isnan(y))[0]
    X = X[idxs, :]
    y = y[idxs]

    return X, y


    
    
