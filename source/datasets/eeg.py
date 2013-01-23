import numpy as np
import scipy.io

def eeg_data_file():
    return '../data/eeg_cba_data.mat'

def channel_locs_file():
    return '../data/eeg_channel_locs.txt'

class WaveformInfo:
    """Waveform data for a single subject. Waveforms is a 3-D array, where the dimensions
    correspond to channel, time, and event."""
    def __init__(self, waveforms, times):
        self.waveforms = waveforms
        self.times = times

def load_waveform_data(data_file):
    eeg = scipy.io.loadmat(data_file)['EEG']
    waveforms = eeg['data'][0, 0]
    times = eeg['times'][0, 0].ravel()
    return WaveformInfo(waveforms, times)

def load_channel_locs(locs_file):
    """Load an array giving the spatial coordinates of the 31 electrodes.
    I'm not clear what coordinate system is being used. My best guess is the
    first column is horizontal angle in degrees and the second is vertical
    angle in radians."""
    locs = []
    for line in open(locs_file):
        phi, theta = map(float, line.strip().split())
        locs.append((phi, theta))
    return np.array(locs)


def load_one_channel(channel_id=0, event_id=0):
    """Returns a 1-D dataset corresponding to a single electrode for a single subject
    for a single event."""
    info = load_waveform_data(eeg_data_file())
    X = info.times.astype(float)
    y = info.waveforms[channel_id, :, event_id].astype(float)
    return X.copy(), y.copy()

def load_all_channels(event_id=0):
    """Returns a 3-D dataset corresponding to all the electrodes for a single subject
    and a single event. The first two columns of X give the spatial dimensions, and
    the third dimension gives the time."""
    info = load_waveform_data(eeg_data_file())
    locs = load_channel_locs(channel_locs_file())
    nchan, ntime, nev = info.waveforms.shape

    X = np.zeros((0, 3))
    y = np.zeros(0)
    for c in range(nchan):
        curr_X = np.zeros((ntime, 3))
        curr_X[:, 0] = locs[c, 0]
        curr_X[:, 1] = locs[c, 1]
        curr_X[:, 2] = info.times
        curr_y = info.waveforms[c, :, event_id].astype(float)
        X = np.vstack([X, curr_X])
        y = np.concatenate([y, curr_y])

    return X, y

        

