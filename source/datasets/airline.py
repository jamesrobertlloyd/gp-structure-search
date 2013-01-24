import numpy as np

def data_file():
    return '../data/airline-pass.txt'

def load_data(fname):
    result = []
    for line in open(fname):
        values = map(float, line.strip().split())
        result += values
    return np.array(result)

def load_X_y():
    """X is a vector giving the time step, and y is the total number of international
    airline passengers, in thousands. Each element corresponds to one
    month, and it goes from Jan. 1949 through Dec. 1960."""
    values = load_data(data_file())
    return np.arange(values.size).astype(float), values.astype(float)


