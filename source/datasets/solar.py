import numpy as np
nax = np.newaxis

"""Solar irradiance data, downloaded from
lasp.colorado.edu/sorce/tsi_data/TSI_TIM_Reconstruction.txt"""

def irradiance_data_file():
    return '../data/TSI_TIM_Reconstruction.txt'

def get_X_y():
    """Returns a Tx1 matrix X representing the year, and a length-T
    vector y representing the solar irradiance."""
    x_list = []
    y_list = []
    for line in open(irradiance_data_file()):
        if line[0] == ';':
            continue

        parts = line.strip().split()
        year = float(parts[0])
        irrad = float(parts[1])
        x_list.append(year)
        y_list.append(irrad)

    X = np.array(x_list)[nax]
    y = np.array(y_list)
    return X, y

    
