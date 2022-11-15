import pytest
import numpy as np
import pandas as pd



def gen_circle(latMin, latMax, longMin, longMax, radius, centre=None, n=100, time_str='1970-01-01'):
    """
        Generates a circle within bounds of lat/long min/max.

        Args:
            latMin (float)       : Minimum latitude of mesh
            latMax (float)       : Maximum latitude of mesh
            longMin (float)      : Minimum longitude of mesh
            longMax (float)      : Maximum longitude of mesh
            radius (float)       : Radius of circle to generate, in degrees
            centre (float, float): Tuple of central coordinates of circle in form (lat(deg), long(deg)).
                                   If None, then centre of mesh is chosen
            n (int)              : Intervals to divide lat and long range into
            time_str (str)       : Dummy timestamp for dummy data, in form 'YYYY-MM-DD'
    """

    lat  = np.linspace(latMin, latMax, n)    # Generate rows
    long = np.linspace(longMin, longMax, n)  # Generate cols
    
    # Set centre as centre of data_grid if none specified
    if centre is None:
        centre = (long[int(n/2)], lat[int(n/2)])
    
    # Create vectors for row and col idx's
    y = np.vstack(np.linspace(latMin, latMax, n))
    x = np.linspace(longMin, longMax, n)
   
    # y, x = np.ogrid[:n, :n]              
    dist_from_centre = np.sqrt((x-centre[0])**2 + (y-centre[1])**2)     # Create a 2D-array with distance from defined centre
    mask = dist_from_centre <= radius    # Create mask
    # Set up empty dataframe to populate with dummy data
    dummy_df = pd.DataFrame(columns = ['lat', 'long', 'time', 'dummy_data'])
    # For each combination of lat/long
    for i in range(n):
        for j in range(n):
            # Create a new row, adding mask value
            row = pd.DataFrame(data={'lat':lat[i], 'long':long[j], 
                                     'time': time_str, 'dummy_data':mask[i][j]}, 
                                     index=[0])
            dummy_df = pd.concat([dummy_df, row], ignore_index=True)
            
    # Change boolean values to int
    dummy_df = dummy_df.replace(False, 0)
    dummy_df = dummy_df.replace(True, 1)

    return dummy_df

def gen_gradient(latMin, latMax, longMin, longMax, vertical=True, n=100, time_str='1970-01-01'):
    """
        Generates a gradient across the map

        Args:
            latMin (float)       : Minimum latitude of mesh
            latMax (float)       : Maximum latitude of mesh
            longMin (float)      : Minimum longitude of mesh
            longMax (float)      : Maximum longitude of mesh
            vertical (bool)      : Vertical gradient (true) or horizontal gradient (false)
            n (int)              : Intervals to divide lat and long range into
            time_str (str)       : Dummy timestamp for dummy data, in form 'YYYY-MM-DD'
    """
    lat  = np.linspace(latMin, latMax, n)    # Generate rows
    long = np.linspace(longMin, longMax, n)  # Generate cols
    #Create 1D gradient
    gradient = np.linspace(0,1,n)
    
    dummy_df = pd.DataFrame(columns = ['lat', 'long', 'dummy_data'])
    # For each combination of lat/long
    for i in range(n):
        for j in range(n):
            # Change dummy data depending on which axis to gradient
            datum = gradient[i] if vertical else gradient[j]
            # Create a new row, adding datum value
            row = pd.DataFrame(data={'lat':lat[i], 'long':long[j], 
                                     'time': time_str, 'dummy_data':datum}, 
                                     index=[0])
            dummy_df = pd.concat([dummy_df, row], ignore_index=True)  
    return dummy_df
    
def gen_checkerboard(latMin, latMax, longMin, longMax, gridsize=(1,1), n=100, time_str='1970-01-01'):
    """
        Generates a checkerboard pattern across map

        Args:
            latMin (float)       : Minimum latitude of mesh
            latMax (float)       : Maximum latitude of mesh
            longMin (float)      : Minimum longitude of mesh
            longMax (float)      : Maximum longitude of mesh
            gridsize (int, int)  : Tuple of size of boxes in checkerboard pattern, in form (lat(deg), long(deg))
            n (int)              : Intervals to divide lat and long range into
            time_str (str)       : Dummy timestamp for dummy data, in form 'YYYY-MM-DD'
    """
    lat  = np.linspace(latMin, latMax, n)    # Generate rows
    long = np.linspace(longMin, longMax, n)  # Generate cols
    
    # Create checkerboard pattern
    horizontal = np.floor((lat - latMin)   / (2 * gridsize[0])) % 2   # Create horizontal stripes of 0's and 1's, stripe size defined by gridsize
    vertical   = np.floor((long - longMin) / (2 * gridsize[0])) % 2   # Create vertical stripes of 0's and 1's, stripe size defined by gridsize
    dummy_df = pd.DataFrame(columns = ['lat', 'long', 'dummy_data'])
    # For each combination of lat/long
    for i in range(n):
        for j in range(n):
            # Horizontal XOR Vertical should create boxes
            datum = (horizontal[i] + vertical[j]) % 2
            # Create a new row, adding datum value
            row = pd.DataFrame(data={'lat':lat[i], 'long':long[j], 
                                     'time': time_str, 'dummy_data':datum}, 
                                     index=[0])
            dummy_df = pd.concat([dummy_df, row], ignore_index=True)  
    return dummy_df   

