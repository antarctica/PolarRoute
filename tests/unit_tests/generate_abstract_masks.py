import numpy as np
import pandas as pd


def find_data_fn(data_name):
    '''
    Retreive correct function to generate abstract mesh

    Args:
        data_name (str)  : name of abstract shape to generate
                        'circle', 'h_grad', 'v_grad', 'checkerboard', 'flat'
    Returns:
        gen_XYZ (fn)     : Function to generate data with
    '''
    if data_name == 'circle':
        return gen_circle
    elif data_name in ['gradient', 'v_grad', 'h_grad']:
        return gen_gradient
    elif data_name == 'checkerboard':
        return gen_checkerboard
    elif data_name == 'flat':
        return gen_flat
    else:
        raise Exception("Invalid data generation method: {}".format(data_name))

def gen_circle(latMin, latMax, longMin, longMax, time='1970-01-01', radius=1, centre=None, n=100, **kwargs):
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
        
        Returns:
            dummy_df (pd.DataFrame): Dataframe with abstract data to create mesh with
    """

    lat  = np.linspace(latMin, latMax, n)    # Generate rows
    long = np.linspace(longMin, longMax, n)  # Generate cols
    
    # Set centre as centre of data_grid if none specified
    if centre is None:
        centre = (lat[int(n/2)], long[int(n/2)])
    
    # Create vectors for row and col idx's
    y = np.vstack(np.linspace(latMin, latMax, n))
    x = np.linspace(longMin, longMax, n)
   
    # y, x = np.ogrid[:n, :n]              
    dist_from_centre = np.sqrt((x-centre[1])**2 + (y-centre[0])**2)     # Create a 2D-array with distance from defined centre
    mask = dist_from_centre <= radius    # Create mask
    # Set up empty dataframe to populate with dummy data
    dummy_df = pd.DataFrame(columns = ['lat', 'long', 'dummy_data'])
    # For each combination of lat/long
    for i in range(n):
        for j in range(n):
            # Create a new row, adding mask value
            row = pd.DataFrame(data={'lat':lat[i], 'long':long[j], 'dummy_data':mask[i][j]}, index=[0])
            dummy_df = pd.concat([dummy_df, row], ignore_index=True)
            
    # Change boolean values to int
    dummy_df = dummy_df.replace(False, 0)
    dummy_df = dummy_df.replace(True, 1)
    
    # Fill dummy time values
    dummy_df['time'] = time

    return dummy_df

def gen_gradient(latMin, latMax, longMin, longMax, time='1970-01-01', vertical=True, n=100, **kwargs):
    """
        Generates a gradient across the map

        Args:
            latMin (float)       : Minimum latitude of mesh
            latMax (float)       : Maximum latitude of mesh
            longMin (float)      : Minimum longitude of mesh
            longMax (float)      : Maximum longitude of mesh
            vertical (bool)      : Vertical gradient (true) or horizontal gradient (false)
            n (int)              : Intervals to divide lat and long range into
        
        Returns:
            dummy_df (pd.DataFrame): Dataframe with abstract data to create mesh with
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
            row = pd.DataFrame(data={'lat':lat[i], 'long':long[j], 'dummy_data':datum}, index=[0])
            dummy_df = pd.concat([dummy_df, row], ignore_index=True)  
    
    # Fill dummy time values
    dummy_df['time'] = time
    
    return dummy_df
    
def gen_checkerboard(latMin, latMax, longMin, longMax, time='1970-01-01', n=100, gridsize=(1,1), **kwargs):
    """
        Generates a checkerboard pattern across map

        Args:
            latMin (float)       : Minimum latitude of mesh
            latMax (float)       : Maximum latitude of mesh
            longMin (float)      : Minimum longitude of mesh
            longMax (float)      : Maximum longitude of mesh
            n (int)              : Intervals to divide lat and long range into
            gridsize (int, int)  : Tuple of size of boxes in checkerboard pattern, in form (lat(deg), long(deg))
        
        Returns:
            dummy_df (pd.DataFrame): Dataframe with abstract data to create mesh with    
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
            row = pd.DataFrame(data={'lat':lat[i], 'long':long[j], 'dummy_data':datum}, index=[0])
            dummy_df = pd.concat([dummy_df, row], ignore_index=True)  
    
    # Fill dummy time values
    dummy_df['time'] = time
    
    return dummy_df    

def gen_flat(*args, **kwargs):
    pass