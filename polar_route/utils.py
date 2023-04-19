import logging
import time
import tracemalloc

from datetime import datetime, timedelta
from functools import wraps
from calendar import monthrange

import numpy as np
from scipy.fftpack import fftshift
from math import log10, floor

"""
Utilities that might be of use
"""
def rectangle_overlap(a_coords, b_coords):
    # Rectangles must have parallel lines
    # Coords are tuples of format
    # ((x0, y0), (x2, y2))
    # where 0/2 denote the min/max extent 
    # of the x/y coords of the rect
    min_a_coords = a_coords[0]
    max_a_coords = a_coords[1]
    min_b_coords = b_coords[0]
    max_b_coords = b_coords[1]
    
    dx = min(max_a_coords[0], max_b_coords[0]) - \
         max(min_a_coords[0], min_b_coords[0])
         
    dy = min(max_a_coords[1], max_b_coords[1]) - \
         max(min_a_coords[1], min_b_coords[1])
         
    return dx*dy
    
    
def frac_of_month(year, month, start_date=None, end_date=None):
    
    # Determine the number of days in the month specified
    days_in_month = monthrange(year, month)[1]
    # If not specified, default to beginning/end of month
    if start_date is None:
        start_date = str_to_datetime(f'{year}-{month}-01')
    if end_date is None:
        end_date = str_to_datetime(f'{year}-{month}-{days_in_month}')
        
    # Ensure that input to fn was valid
    assert(start_date.month == month), 'Start date not in same month!'
    assert(end_date.month == month), 'End date not in same month!'
    # Determine overlap from dates (inclusive)
    days_overlap = (end_date - start_date).days + 1
    # Return fraction
    return days_overlap / days_in_month
    
def boundary_to_coords(bounds):
    min_coords = (bounds.get_lat_min(), bounds.get_long_min())
    max_coords = (bounds.get_lat_max(), bounds.get_long_max())
    return (min_coords, max_coords)
    
def str_to_datetime(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d')

def date_range(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def round_to_sigfig(x, sigfig=5):
    """
    Rounds numbers to some number of significant figures

    Args:
        x (float or np.array): Value(s) to round to sig figs
        sigfig (int): Number of significant figures desired

    Returns:
        np.array:
            Values rounded to the desired number of significant figures
    """
    # Save original type of data so can be returned as input
    orig_type = type(x)
    if orig_type not in [list, float, int, np.ndarray]:
        raise ValueError(f'Cannot round {type(x)} to sig figs!')
    
    # Cast as array if not initially, so that later processes all act as expected
    if orig_type in [int, float]:
        x = [x]
    x = np.array(x)
    # Create a mask disabling any values of inf or zero being passed to log10
    loggable_idxs  = ([x!=0] & np.isfinite(x))[0]
    # Determine number of decimal places to round each number to
    # np.abs because can't find log of negative number
    # np.log10 to get position of most significant digit
    #   where x is finite and non-zero, avoiding overflow from log10
    #   out = 0, setting default value where x=0 or inf
    # np.floor to round to position of most significant digit
    # np.array.astype(int) to enable np.around to work later
    dec_pl = sigfig - np.floor(np.log10(np.abs(x), 
                                        where = loggable_idxs,
                                        out   = np.zeros_like(x))
                               ).astype(int) - 1
    # Round to sig figs
    rounded = np.array(
                    [np.around(x[i], decimals=dec_pl[i]) 
                    for i in range(len(x))]
                )
    # Return as single value if input that way
    if orig_type in [int, float]:
        return rounded.item()
    # Return as python list
    elif orig_type == list:
        return rounded.tolist()
    # Otherwise, return np.array
    else:
        return rounded


# GRF functions
def fftind(size):
    '''
    Creates a numpy array of shifted Fourier coordinates.
    
    Args:
        size (int):
            The size of the coordinate array to create
    
    Returns:
        np.array:
            Numpy array of shifted Fourier coordinates (k_x, k_y). 
            Has shape (2, size, size), with:\n
            array[0,:,:] = k_x components\n
            array[1,:,:] = k_y components
    '''
    # Create array
    k_ind = np.mgrid[:size, :size] - int( (size + 1)/2 )
    # Fourier shift
    k_ind = fftshift(k_ind)
    return( k_ind )

def gaussian_random_field(size, alpha):
            '''
            Creates a gaussian random field with normal (circular) distribution
            Code from https://github.com/bsciolla/gaussian-random-fields/blob/master/gaussian_random_fields.py
            
            Args:
                size (int):
                   Default = 512; 
                   The number of datapoints created per axis in the GRF
                alpha (float):
                    Default = 3.0;
                    The power of the power-law momentum distribution
            
            Returns:
                np.array:
                    2D Array of datapoints, shape (size, size)
            '''
                
            # Defines momentum indices
            k_idx = fftind(size)

            # Defines the amplitude as a power law 1/|k|^(alpha/2)
            amplitude = np.power( k_idx[0]**2 + k_idx[1]**2 + 1e-10, -alpha/4.0 )
            amplitude[0,0] = 0
            
            # Draws a complex gaussian random noise with normal
            # (circular) distribution
            noise = np.random.normal(size = (size, size)) \
                + 1j * np.random.normal(size = (size, size))
            
            # To real space
            grf = np.fft.ifft2(noise * amplitude).real
            
            # Normalise the GRF:
            grf = grf - np.min(grf)
            grf = grf/(np.max(grf)-np.min(grf))
                
            return grf

def memory_trace(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start(20)
        res = func(*args, **kwargs)
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('traceback')

        stat = top_stats[0]
        logging.info("{} memory blocks: {.1f} KiB".
                     format(stat.count, stat.size / 1024))
        logging.info("\n".join(stat.traceback.format()))
        return res
    return wrapper


def timed_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        logging.info("Timed call to {} took {:02f} seconds".
                     format(func.__name__, end - start))
        return res
    return wrapper


# CLI utilities
def setup_logging(func,
                  log_format="[%(asctime)-17s :%(levelname)-8s] - %(message)s"):
    """Wraps a CLI endpoint and sets up logging for it

    This is probably not the smoothest implementation, but it's an educational
    one for people who aren't aware of decorators and how they're implemented.
    In addition, it supports a nice pattern for CLI endpoints

    TODO: start handling level configuration from logging yaml config

    :param func:
    :param log_format:
    :return:
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        parsed_args = func(*args, **kwargs)
        level = logging.INFO

        if hasattr(parsed_args, "verbose") and parsed_args.verbose:
            level = logging.DEBUG

        logging.basicConfig(
            level=level,
            format=log_format,
            datefmt="%d-%m-%y %T",
        )

        logging.getLogger("cdsapi").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("tensorflow").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        return parsed_args
    return wrapper


if __name__ == '__main__':
    aa = str_to_datetime('2020-03-01')
    ab = str_to_datetime('2020-03-15')
    
    ba = str_to_datetime('2020-02-01')
    bb = str_to_datetime('2020-04-01')
    
    ax = date_range(aa, ab)
    bx = date_range(ba, bb)
    
    print(frac_of_month(2020, 2, start_date='2020-02-29'))
    # print(time_overlap(ax, bx))
    