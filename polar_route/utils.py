"""
    Miscellaneous utility functions that may be of use throughout PolarRoute
"""

import logging
import time
import tracemalloc
import json
import numpy as np
import pandas as pd
import geopandas as gpd

from datetime import datetime, timedelta
from functools import wraps
from calendar import monthrange
from scipy.fftpack import fftshift


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
    return min_coords, max_coords


def str_to_datetime(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d')


def date_range(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def convert_decimal_days(decimal_days, mins=False):
    """
        Convert decimal days to more readable Days, Hours and (optionally) Minutes
        Args:
            decimal_days (float): Number of days as a decimal
            mins (bool): Determines whether to return minutes or decimal hours
        Returns:
            new_time (str): The time in the new format
    """
    frac_d, days = np.modf(decimal_days)
    hours = frac_d * 24.0

    if mins:
        frac_h, hours = np.modf(hours)
        minutes = round(frac_h * 60.0)
        if days:
            if round(days) == 1:
                new_time = f"{round(days)} day {round(hours)} hours {minutes} minutes"
            else:
                new_time = f"{round(days)} days {round(hours)} hours {minutes} minutes"
        elif hours:
            new_time = f"{round(hours)} hours {minutes} minutes"
        else:
            new_time = f"{minutes} minutes"
    else:
        hours = round(hours, 2)
        if days:
            if round(days) == 1:
                new_time = f"{round(days)} day {hours} hours"
            else:
                new_time = f"{round(days)} days {hours} hours"
        else:
            new_time = f"{hours} hours"

    return new_time


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
    if orig_type not in [list, float, int, np.ndarray, np.float64]:
        raise ValueError(f'Cannot round {type(x)} to sig figs!')
    
    # Cast as array if not initially, so that later processes all act as expected
    if orig_type in [int, float, np.float64]:
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


def divergence(flow):
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dx = np.gradient(Fx, axis=0)
    dFy_dy = np.gradient(Fy, axis=1)
    return dFx_dx + dFy_dy


def curl(flow):
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dy = np.gradient(Fx, axis=1)
    dFy_dx = np.gradient(Fy, axis=0)
    curl = dFy_dx - dFx_dy
    return curl


# GRF functions
def fftind(size):
    """
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
    """
    # Create array
    k_ind = np.mgrid[:size, :size] - int( (size + 1)/2 )
    # Fourier shift
    k_ind = fftshift(k_ind)
    return k_ind


def gaussian_random_field(size, alpha):
    """
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
    """
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


def json_str(input_json):
    """
    Normalise input for classes and functions that take in json, return dict whether a dict or path is given as input.
    Args:
        input_json (dict or str):

    Returns:
        output_json (dict)

    """
    if type(input_json) is dict:
        # If dict, assume it's the json content
        output_json = input_json
    elif type(input_json) is str:
        # If str, assume filename
        with open(input_json, 'r') as f:
            output_json = json.load(f)
    else:
        # Otherwise, can't deal with it
        raise TypeError(f"Expected 'str' or 'dict', instead got '{type(input_json)}'")

    return output_json


def pandas_dataframe_str(input_df):
    """
    Normalise input for classes and functions that take in pandas dataframes, return a df whether a df or path is given
    as input.
    Args:
        input_df (pd.DataFrame or str):

    Returns:
        output_df (pd.DataFrame)
    """
    if (type(input_df) is dict) or (type(input_df) is pd.core.frame.DataFrame):
        output_df = input_df
    elif type(input_df) is str:
        try:
            output_df = pd.read_csv(input_df)
        except:
            raise Exception("Unable to load '{}', please check path name".format(input_df))

    return output_df


def case_from_angle(start, end):
        """
            Determine the direction of travel between two points in the same cell and return the associated case

            Args:
                start (list): the coordinates of the start point within the cell
                end (list):  the coordinates of the end point within the cell

            Returns:
                case (int): the case to use to select variable values from a list
        """

        direct_vec = [end[0]-start[0], end[1]-start[1]]
        direct_ang = np.degrees(np.arctan2(direct_vec[0], direct_vec[1]))

        case = None

        if -22.5 <= direct_ang < 22.5:
            case = -4
        elif 22.5 <= direct_ang < 67.5:
            case = 1
        elif 67.5 <= direct_ang < 112.5:
            case = 2
        elif 112.5 <= direct_ang < 157.5:
            case = 3
        elif 157.5 <= abs(direct_ang) <= 180:
            case = 4
        elif -67.5 <= direct_ang < -22.5:
            case = -3
        elif -112.5 <= direct_ang < -67.5:
            case = -2
        elif -157.5 <= direct_ang < -112.5:
            case = -1

        return case


def unit_time(val, unit):
    """
        Converting time from seconds to the given units

        Args:
            val (float): input time in s
            unit (str): the unit to convert to
        Returns:
            val (float) - output time in the converted units
    """
    if unit == 'days':
        return val/(60*60*24)
    elif unit == 'hr':
        return val/(60*60)
    elif unit == 'min':
        return val/60
    elif unit == 's':
        return val


def unit_speed(val, unit):
    """
        Converting speed from m/s to the given units

        Args:
            val (float): input speed in m/s
            unit (str): the unit to convert to
        Returns:
            val (float) - output speed in unit type 'unit'

    """
    if val is not None:
        if unit == 'km/hr':
            val = val*(1000/(60*60))
        if unit == 'knots':
            val = (val*0.51)
        return val
    else:
        return None


def gpx_route_import(f_name):
    """
        Function to import a route in gpx format and convert it to geojson format

        Args:
            f_name: Filename of gpx route file

        Returns:
            geojson: Route in geojson format
    """
    gdf_r = gpd.read_file(f_name, layer="routes")
    gdf_p = gpd.read_file(f_name, layer="route_points")

    # Drop empty fields from original gpx file
    gdf_r = gdf_r.dropna(how='all', axis=1)
    # Convert route to geojson linestring
    geojson = json.loads(gdf_r.to_json())

    # Extract start and end waypoints and add to geojson properties
    geojson['features'][0]['properties']['from'] = gdf_p['name'].iloc[0]
    geojson['features'][0]['properties']['to'] = gdf_p['name'].iloc[-1]

    return geojson


def to_chart_track_csv(route):
    """
        Output a route in Chart Track csv format
    """

    def dd_to_dmm(dd, axis):
        """
            Converts decimal degrees to dmm formatted string
        """
        if dd >= 0:
            degs, mins = divmod(dd, 1)
            cardinal_dir = 'E' if axis == 'long' else 'N'
        else:
            degs, mins = divmod(-dd, 1)
            cardinal_dir = 'W' if axis == 'long' else 'S'
        return f"{int(degs)}-{60 * mins:.3f}'{cardinal_dir}"

    def get_bearing(lat1, long1, lat2, long2):
        """
            Calculate bearing of travel from lat/long pairs
        """
        dlon = long2 - long1
        x = np.cos(np.radians(lat2)) * np.sin(np.radians(dlon))
        y = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - \
            np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(dlon))
        bearing = np.arctan2(x, y)
        return np.degrees(bearing)

    # For path, generate a csv string (typo to match output from Chart Track)
    header = f"Route Name:,{route['properties']['from']}_{route['properties']['to']}\n" + \
             "Way Point,Position,,Radius,Reach,ROT,XTD,SPD,RL/GC,Leg,Disance(NM),,ETA\n" + \
             "ID,LAT,LON,,,,,,,,To WPT,TOTAL\n"
    # Turn coords into DMM format
    coords = np.array(route['geometry']['coordinates'])
    long = [dd_to_dmm(long, 'long') for long in coords[:, 0]]
    lat = [dd_to_dmm(lat, 'lat') for lat in coords[:, 1]]
    # Distance column
    cumulative_distance = np.array(route['properties']['distance']) * 0.000539957  # In nautical miles
    distance = np.diff(cumulative_distance)
    # Waypoint names
    wps = [f'WP{i}' for i in range(len(cumulative_distance))]
    leg = get_bearing(coords[:, 1][:-1], coords[:, 0][:-1],
                      coords[:, 1][1:], coords[:, 0][1:]) % 360
    eta = route['properties']['traveltime']
    # Construct table with information
    path_df = pd.DataFrame({'ID': wps,
                            'LAT': lat,
                            'LON': long,
                            'Radius': '',
                            'Reach': '',
                            'ROT': '',
                            'XTD': '',
                            'SPD': '',
                            'RL/GC': 'RL',
                            'Leg': np.concatenate((leg, [np.nan])),
                            'To WPT': np.concatenate(([np.nan], distance)),
                            'TOTAL': cumulative_distance})
    # Combine to one string and add to list of strs
    csv_str = header + path_df.to_csv()
    return csv_str


def extract_geojson_routes(mesh):
    """
    
    Extract routes in a precomputed mesh in GEOJSON format

    Args:
        mesh (dict): Precomputed mesh JSON with routes embedded
        
    Returns:
        list: 
            List of all routes found in mesh. If no routes found, returns 
            empty list
    """
    
    logging.info("Extracting routes in geojson format")
    
    # Extract the computed routes from the mesh
    if "paths" in mesh.keys():
        routes = mesh["paths"]["features"]
    else:
        routes = []

    # Reformat every route to geojson format and append to list
    geojson_routes = []
    for route in routes:
        # Using FeatureCollection to keep consistent formatting with 
        # multi-route geojsons
        geojson_route = {"type": "FeatureCollection",
                            "features": [route]}
        
        geojson_routes.append(geojson_route)

    # Return list of individual geojson routes
    return geojson_routes
