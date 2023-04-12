import numpy as np
import pandas as pd
import json
import pytest

from polar_route.route_planner import RoutePlanner
from polar_route.utils import round_to_sigfig

SIG_FIG_TOLERANCE = 5

ROUTE_INFO_FUEL = './example_paths/route_info.fuel.json'
ROUTE_INFO_TIME = './example_paths/route_info.time.json'

TEST_SMOOTHED_PATHS = [
    './example_paths/smoothed/gaussian_random_field/',
    './example_paths/smoothed/checkerboard/',
    './example_paths/smoothed/great_circle_1/'
    './example_paths/smoothed/great_circle_2/',
    './example_paths/smoothed/real/'
]

TEST_DIJKSTRA_PATHS = [
    './example_paths/dijsktra/gaussian_random_field/',
    './example_paths/dijsktra/checkerboard/',
    './example_paths/dijsktra/great_circle_1/'
    './example_paths/dijsktra/great_circle_2/',
    './example_paths/dijsktra/real/'
]



@pytest.fixture
def fuel(request):
    return ROUTE_INFO_FUEL, request.param

def time(request):
    return ROUTE_INFO_TIME, request.param


@pytest.fixture(scope='session', autouse=False, params=TEST_DIJKSTRA_PATHS)
def dijkstra_fuel_path_pair(request):
    return calculate_dijkstra_fuel_path(request.param)

def calculate_dijkstra_fuel_path(path_location):
    """
    Calculates dijkstra route optimising for fuel
    """
    # Initial set up
    config      = ROUTE_INFO_FUEL
    mesh        = path_location + 'optimise_routes.output.json'
    waypoints   = path_location + 'waypoints.csv'
    
    # Calculate dijskstra path
    rp = RoutePlanner(mesh, config, waypoints)
    rp.compute_routes()
    
    # Generate json to compare to old output
    new_path = rp.to_json()
    # Read in old output for comparison
    with open(mesh, 'r') as f:
        regression_path = json.load(f)

    return [regression_path, new_path]

def calculate_dijkstra_time_path(path_location):
    """
    Calculates dijkstra route optimising for fuel
    """
    # Initial set up
    config      = ROUTE_INFO_TIME
    mesh        = path_location + 'optimise_routes.output.json'
    waypoints   = path_location + 'waypoints.csv'
    
    # Calculate dijskstra path
    rp = RoutePlanner(mesh, config, waypoints)
    rp.compute_routes()
    
    # Generate json to compare to old output
    new_path = rp.to_json()
    # Read in old output for comparison
    with open(mesh, 'r') as f:
        regression_path = json.load(f)

    return [regression_path, new_path]

def compare_path_coordinates(path_a, path_b):
    """
    Tests if coordinates of each node are the same between both paths.
    
    Args:
        path_a (json)
        path_b (json)
        
    Raises:
        AssertionError:
            Fails if length of coord arrays is different,
            which implies different nodes along path
        AssertionError:
            Fails if coords of each node are different at any point 
            (beyond sig fig limit)
    """
    coords_a = path_a['geometry']['coordinates']
    coords_b = path_b['geometry']['coordinates']

    len_a = len(coords_a)
    len_b = len(coords_b)
    
    assert(len_a == len_b), \
        f"Number of nodes different! Expected {len_a}, got {len_b}"
    
    rounded_x_a = round_to_sigfig(coords_a[:][0], sigfig=SIG_FIG_TOLERANCE)
    rounded_x_b = round_to_sigfig(coords_b[:][0], sigfig=SIG_FIG_TOLERANCE)
    differences_x = np.nonzero(rounded_x_a - rounded_x_b)[0]

    rounded_y_a = round_to_sigfig(coords_a[:][1], sigfig=SIG_FIG_TOLERANCE)
    rounded_y_b = round_to_sigfig(coords_b[:][1], sigfig=SIG_FIG_TOLERANCE)
    differences_y = np.nonzero(rounded_y_a - rounded_y_b)[0]
    
    mismatch_idxs = set(np.append(differences_x, differences_y))

    assert((rounded_x_a == rounded_x_b) and (rounded_y_a == rounded_y_b)), \
        f"Coordinates of nodes do not match! Indexes {mismatch_idxs} are mismatched"

def compare_waypoint_names(path_a, path_b):
    """
    Tests if waypoint names are the same between both paths.
    
    Args:
        path_a (json)
        path_b (json)
        
    Raises:
        AssertionError:
            Fails if source or destination names don't match 
    """
    from_a = path_a['properties']['from']
    from_b = path_b['properties']['from']
    assert(from_a == from_b), \
        f"Waypoint source names don't match! Expected {from_a}, got {from_b}"
    
    to_a = path_a['properties']['to']
    to_b = path_b['properties']['to']
    assert(to_a == to_b), \
        f"Waypoint destination names don't match! Expected {to_a}, got {to_b}"

def compare_time(path_a, path_b):
    """
    Tests if times to each node are the same between both paths.
    
    Args:
        path_a (json)
        path_b (json)
        
    Raises:
        AssertionError:
            Fails if time to each node is different at any point 
            (beyond sig fig limit)
    """
    times_a = path_a['properties']['traveltime']
    times_b = path_b['properties']['traveltime']

    rounded_a = round_to_sigfig(times_a, sigfig=SIG_FIG_TOLERANCE)
    rounded_b = round_to_sigfig(times_b, sigfig=SIG_FIG_TOLERANCE)

    assert(rounded_a == rounded_b), \
        f"Travel time to nodes different! Max difference of {np.max(np.abs(rounded_a - rounded_b))}"

def compare_fuel(path_a, path_b):
    """
    Tests if fuel to each node are the same between both paths.
    
    Args:
        path_a (json)
        path_b (json)
        
    Raises:
        AssertionError:
            Fails if fuel to each node is different at any point 
            (beyond sig fig limit)
    """
    fuel_a = path_a['properties']['fuel']
    fuel_b = path_b['properties']['fuel']

    rounded_a = round_to_sigfig(fuel_a, sigfig=SIG_FIG_TOLERANCE)
    rounded_b = round_to_sigfig(fuel_b, sigfig=SIG_FIG_TOLERANCE)

    assert(rounded_a == rounded_b), \
        f"Fuel to nodes different! Max difference of {np.max(np.abs(rounded_a - rounded_b))}"

def compare_distance(path_a, path_b):
    """
    Tests if distance to each node are the same between both paths.
    
    Args:
        path_a (json)
        path_b (json)
        
    Raises:
        AssertionError:
            Fails if distance to each node is different at any point 
            (beyond sig fig limit)
    """
    dist_a = path_a['properties']['distance']
    dist_b = path_b['properties']['distance']
  
    rounded_a = round_to_sigfig(dist_a, sigfig=SIG_FIG_TOLERANCE)
    rounded_b = round_to_sigfig(dist_b, sigfig=SIG_FIG_TOLERANCE)

    assert(rounded_a == rounded_b), \
        f"Fuel to nodes different! Max difference of {np.max(np.abs(rounded_a - rounded_b))}"

def compare_speed(path_a, path_b):
    """
    Tests if speed between each node are the same between both paths.
    
    Args:
        path_a (json)
        path_b (json)
        
    Raises:
        AssertionError:
            Fails if speed between each node is different at any point 
            (beyond sig fig limit)
    """
    speed_a = path_a['properties']['speed']
    speed_b = path_b['properties']['speed']
 
    rounded_a = round_to_sigfig(speed_a, sigfig=SIG_FIG_TOLERANCE)
    rounded_b = round_to_sigfig(speed_b, sigfig=SIG_FIG_TOLERANCE)

    assert(rounded_a == rounded_b), \
        f"Fuel to nodes different! Max difference of {np.max(np.abs(rounded_a - rounded_b))}"
    
