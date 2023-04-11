import numpy as np
import pandas as pd
import json
import pytest

from polar_route.route_planner import RoutePlanner


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