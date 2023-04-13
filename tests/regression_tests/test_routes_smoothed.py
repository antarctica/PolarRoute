import json
import pytest

from polar_route.route_planner import RoutePlanner
from .route_methods import extract_waypoints

# Import tests, which are automatically run
from .route_methods import test_fuel_route_coordinates
from .route_methods import test_fuel_waypoint_names
from .route_methods import test_fuel_time
from .route_methods import test_fuel_fuel
from .route_methods import test_fuel_distance
from .route_methods import test_fuel_speed
from .route_methods import test_time_route_coordinates
from .route_methods import test_time_waypoint_names
from .route_methods import test_time_time
from .route_methods import test_time_fuel
from .route_methods import test_time_distance
from .route_methods import test_time_speed

FUEL_ROUTE_INFO = './example_routes/route_info_fuel.json'
TIME_ROUTE_INFO = './example_routes/route_info_time.json'

TEST_FUEL_ROUTES = [
    './example_routes/smoothed/fuel/gaussian_random_field.json',
    './example_routes/smoothed/fuel/checkerboard.json',
    './example_routes/smoothed/fuel/great_circle_forward.json',
    './example_routes/smoothed/fuel/great_circle_reverse.json',
    # './example_routes/smoothed/fuel/real.json'
]

TEST_TIME_ROUTES = [
    './example_routes/smoothed/time/gaussian_random_field.json',
    './example_routes/smoothed/time/checkerboard.json',
    './example_routes/smoothed/time/great_circle_forward.json',
    './example_routes/smoothed/time/great_circle_reverse.json',
    # './example_routes/smoothed/time/real.json'
]

# Pairing old and new outputs
@pytest.fixture(scope='session', autouse=False, params=TEST_FUEL_ROUTES)
def fuel_route_pair(request):
    """
    Creates a pair of JSON objects, one newly generated, one as old reference
    Args:
        request (fixture): 
            fixture object including list of jsons of fuel optimised routes

    Returns:
        list: [reference json, new json]
    """
    # Load reference JSON
    with open(request.param, 'r') as fp:
        old_route = json.load(fp)
    # Create new json (cast old to dict to create copy to avoid modifying)
    new_route = calculate_smoothed_route(FUEL_ROUTE_INFO, dict(old_route))
    return [old_route, new_route]

@pytest.fixture(scope='session', autouse=False, params=TEST_TIME_ROUTES)
def time_route_pair(request):
    """
    Creates a pair of JSON objects, one newly generated, one as old reference
    Args:
        request (fixture): 
            fixture object including list of jsons of time optimised routes

    Returns:
        list: [reference json, new json]
    """
    # Load reference JSON
    with open(request.param, 'r') as fp:
        old_route = json.load(fp)
    # Create new json (cast old to dict to create copy to avoid modifying)

    new_route = calculate_smoothed_route(TIME_ROUTE_INFO, dict(old_route))
    return [old_route, new_route]

# Generating new outputs
def calculate_smoothed_route(route_info, route_json):
    """
    Calculates the fuel-optimised route, with dijkstra but no smoothing

    Args:
        route_filename (str): Filename of regression test route

    Returns:
        json: New route output
    """

    # Initial set up
    config      = route_info
    mesh        = route_json
    waypoints   = extract_waypoints(route_json)
    
    # Calculate dijskstra route
    rp = RoutePlanner(mesh, config, waypoints)
    rp.compute_routes()
    rp.compute_smoothed_routes()
    
    # Generate json to compare to old output
    new_route = rp.to_json()

    return new_route