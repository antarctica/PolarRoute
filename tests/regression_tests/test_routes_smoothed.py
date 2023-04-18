import json
import pytest

from polar_route.route_planner import RoutePlanner
from .route_test_functions import extract_waypoints
from .route_test_functions import extract_route_info

# Import tests, which are automatically run
from .route_test_functions import test_route_coordinates
from .route_test_functions import test_waypoint_names
from .route_test_functions import test_time
from .route_test_functions import test_fuel
from .route_test_functions import test_distance
from .route_test_functions import test_speed

TEST_ROUTES = [
    './example_routes/smoothed/fuel/gaussian_random_field.json',
    './example_routes/smoothed/fuel/checkerboard.json',
    './example_routes/smoothed/fuel/great_circle_forward.json',
    './example_routes/smoothed/fuel/great_circle_reverse.json',
    './example_routes/smoothed/time/gaussian_random_field.json',
    './example_routes/smoothed/time/checkerboard.json',
    './example_routes/smoothed/time/great_circle_forward.json',
    './example_routes/smoothed/time/great_circle_reverse.json',
    # './example_routes/smoothed/time/real.json'
]

# Pairing old and new outputs
@pytest.fixture(scope='session', autouse=False, params=TEST_ROUTES)
def route_pair(request):
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
    route_info = extract_route_info(old_route)
    # Create new json (cast old to dict to create copy to avoid modifying)
    new_route = calculate_smoothed_route(route_info, dict(old_route))

    return [old_route, new_route]

# Generating new outputs
def calculate_smoothed_route(config, mesh):
    """
    Calculates the fuel-optimised route, with dijkstra but no smoothing

    Args:
        route_filename (str): Filename of regression test route

    Returns:
        json: New route output
    """

    # Initial set up
    waypoints   = extract_waypoints(mesh)
    
    # Calculate dijskstra route
    rp = RoutePlanner(mesh, config, waypoints)
    rp.compute_routes()
    rp.compute_smoothed_routes()
    
    # Generate json to compare to old output
    new_route = rp.to_json()

    return new_route