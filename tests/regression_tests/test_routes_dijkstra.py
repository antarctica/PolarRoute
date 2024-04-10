import json
import pytest
import time

from polar_route import __version__ as pr_version
from polar_route import RoutePlanner


from .route_test_functions import extract_waypoints
from .route_test_functions import extract_route_info

# Import tests, which are automatically run
from .route_test_functions import test_route_coordinates
from .route_test_functions import test_waypoint_names
from .route_test_functions import test_time
from .route_test_functions import test_fuel
from .route_test_functions import test_cell_indices
from .route_test_functions import test_cases

import logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# location of test files to be recalcaulted for regression testing
TEST_ROUTES = [
    './example_routes/dijkstra/fuel/gaussian_random_field.json',
    './example_routes/dijkstra/fuel/checkerboard.json',
    './example_routes/dijkstra/fuel/great_circle_forward.json',
    './example_routes/dijkstra/fuel/great_circle_reverse.json',
    './example_routes/dijkstra/time/gaussian_random_field.json',
    './example_routes/dijkstra/time/checkerboard.json',
    './example_routes/dijkstra/time/great_circle_forward.json',
    './example_routes/dijkstra/time/great_circle_reverse.json',
    './example_routes/dijkstra/crossing_point/horizontal/horizontal_0lat.json',
    './example_routes/dijkstra/crossing_point/horizontal/horizontal_20lat_s.json',
    './example_routes/dijkstra/crossing_point/horizontal/horizontal_20lat_n.json',
    './example_routes/dijkstra/crossing_point/horizontal/horizontal_40lat_s.json',
    './example_routes/dijkstra/crossing_point/horizontal/horizontal_40lat_n.json',
    './example_routes/dijkstra/crossing_point/horizontal/horizontal_60lat_s.json',
    './example_routes/dijkstra/crossing_point/horizontal/horizontal_60lat_n.json',
    './example_routes/dijkstra/crossing_point/horizontal/horizontal_80lat_s.json',
    './example_routes/dijkstra/crossing_point/horizontal/horizontal_80lat_n.json',
    './example_routes/dijkstra/crossing_point/diagonal/diagonal_0lat.json',
    './example_routes/dijkstra/crossing_point/vertical/vertical_20lat_s.json',
    './example_routes/dijkstra/crossing_point/vertical/vertical_20lat_n.json',
    './example_routes/dijkstra/crossing_point/vertical/vertical_40lat_s.json',
    './example_routes/dijkstra/crossing_point/vertical/vertical_40lat_n.json',
    './example_routes/dijkstra/crossing_point/vertical/vertical_60lat_s.json',
    './example_routes/dijkstra/crossing_point/vertical/vertical_60lat_n.json',
    './example_routes/dijkstra/crossing_point/vertical/vertical_80lat_s.json',
    './example_routes/dijkstra/crossing_point/vertical/vertical_80lat_n.json',
    './example_routes/dijkstra/crossing_point/horizontal/horizontal_0lat_split.json'
]

def setup_module():
    LOGGER.info(f'PolarRoute version: {pr_version}')

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

    LOGGER.info(f'Test File: {request.param}')

    # Load reference JSON
    with open(request.param, 'r') as fp:
        old_route = json.load(fp)
    route_info = extract_route_info(old_route)
    # Create new json (cast old to dict to create copy to avoid modifying)
    new_route = calculate_dijkstra_route(route_info, dict(old_route))

    return [old_route, new_route]

# Generating new outputs
def calculate_dijkstra_route(config, mesh):
    """
    Calculates the fuel-optimised route, with dijkstra but no smoothing

    Args:
        route_filename (str): Filename of regression test route

    Returns:
        json: New route output
    """
    start = time.perf_counter()

    # Initial set up
    waypoints   = extract_waypoints(mesh)
    
    # Calculate dijskstra route
    rp = RoutePlanner(mesh, config, waypoints)
    rp.compute_routes()
    
    # Generate json to compare to old output
    new_route = rp.to_json()

    end = time.perf_counter()
    LOGGER.info(f'Route calculated in {end - start} seconds')

    return new_route