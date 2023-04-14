"""
    Regression testing package to ensure consistent functionality in development
    of the PolarRoute python package.
"""

import json
import pytest

# from polar_route.mesh import Mesh
from polar_route.vessel_performance.vessel_performance_modeller import VesselPerformanceModeller

# Import tests, which are automatically run
from .mesh_test_functions import test_mesh_cellbox_attributes
from .mesh_test_functions import test_mesh_cellbox_count
from .mesh_test_functions import test_mesh_cellbox_ids
from .mesh_test_functions import test_mesh_cellbox_values
from .mesh_test_functions import test_mesh_neighbour_graph_count
from .mesh_test_functions import test_mesh_neighbour_graph_ids
from .mesh_test_functions import test_mesh_neighbour_graph_values

#File locations of all vessel performance meshes to be recalculated for regression testing.
INPUT_MESHES = [
    './example_meshes/env_meshes/mesh_2013.json',
    './example_meshes/env_meshes/mesh_2017.json',
    './example_meshes/env_meshes/mesh_2019.json'
]

OUTPUT_MESHES = [
    './example_meshes/vessel_meshes/vessel_2013.json',
    './example_meshes/vessel_meshes/vessel_2017.json',
    './example_meshes/vessel_meshes/vessel_2019.json'
]

@pytest.fixture(scope='session', autouse=False, params=zip(INPUT_MESHES, OUTPUT_MESHES))
def mesh_pair(request):

    input_mesh_file = request.param[0]
    output_mesh_file = request.param[1]

    with open(output_mesh_file, 'r') as fp:
        old_mesh = json.load(fp)

    with open(input_mesh_file, 'r') as fp:
        input_mesh = json.load(fp)
    
    vessel_config = old_mesh['config']['Vessel']
    new_mesh = calculate_mesh(input_mesh, vessel_config)


    
    return [old_mesh, new_mesh]

def calculate_mesh(mesh_json, vessel_config):

    new_mesh = VesselPerformanceModeller(mesh_json, vessel_config)
    new_mesh.model_accessibility()
    new_mesh.model_performance()

    return new_mesh.to_json()