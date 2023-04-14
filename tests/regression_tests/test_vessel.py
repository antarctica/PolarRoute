"""
    Regression testing package to ensure consistent functionality in development
    of the PolarRoute python package.
"""

import json
import pytest

# from polar_route.mesh import Mesh
from polar_route.mesh_generation.mesh_builder import MeshBuilder
from polar_route.mesh_generation.environment_mesh import EnvironmentMesh
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
TEST_VESSEL_MESHES = [
    './example_meshes/vessel_meshes/vessel_2013.json',
    './example_meshes/vessel_meshes/vessel_2017.json',
    './example_meshes/vessel_meshes/vessel_2019.json'
]

@pytest.fixture(scope='session', autouse=False, params=TEST_VESSEL_MESHES)
def mesh_pair(request):
    with open(request.param, 'r') as fp:
        old_mesh = json.load(fp)
    
    new_mesh = calculate_mesh(old_mesh)

    return [old_mesh, new_mesh]

def calculate_mesh(mesh_json):
    config = mesh_json['config']
    mesh_builder = MeshBuilder(config)
    env_mesh = mesh_builder.build_environmental_mesh()

    env_mesh = env_mesh.to_json()

    vessel_config = mesh_json['config']['Vessel']
    # env_mesh = EnvironmentMesh.load_from_json(mesh_json).to_json()
    new_mesh = VesselPerformanceModeller(env_mesh, vessel_config)
    new_mesh.model_accessibility()
    new_mesh.model_performance()

    return new_mesh.to_json()