"""
    Regression testing package to ensure consistent functionality in development
    of the PolarRoute python package.
"""

import json
import pytest

# from polar_route.mesh import Mesh
from polar_route.mesh_generation.mesh_builder import MeshBuilder

# Import tests, which are automatically run
from .mesh_test_functions import test_mesh_cellbox_attributes
from .mesh_test_functions import test_mesh_cellbox_count
from .mesh_test_functions import test_mesh_cellbox_ids
from .mesh_test_functions import test_mesh_cellbox_values
from .mesh_test_functions import test_mesh_neighbour_graph_count
from .mesh_test_functions import test_mesh_neighbour_graph_ids
from .mesh_test_functions import test_mesh_neighbour_graph_values

#File locations of all environmental meshes to be recalculated for regression testing.
TEST_ENV_MESHES = [
    './example_meshes/env_meshes/mesh_2013.json',
    './example_meshes/env_meshes/mesh_2017.json',
    './example_meshes/env_meshes/mesh_2019.json'
]

TEST_ABSTRACT_MESHES = [
    './example_meshes/abstract_env_meshes/vgrad.json',
    './example_meshes/abstract_env_meshes/hgrad.json',
    './example_meshes/abstract_env_meshes/checkerboard_1.json',
    './example_meshes/abstract_env_meshes/checkerboard_2.json',
    './example_meshes/abstract_env_meshes/checkerboard_3.json',
    './example_meshes/abstract_env_meshes/circle.json',
    './example_meshes/abstract_env_meshes/circle_quadrant_split.json',
    './example_meshes/abstract_env_meshes/circle_quadrant_nosplit.json'
]

@pytest.fixture(scope='session', autouse=False, params=TEST_ENV_MESHES + TEST_ABSTRACT_MESHES)
def mesh_pair(request):
    with open(request.param, 'r') as fp:
        old_mesh = json.load(fp)
    
    new_mesh = calculate_mesh(old_mesh)

    return [old_mesh, new_mesh]

def calculate_mesh(mesh_json):
    config = mesh_json['config']
    mesh_builder = MeshBuilder(config)
    new_mesh = mesh_builder.build_environmental_mesh()

    return new_mesh.to_json()
