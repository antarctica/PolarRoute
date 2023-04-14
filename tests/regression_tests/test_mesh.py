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
    './example_meshes/Enviromental_Meshes/create_mesh.output2013_4_80.json',
    './example_meshes/Enviromental_Meshes/create_mesh.output2016_6_80.json',
    './example_meshes/Enviromental_Meshes/create_mesh.output2019_6_80.json'
]

TEST_ABSTRACT_MESHES = [
    './example_meshes/Abstract_Environmental_Meshes/vgrad_n201_vT_mesh.json',
    './example_meshes/Abstract_Environmental_Meshes/hgrad_n201_vF_mesh.json',
    './example_meshes/Abstract_Environmental_Meshes/checkerboard_n201_gw2.5_gh2.5_mesh.json',
    './example_meshes/Abstract_Environmental_Meshes/checkerboard_n201_gw5_gh2.5_mesh.json',
    './example_meshes/Abstract_Environmental_Meshes/checkerboard_n201_gw6_gh3_mesh.json',
    './example_meshes/Abstract_Environmental_Meshes/circle_n201_r2_cy-62.5_cx-60.0_mesh.json',
    './example_meshes/Abstract_Environmental_Meshes/cornercirclesplit_n201_r3_cy-65_cx-70_mesh.json',
    './example_meshes/Abstract_Environmental_Meshes/cornercirclenosplit_n201_r3_cy-65_cx-70_mesh.json'
]

@pytest.fixture(scope='session', autouse=False, params=TEST_ENV_MESHES)
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
