"""
    Regression testing package to ensure consistance functionality in development
    of the PolarRoute python package.
"""
import numpy as np
import json
import pytest

from polar_route.mesh import Mesh
from polar_route.vessel_performance import VesselPerformance

#File locations of all meshes to be recaculated for regression testing.
TEST_MESHES = [
    './example_meshes/smallmesh_sl5.json',
    './example_meshes/smallmesh_sl2.json',
    './example_meshes/WeddellSea.json'
]

@pytest.fixture(scope='session', autouse=True, params=TEST_MESHES)
def mesh_pair(request):
    """
        Reconstructs all meshes listed TEST_MESHES.

        Returns: 
            mesh_pair (list):
                mesh_pair[0] -> Regression mesh
                mesh_pair[1] -> Newly calculated mesh.
    """
    with open(request.param, 'r') as f:
        regression_mesh = json.load(f)

    config = regression_mesh['config']
    new_mesh = Mesh(config)

    new_mesh = new_mesh.to_json()

    vp = VesselPerformance(new_mesh)
    new_mesh = vp.to_json()

    return [regression_mesh, new_mesh]


def test_cellbox_count(mesh_pair):
    """
        Test if two provided meshes contain the same number of cellboxes

        Args:
            mesh_pair (list): A pair of meshes to compare in regression testing:
                mesh_pair[0] -> regression_mesh (dict): A verfided correct mesh for use in regression
                testing
                mesh_pair[1] -> new_mesh (dict): The currently calculated mesh from PolarRoute

        Throws:
            Fails if the number of cellboxes in regression_mesh and new_mesh are
            not equal
    """
    regression_mesh = mesh_pair[0]['cellboxes']
    new_mesh = mesh_pair[1]['cellboxes']

    cellbox_count_a = len(regression_mesh)
    cellbox_count_b = len(new_mesh)

    assert(cellbox_count_a == cellbox_count_b), \
        f"Incorrect number of cellboxes in new mesh. Expected :{cellbox_count_a}, got: {cellbox_count_b}"

def test_cellbox_ids(mesh_pair):
    """
        Test if two provided meshes contain cellboxes with the same IDs

        Args:
            mesh_pair (list): A pair of meshes to compare in regression testing:
                mesh_pair[0] -> regression_mesh (dict): A verfided correct mesh for use in regression
                testing
                mesh_pair[1] -> new_mesh (dict): The currently calculated mesh from PolarRoute

        Throws:
            Fails if any cellbox exists in regression_mesh that or not in new_mesh,
            or any cellbox exsits in new_mesh that is not in regression_mesh
    """
    regression_mesh = mesh_pair[0]['cellboxes']
    new_mesh = mesh_pair[1]['cellboxes']

    indxed_a = dict()
    for cellbox in regression_mesh:
        indxed_a[cellbox['id']] = cellbox

    indxed_b = dict()
    for cellbox in new_mesh:
        indxed_b[cellbox['id']] = cellbox

    regression_mesh_ids = set(indxed_a.keys())
    new_mesh_ids = set(indxed_b.keys())

    missing_a_ids = list(new_mesh_ids - regression_mesh_ids)
    missing_b_ids = list(regression_mesh_ids - new_mesh_ids)

    assert(indxed_a.keys()  == indxed_b.keys()), \
        f"Mismatch in cellbox IDs. ID's {missing_a_ids} have appeared in the new mesh. ID's {missing_b_ids} are missing from the new mesh"

def test_cellbox_values(mesh_pair):
    """
        Tests if the values in of all attributes in each cellbox and the
        same in both provided meshes.

       Args:
            mesh_pair (list): A pair of meshes to compare in regression testing:
                mesh_pair[0] -> regression_mesh (dict): A verfided correct mesh for use in regression
                testing
                mesh_pair[1] -> new_mesh (dict): The currently calculated mesh from PolarRoute

        Throws:
            Fails if any values of any attributes differ between regression_mesh
            and new_mesh
    """
    regression_mesh = mesh_pair[0]['cellboxes']
    new_mesh = mesh_pair[1]['cellboxes']

    indxed_a = dict()
    for cellbox in regression_mesh:
        indxed_a[cellbox['id']] = cellbox

    indxed_b = dict()
    for cellbox in new_mesh:
        indxed_b[cellbox['id']] = cellbox

    mismatch_cellboxes = dict()
    for cellbox_a in indxed_a.values():
        # Prevent crashing if cellbox not in new mesh
        # This error will be detected by 'test_cellbox_ids'
        if cellbox_a['id'] in indxed_b.keys():
            cellbox_b = indxed_b[cellbox_a['id']]

            mismatch_values = []
            for key in cellbox_a.keys():
                # To prevent crashing if cellboxes have different attributes
                # This error will be detected by the 'test_cellbox_attributes' test
                if key in cellbox_b.keys():
                    value_a = cellbox_a[key]
                    value_b = cellbox_b[key]

                    if not(value_a == value_b) and not(np.isnan(value_a) or np.isnan(value_b)):
                        mismatch_values.append(key)
                        mismatch_cellboxes[cellbox_a['id']] = mismatch_values

    assert(len(mismatch_cellboxes) == 0) , \
        f"Values in <{len(mismatch_cellboxes.keys())}> cellboxes in the new mesh have changed. The changes cellboxes are: {mismatch_cellboxes}"

def test_cellbox_attributes(mesh_pair):
    """
        Tests if the attributes of cellboxes in regression_mesh and the same as
        attributes of cellboxes in new_mesh

        Note:
            This assumes that every cellbox in mesh has the same amount
            of attributes, so only compares the attributes of the first
            two cellboxes in the mesh

        Args:
            mesh_pair (list): A pair of meshes to compare in regression testing:
                mesh_pair[0] -> regression_mesh (dict): A verfided correct mesh for use in regression
                testing
                mesh_pair[1] -> new_mesh (dict): The currently calculated mesh from PolarRoute

        Throws:
            Fails if the cellboxes in the provided meshes contain different
            attributes
    """
    regression_mesh = mesh_pair[0]['cellboxes']
    new_mesh = mesh_pair[1]['cellboxes']

    regression_regression_meshttributes = set(regression_mesh[0].keys())
    new_mesh_attributes = set(new_mesh[0].keys())

    missing_a_attributes = list(new_mesh_attributes - regression_regression_meshttributes)
    missing_b_attributes = list(regression_regression_meshttributes - new_mesh_attributes)

    assert(regression_regression_meshttributes == new_mesh_attributes), \
        f"Mismatch in cellbox attributes. Attributes {missing_a_attributes} have appeared in the new mesh. Attributes {missing_b_attributes} are missing in the new mesh"

def test_neighbour_graph_count(mesh_pair):
    """
        Tests that the neighbour_graph in the regression mesh and the newly calculated mesh have the 
        same number of nodes.

        Args:
            mesh_pair (list): A pair of meshes to compare in regression testing:
                mesh_pair[0] -> regression_mesh (dict): A verfided correct mesh for use in regression
                testing
                mesh_pair[1] -> new_mesh (dict): The currently calculated mesh from PolarRoute

    """
    regression_graph = mesh_pair[0]['neighbour_graph']
    new_graph = mesh_pair[1]['neighbour_graph']

    regression_graph_count = len(regression_graph.keys())
    new_graph_count = len(new_graph.keys())

    assert(regression_graph_count == new_graph_count), \
        f"Incorrect number of nodes in neighbour graph. Expected: <{regression_graph_count}> nodes, got: <{new_graph_count}> nodes."

def test_neighbour_graph_ids(mesh_pair):
    """
        Tests that the neighbour_graph in the regression mesh and the newly calculated mesh contain
        all the same node IDs.

        Args:
            mesh_pair (list): A pair of meshes to compare in regression testing:
                mesh_pair[0] -> regression_mesh (dict): A verfided correct mesh for use in regression
                testing
                mesh_pair[1] -> new_mesh (dict): The currently calculated mesh from PolarRoute
    """
    regression_graph = mesh_pair[0]['neighbour_graph']
    new_graph = mesh_pair[1]['neighbour_graph']

    regression_graph_ids = set(regression_graph.keys())
    new_graph_ids = set(new_graph.keys())

    missing_a_keys = list(new_graph_ids - regression_graph_ids)
    missing_b_keys = list(regression_graph_ids - new_graph_ids)

    assert(regression_graph_ids == new_graph_ids) , \
        f"Mismatch in neighbour graph nodes. <{len(missing_a_keys)}> nodes  have appeared in the new neighbour graph. <{len(missing_b_keys)}> nodes  are missing from the new neighbour graph."

def test_neighbour_graph_values(mesh_pair):
    """
        Tests that each node in the neighbour_graph of the regression mesh and the newly calculated
        mesh have the same neighbours.

        Args:
            mesh_pair (list): A pair of meshes to compare in regression testing:
                mesh_pair[0] -> regression_mesh (dict): A verfided correct mesh for use in regression
                testing
                mesh_pair[1] -> new_mesh (dict): The currently calculated mesh from PolarRoute

    """
    regression_graph = mesh_pair[0]['neighbour_graph']
    new_graph = mesh_pair[1]['neighbour_graph']

    mismatch_neighbors = dict()

    for node in regression_graph.keys():
        # Prevent crashing if node not found. 
        # This will be detected by 'test_neighbour_graph_ids'.
        if node in new_graph.keys():
            neighbours_a = regression_graph[node]
            neighbours_b = new_graph[node]

            if not neighbours_b == neighbours_a:
                mismatch_neighbors[node] = neighbours_b

    assert(len(mismatch_neighbors) == 0), \
        f"Mismatch in neighbour graph neighbours. <{len(mismatch_neighbors.keys())}> nodes have changed in new mesh."
