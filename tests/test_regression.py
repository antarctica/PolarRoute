"""
    Regression testing package to ensure consistance functionality in development
    of the PolarRoute python package.
"""
import sys
import traceback
import numpy as np
import json

from polar_route.mesh import Mesh
from polar_route.vessel_performance import VesselPerformance

def test_cellbox_count(mesh_a, mesh_b):
    """
        Test if two provided meshes contain the same number of cellboxes

        Args:
            mesh_a (dict): A verfided correct mesh for use in regression
            testing
            mesh_b (dict): The currently calculated mesh from PolarRoute

        Throws:
            Fails if the number of cellboxes in mesh_a and mesh_b are
            not equal
    """
    cellbox_count_a = len(mesh_a)
    cellbox_count_b = len(mesh_b)

    assert(cellbox_count_a == cellbox_count_b), \
        f"Incorrect number of cellboxes in new mesh. Expected :{cellbox_count_a}, got: {cellbox_count_b}"

def test_cellbox_ids(mesh_a, mesh_b):
    """
        Test if two provided meshes contain cellboxes with the same IDs

        Args:
            mesh_a (dict): A verfided correct mesh for use in regression
            testing
            mesh_b (dict): The currently calculated mesh from PolarRoute

        Throws:
            Fails if any cellbox exists in mesh_a that or not in mesh_b,
            or any cellbox exsits in mesh_B that is not in mesh_a
    """
    indxed_a = dict()
    for cellbox in mesh_a:
        indxed_a[cellbox['id']] = cellbox

    indxed_b = dict()
    for cellbox in mesh_b:
        indxed_b[cellbox['id']] = cellbox

    mesh_a_ids = set(indxed_a.keys())
    mesh_b_ids = set(indxed_b.keys())

    missing_a_ids = list(mesh_b_ids - mesh_a_ids)
    missing_b_ids = list(mesh_a_ids - mesh_b_ids)

    assert(indxed_a.keys()  == indxed_b.keys()), \
        f"Mismatch in cellbox IDs. ID's {missing_a_ids} have appeared in the new mesh. ID's {missing_b_ids} are missing from the new mesh"

def test_cellbox_values(mesh_a, mesh_b):
    """
        Tests if the values in of all attributes in each cellbox and the
        same in both provided meshes.

        Args:
            mesh_a (dict): A verfided correct mesh for use in regression
            testing
            mesh_b (dict): The currently calculated mesh from PolarRoute

        Throws:
            Fails if any values of any attributes differ between mesh_a
            and mesh_b
    """
    indxed_a = dict()
    for cellbox in mesh_a:
        indxed_a[cellbox['id']] = cellbox

    indxed_b = dict()
    for cellbox in mesh_b:
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

def test_cellbox_attributes(mesh_a, mesh_b):
    """
        Tests if the attributes of cellboxs in mesh_a and the same as
        attributes of cellboxes in mesh_b

        Note:
            This assumes that every cellbox in mesh has the same amount
            of attributes, so only compares the attributes of the first
            two cellboxes in the mesh

        Args:
            mesh_a (dict): A verfided correct mesh for use in regression
            testing
            mesh_b (dict): The currently calculated mesh from PolarRoute

        Throws:
            Fails if the cellboxes in the provided meshes contain different
            attributes
    """
    mesh_a_attributes = set(mesh_a[0].keys())
    mesh_b_attributes = set(mesh_b[0].keys())

    missing_a_attributes = list(mesh_b_attributes - mesh_a_attributes)
    missing_b_attributes = list(mesh_a_attributes - mesh_b_attributes)

    assert(mesh_a_attributes == mesh_b_attributes), \
        f"Mismatch in cellbox attributes. Attributes {missing_a_attributes} have appeared in the new mesh. Attributes {missing_b_attributes} are missing in the new mesh"

def test_neighbour_graph_count(graph_a, graph_b):
    """

        Args:
            graph_a (dict): A verfided correct neighbour graph for use in regression
            testing
            graph_b (dict): The currently calculated neighbour graph from PolarRoute

    """
    graph_a_count = len(graph_a.keys())
    graph_b_count = len(graph_b.keys())

    assert(graph_a_count == graph_b_count), \
        f"Incorrect number of nodes in neighbour graph. Expected: <{graph_a_count}> nodes, got: <{graph_b_count}> nodes."

def test_neighbour_graph_ids(graph_a, graph_b):
    """

        Args:
            graph_a (dict): A verfided correct neighbour graph for use in regression
            testing
            graph_b (dict): The currently calculated neighbour graph from PolarRoute

    """
    graph_a_ids = set(graph_a.keys())
    graph_b_ids = set(graph_b.keys())

    missing_a_keys = list(graph_b_ids - graph_a_ids)
    missing_b_keys = list(graph_a_ids - graph_b_ids)

    assert(graph_a_ids == graph_b_ids) , \
        f"Mismatch in neighbour graph nodes. <{len(missing_a_keys)}> nodes  have appeared in the new neighbour graph. <{len(missing_b_keys)}> nodes  are missing from the new neighbour graph."

def test_neighbour_graph_values(graph_a, graph_b):
    """

        Args:
            graph_a (dict): A verfided correct neighbour graph for use in regression
            testing
            graph_b (dict): The currently calculated neighbour graph from PolarRoute

    """
    mismatch_neighbors = dict()

    for node in graph_a.keys():
        # Prevent crashing if node not found. 
        # This will be detected by 'test_neighbour_graph_ids'.
        if node in graph_b.keys():
            neighbours_a = graph_a[node]
            neighbours_b = graph_b[node]

            if not neighbours_b == neighbours_a:
                mismatch_neighbors[node] = neighbours_b

    assert(len(mismatch_neighbors) == 0), \
        f"Mismatch in neighbour graph neighbours. <{len(mismatch_neighbors.keys())}> nodes have changed in new mesh."

def main():
    """
        main method for executions
    """
    args = sys.argv[1:]
    file = args[0]

    with open(file, 'r') as f:
        info = json.load(f)

    print("loading regression...")
    config = info['config']

    print("building Mesh...")
    test_mesh = Mesh(config)

    print("converting Mesh to json...")
    t_mesh_json = test_mesh.to_json()

    print("applying vessel performance...")
    vp = VesselPerformance(t_mesh_json)
    t_mesh_json = vp.to_json()

    print("regression testing Mesh...")
    # Testing cellboxes
    regression_mesh = info['cellboxes']
    new_mesh = t_mesh_json['cellboxes']
    try:
        test_cellbox_count(regression_mesh, new_mesh)
        print("test_cellbox_count : passed")
    except AssertionError as warning:
        print("test_cellbox_count : failed")
        print("    " + str(warning))

    try:
        test_cellbox_ids(regression_mesh, new_mesh)
        print("test_cellbox_ids : passed")
    except AssertionError as warning:
        print("test_cellbox_ids : failed")
        print("    " + str(warning))

    try:
        test_cellbox_attributes(regression_mesh, new_mesh)
        print("test_cellbox_attributes : passed")
    except AssertionError as warning:
        print("test_cellbox_attributes : failed")
        print("    " + str(warning))

    try:
        test_cellbox_values(regression_mesh, new_mesh)
        print("test_cellbox_values : passed")
    except AssertionError as warning:
        print("test_cellbox_values' : failed")
        print("    " + str(warning))

    # Testing neighbour_graphs
    regression_graph = info['neighbour_graph']
    new_graph = t_mesh_json['neighbour_graph']
    try:
        test_neighbour_graph_count(regression_graph, new_graph)
        print("test_neighbour_graph_count : passed")
    except AssertionError as warning:
        print("test_neighbour_graph_count : failed")
        print("    " + str(warning))
    try:
        test_neighbour_graph_ids(regression_graph, new_graph)
        print("test_neighbour_graph_ids : passed")
    except AssertionError as warning:
        print("test_neighbour_graph_ids : failed")
        print("    " + str(warning))
    try:
        test_neighbour_graph_values(regression_graph, new_graph)
        print("test_neighbour_graph_values : passed")
    except AssertionError as warning:
        print("test_neighbour_graph_values : failed")
        print("    " + str(warning))
    print("tests complete!")
    

if __name__ == "__main__":
    main()