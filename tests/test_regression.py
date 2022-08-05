"""
    Regression testing package to ensure consistance functionality in development
    of the PolarRoute python package.
"""

import numpy as np

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

            if not cellbox_a == cellbox_b:
                mismatch_values = []
                for key in cellbox_a.keys():
                    value_a = cellbox_a[key]
                    value_b = cellbox_b[key]

                    if not(value_a == value_b) and not(np.isnan(value_a) and np.isnan(value_b)):
                        mismatch_values.append(key)
                mismatch_cellboxes[cellbox_a['id']] = mismatch_values

    assert(len(mismatch_cellboxes) == 0) , \
        f"Values in some cellboxes in the new mesh have changed. The changes cellboxes are: {mismatch_cellboxes}"

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
        f"Incorrect number of nodes in neighbour graph. Expected: {graph_a_count}, got {graph_b_count}"

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
        f"Mismatch in neighbour graph nodes. Nodes {missing_a_keys} have appeared in the new neighbour graph. Nodes {missing_b_keys} are missing from the new neighbour graph"

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
        f"Mismatch in neighbour graph neighbours. Nodes {mismatch_neighbors.keys()} have changed in new mesh."
