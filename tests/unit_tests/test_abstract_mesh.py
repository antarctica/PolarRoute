from polar_route.mesh import Mesh
from polar_route.cellbox import CellBox
from generate_abstract_masks import find_data_fn, gen_circle, gen_gradient, gen_checkerboard
import json
import pytest

ABSTRACT_MESHES = [
    './abstract_meshes/h_grad_mesh.json',
    './abstract_meshes/v_grad_mesh.json',
    './abstract_meshes/circle_mesh.json',
    './abstract_meshes/checkerboard_mesh_nosplit.json',
    './abstract_meshes/checkerboard_mesh_singlesplit.json',
    './abstract_meshes/checkerboard_mesh_complexsplit.json',
]

@pytest.fixture(scope='session', autouse=True, params=ABSTRACT_MESHES)
def abstract_mesh_pair(request):
    # Read in settings for reference mesh
    with open(request.param, 'r') as f:
        reference_mesh = json.load(f)
    
    config = reference_mesh['config']
    details = config['Mesh_info']['Extra']

    # Set parameters for general mesh generation
    latMin = config['Mesh_info']['Region']['latMin']
    latMax = config['Mesh_info']['Region']['latMax']
    longMin = config['Mesh_info']['Region']['longMin']
    longMax = config['Mesh_info']['Region']['longMax']
    
    # Set parameters for abstract mesh generation
    inclusive = details['inclusive']
    radius = details['radius']
    centre = details['centre']
    gridsize = details['gridsize']
    vertical = details['vertical']
    resolution = details['resolution']
    value_output_type = json.loads(details['value_output_type'])
    value_fill_type = json.loads(details['value_fill_type'])
    splitting_condition = json.loads(details['splitting_conditions'])
    split_depth = details['split_depth']

    # Set which function will be used to compare against
    gen_data = find_data_fn(details['test_type'])
    # Create dataset to compare against
    dummy_data = gen_data(latMin, latMax, longMin, longMax, 
                          inclusive=inclusive, 
                          radius=radius, centre=centre,
                          gridsize=gridsize,
                          vertical=vertical,
                          resolution=resolution)
    # Create new mesh to test against reference
    test_mesh = Mesh(config)
    test_mesh.add_data_points(dummy_data)

    # Split cellboxes
    for cellbox in test_mesh.cellboxes:
        if isinstance(cellbox, CellBox):
            cellbox.add_value_output_type(value_output_type)
            cellbox.set_value_fill_types(value_fill_type)
            cellbox.add_splitting_condition(splitting_condition)
    test_mesh.split_to_depth(split_depth)

    # Set to output format
    test_output = test_mesh.to_json()

    # Remove extra info that's not part of mesh generation
    # del reference_mesh['config']['Mesh_info']['Extra']

    return [test_output, reference_mesh]


def test_cellbox_count(abstract_mesh_pair):
    """
        Test if two provided meshes contain the same number of cellboxes

        Args:
            abstract_mesh_pair (list): A pair of meshes to compare in regression testing:
                abstract_mesh_pair[0] -> regression_mesh (dict): A verfided correct mesh for use in regression
                testing
                abstract_mesh_pair[1] -> new_mesh (dict): The currently calculated mesh from PolarRoute

        Throws:
            Fails if the number of cellboxes in regression_mesh and new_mesh are
            not equal
    """
    regression_mesh = abstract_mesh_pair[0]['cellboxes']
    new_mesh = abstract_mesh_pair[1]['cellboxes']

    cellbox_count_a = len(regression_mesh)
    cellbox_count_b = len(new_mesh)

    assert(cellbox_count_a == cellbox_count_b), \
        f"Incorrect number of cellboxes in new mesh. Expected :{cellbox_count_a}, got: {cellbox_count_b}"

def test_cellbox_ids(abstract_mesh_pair):
    """
        Test if two provided meshes contain cellboxes with the same IDs

        Args:
            abstract_mesh_pair (list): A pair of meshes to compare in regression testing:
                abstract_mesh_pair[0] -> regression_mesh (dict): A verfided correct mesh for use in regression
                testing
                abstract_mesh_pair[1] -> new_mesh (dict): The currently calculated mesh from PolarRoute

        Throws:
            Fails if any cellbox exists in regression_mesh that or not in new_mesh,
            or any cellbox exsits in new_mesh that is not in regression_mesh
    """
    regression_mesh = abstract_mesh_pair[0]['cellboxes']
    new_mesh = abstract_mesh_pair[1]['cellboxes']

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

def test_cellbox_values(abstract_mesh_pair):
    """
        Tests if the values in of all attributes in each cellbox and the
        same in both provided meshes.

       Args:
            abstract_mesh_pair (list): A pair of meshes to compare in regression testing:
                abstract_mesh_pair[0] -> regression_mesh (dict): A verfided correct mesh for use in regression
                testing
                abstract_mesh_pair[1] -> new_mesh (dict): The currently calculated mesh from PolarRoute

        Throws:
            Fails if any values of any attributes differ between regression_mesh
            and new_mesh
    """
    regression_mesh = abstract_mesh_pair[0]['cellboxes']
    new_mesh = abstract_mesh_pair[1]['cellboxes']

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

def test_cellbox_attributes(abstract_mesh_pair):
    """
        Tests if the attributes of cellboxes in regression_mesh and the same as
        attributes of cellboxes in new_mesh

        Note:
            This assumes that every cellbox in mesh has the same amount
            of attributes, so only compares the attributes of the first
            two cellboxes in the mesh

        Args:
            abstract_mesh_pair (list): A pair of meshes to compare in regression testing:
                abstract_mesh_pair[0] -> regression_mesh (dict): A verfided correct mesh for use in regression
                testing
                abstract_mesh_pair[1] -> new_mesh (dict): The currently calculated mesh from PolarRoute

        Throws:
            Fails if the cellboxes in the provided meshes contain different
            attributes
    """
    regression_mesh = abstract_mesh_pair[0]['cellboxes']
    new_mesh = abstract_mesh_pair[1]['cellboxes']

    regression_regression_meshttributes = set(regression_mesh[0].keys())
    new_mesh_attributes = set(new_mesh[0].keys())

    missing_a_attributes = list(new_mesh_attributes - regression_regression_meshttributes)
    missing_b_attributes = list(regression_regression_meshttributes - new_mesh_attributes)

    assert(regression_regression_meshttributes == new_mesh_attributes), \
        f"Mismatch in cellbox attributes. Attributes {missing_a_attributes} have appeared in the new mesh. Attributes {missing_b_attributes} are missing in the new mesh"

def test_neighbour_graph_count(abstract_mesh_pair):
    """
        Tests that the neighbour_graph in the regression mesh and the newly calculated mesh have the 
        same number of nodes.

        Args:
            abstract_mesh_pair (list): A pair of meshes to compare in regression testing:
                abstract_mesh_pair[0] -> regression_mesh (dict): A verfided correct mesh for use in regression
                testing
                abstract_mesh_pair[1] -> new_mesh (dict): The currently calculated mesh from PolarRoute

    """
    regression_graph = abstract_mesh_pair[0]['neighbour_graph']
    new_graph = abstract_mesh_pair[1]['neighbour_graph']

    regression_graph_count = len(regression_graph.keys())
    new_graph_count = len(new_graph.keys())

    assert(regression_graph_count == new_graph_count), \
        f"Incorrect number of nodes in neighbour graph. Expected: <{regression_graph_count}> nodes, got: <{new_graph_count}> nodes."

def test_neighbour_graph_ids(abstract_mesh_pair):
    """
        Tests that the neighbour_graph in the regression mesh and the newly calculated mesh contain
        all the same node IDs.

        Args:
            abstract_mesh_pair (list): A pair of meshes to compare in regression testing:
                abstract_mesh_pair[0] -> regression_mesh (dict): A verfided correct mesh for use in regression
                testing
                abstract_mesh_pair[1] -> new_mesh (dict): The currently calculated mesh from PolarRoute
    """
    regression_graph = abstract_mesh_pair[0]['neighbour_graph']
    new_graph = abstract_mesh_pair[1]['neighbour_graph']

    regression_graph_ids = set(regression_graph.keys())
    new_graph_ids = set(new_graph.keys())

    missing_a_keys = list(new_graph_ids - regression_graph_ids)
    missing_b_keys = list(regression_graph_ids - new_graph_ids)

    assert(regression_graph_ids == new_graph_ids) , \
        f"Mismatch in neighbour graph nodes. <{len(missing_a_keys)}> nodes  have appeared in the new neighbour graph. <{len(missing_b_keys)}> nodes  are missing from the new neighbour graph."

def test_neighbour_graph_values(abstract_mesh_pair):
    """
        Tests that each node in the neighbour_graph of the regression mesh and the newly calculated
        mesh have the same neighbours.

        Args:
            abstract_mesh_pair (list): A pair of meshes to compare in regression testing:
                abstract_mesh_pair[0] -> regression_mesh (dict): A verfided correct mesh for use in regression
                testing
                abstract_mesh_pair[1] -> new_mesh (dict): The currently calculated mesh from PolarRoute

    """
    regression_graph = abstract_mesh_pair[0]['neighbour_graph']
    new_graph = abstract_mesh_pair[1]['neighbour_graph']

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

# def test_h_grad():
#     """
#         Test if horizontal gradient creates same output as reference

#         Throws:
#             Fails if output JSON doesn't match reference
#     """
#     # Open reference file to retrieve config settings
#     with open(H_GRAD_MESH, 'r') as f:
#         reference_output = json.load(f)
#     config = reference_output['config'] 

#     # Create dummy data   
#     dummy_data = gen_gradient(LATMIN, LATMAX, LONGMIN, LONGMAX, vertical=False, n=RESOLUTION)

#     # Create new mesh to test against reference
#     test_mesh = Mesh(config)
#     test_mesh.add_data_points(dummy_data)
#     value_output_type   = json.loads(VALUE_OUTPUT_TYPE)
#     value_fill_type     = json.loads(VALUE_FILL_TYPE)
#     splitting_condition = json.loads(SPLITTING_CONDITION)

#     # Split cellboxes
#     for cellbox in test_mesh.cellboxes:
#         if isinstance(cellbox, CellBox):
#             cellbox.add_value_output_type(value_output_type)
#             cellbox.set_value_fill_types(value_fill_type)
#             cellbox.add_splitting_condition(splitting_condition)
#     test_mesh.split_to_depth(SPLIT_DEPTH)

#     # Set to putput format
#     test_output = test_mesh.to_json()

#     # Assert to test differences
#     assert(test_output == reference_output), \
#         f"Horizontal gradient mesh does not match reference"

