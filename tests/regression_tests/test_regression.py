"""
    Regression testing package to ensure consistance functionality in development
    of the PolarRoute python package.
"""
import numpy as np
import pandas as pd
import json
import pytest

from polar_route.mesh import Mesh
from polar_route.vessel_performance import VesselPerformance

#File locations of all vessel performance meshes to be recaculated for regression testing.
TEST_VESSEL_MESHES = [
    './example_meshes/Vessel_Performance_Meshes/add_vehicle.output2013_4_80.json',
    './example_meshes/Vessel_Performance_Meshes/add_vehicle.output2017_6_80.json',
    './example_meshes/Vessel_Performance_Meshes/add_vehicle.output2019_6_80.json'
]

#File locations of all enviromental meshes to be recaculated for regression testing.
TEST_ENV_MESHES = [
    './example_meshes/Enviromental_Meshes/create_mesh.output2013_4_80.json',
    './example_meshes/Enviromental_Meshes/create_mesh.output2016_6_80.json',
    './example_meshes/Enviromental_Meshes/create_mesh.output2019_6_80.json'
]

TEST_V_GRAD_MESHES = [2.5
    './example_meshes/Abstract_Environmental_Meshes/vgrad_n201_vT_mesh.json'
]
TEST_H_GRAD_MESHES = [
    './example_meshes/Abstract_Environmental_Meshes/hgrad_n201_vF_mesh.json'
]
TEST_CHECKERBOARD_MESHES = [
    './example_meshes/Abstract_Environmental_Meshes/checkerboard_n201_gy2.5_gx2.5_mesh.json'
    './example_meshes/Abstract_Environmental_Meshes/checkerboard_n201_gy5_gx2.5_mesh.json'
    './example_meshes/Abstract_Environmental_Meshes/checkerboard_n201_gy6_gx3_mesh.json'
]
TEST_CIRCLE_MESHES = [
    './example_meshes/Abstract_Environmental_Meshes/circle_n201_r2_cy-62.5_cx-60.0_mesh.json'
]

@pytest.fixture(scope='session', autouse=False, params=TEST_ENV_MESHES)
def env_mesh_pair(request):
    """
        creates a mesh pair for all meshes listed in array TEST_ENV_MESHES
    """
    return calculate_env_mesh(request.param)

@pytest.fixture(scope='session', autouse=False, params=TEST_VESSEL_MESHES)
def vessel_mesh_pair(request):
    """
        creates a mesh pair for all vessel performance meshes listed 
        in array TEST_VESSEL_MESHES
    """
    return calculate_vessel_mesh(request.param)

@pytest.fixture(scope='session', autouse=False, params=TEST_CIRCLE_MESHES)
def circle_mesh_pair(request):
    """
        creates a mesh pair for circle meshes listed in TEST_ABSTRACT_MESHES
    """
    return calculate_circle_mesh(request.param)


# # Testing Enviromental Meshes
# def test_env_mesh_cellbox_count(env_mesh_pair):
#     compare_cellbox_count(env_mesh_pair[0], env_mesh_pair[1])

# def test_env_mesh_cellbox_ids(env_mesh_pair):
#     compare_cellbox_ids(env_mesh_pair[0], env_mesh_pair[1])

# def test_env_mesh_cellbox_values(env_mesh_pair):
#     compare_cellbox_values(env_mesh_pair[0], env_mesh_pair[1])

# def test_env_mesh_cellbox_attributes(env_mesh_pair):
#     compare_cellbox_attributes(env_mesh_pair[0], env_mesh_pair[1])

# def test_env_mesh_neighbour_graph_count(env_mesh_pair):
#     compare_neighbour_graph_count(env_mesh_pair[0], env_mesh_pair[1])

# def test_env_mesh_neighbour_graph_ids(env_mesh_pair):
#     compare_neighbour_graph_ids(env_mesh_pair[0], env_mesh_pair[1])

# def test_env_mesh_neighbour_graph_values(env_mesh_pair):
#     compare_neighbour_graph_count(env_mesh_pair[0], env_mesh_pair[1])

# # Testing Vessel Performances Meshes
# def test_vp_mesh_cellbox_count(vessel_mesh_pair):
#     compare_cellbox_count(vessel_mesh_pair[0], vessel_mesh_pair[1])

# def test_vp_mesh_cellbox_ids(vessel_mesh_pair):
#     compare_cellbox_ids(vessel_mesh_pair[0], vessel_mesh_pair[1])

# def test_vp_mesh_cellbox_values(vessel_mesh_pair):
#     compare_cellbox_values(vessel_mesh_pair[0], vessel_mesh_pair[1])

# def test_vp_mesh_cellbox_attributes(vessel_mesh_pair):
#     compare_cellbox_attributes(vessel_mesh_pair[0], vessel_mesh_pair[1])

# def test_vp_mesh_neighbour_graph_count(vessel_mesh_pair):
#     compare_neighbour_graph_count(vessel_mesh_pair[0], vessel_mesh_pair[1])

# def test_vp_mesh_neighbour_graph_ids(vessel_mesh_pair):
#     compare_neighbour_graph_ids(vessel_mesh_pair[0], vessel_mesh_pair[1])

# def test_vp_mesh_neighbour_graph_values(vessel_mesh_pair):
#     compare_neighbour_graph_count(vessel_mesh_pair[0], vessel_mesh_pair[1])

# Testing Abstract Meshes
#Circle
def test_env_mesh_cellbox_count(circle_mesh_pair):
    compare_cellbox_count(circle_mesh_pair[0], circle_mesh_pair[1])

def test_env_mesh_cellbox_ids(circle_mesh_pair):
    compare_cellbox_ids(circle_mesh_pair[0], circle_mesh_pair[1])

def test_env_mesh_cellbox_values(circle_mesh_pair):
    compare_cellbox_values(circle_mesh_pair[0], circle_mesh_pair[1])

def test_env_mesh_cellbox_attributes(circle_mesh_pair):
    compare_cellbox_attributes(circle_mesh_pair[0], circle_mesh_pair[1])

def test_env_mesh_neighbour_graph_count(circle_mesh_pair):
    compare_neighbour_graph_count(circle_mesh_pair[0], circle_mesh_pair[1])

def test_env_mesh_neighbour_graph_ids(circle_mesh_pair):
    compare_neighbour_graph_ids(circle_mesh_pair[0], circle_mesh_pair[1])

def test_env_mesh_neighbour_graph_values(circle_mesh_pair):
    compare_neighbour_graph_count(circle_mesh_pair[0], circle_mesh_pair[1])

#TODO test_v_grad_mesh
#TODO test_h_grad_mesh
#TODO test_checkerboard_mesh

# Utility functions
def calculate_env_mesh(mesh_location):
    """
        recreates an enviromental mesh from the config of a pre-computed mesh.

        params:
            mesh_location (string): File location of the mesh to be recomputed

        returns:
            mesh_pair (list):
                mesh_pair[0]: Regression mesh (from pre-computed mesh file)
                mesh_pair[1]: Recomputed mesh (recalculated from config in mesh file)
    """
    with open(mesh_location, 'r') as f:
        regression_mesh = json.load(f)

    config = regression_mesh['config']
    new_mesh = Mesh(config)

    new_mesh = new_mesh.to_json()

    return [regression_mesh, new_mesh]

def calculate_vessel_mesh(mesh_location):
    """
        recreates a vessel performance mesh from the config of a pre-computed mesh.

        params:
            mesh_location (string): File location of the mesh to be recomputed

        returns:
            mesh_pair (list):
                mesh_pair[0]: Regression mesh (from pre-computed mesh file)
                mesh_pair[1]: Recomputed mesh (recalculated from config in mesh file)
    """
    env_meshes = calculate_env_mesh(mesh_location)

    regression_mesh = env_meshes[0]

    new_mesh = VesselPerformance(env_meshes[1])
    new_mesh = new_mesh.to_json()

    return [regression_mesh, new_mesh]

def calculate_circle_mesh(mesh_location):
    """
        recreates a circular environment mesh from the config of a pre-computed mesh.

        params:
            mesh_location (string): File location of the mesh to be recomputed

        returns:
            mesh_pair (list):
                mesh_pair[0]: Regression mesh (from pre-computed mesh file)
                mesh_pair[1]: Recomputed mesh (recalculated from config in mesh file)
    """ 
    with open(mesh_location, 'r') as f:
        regression_mesh = json.load(f)
    # Retrieve config info
    config = regression_mesh['config']
    latMin      = config['Mesh_info']['Region']['latMin'] 
    latMax      = config['Mesh_info']['Region']['latMax'] 
    longMin     = config['Mesh_info']['Region']['longMin'] 
    longMax     = config['Mesh_info']['Region']['longMax'] 
    startTime   = config['Mesh_info']['Region']['startTime']
    split_depth = config['Mesh_info']['splitting']['split_depth']
    
    # Remove path from mesh_location
    filename = mesh_location.split('/')[-1]
    # Retrieve parameters and remove 'circle' and 'mesh.json' from list
    parameters = filename.split('_')[1:-1]
    # Loop through parameters to set variables for circle generation
    for parameter in parameters:
        if parameter[0] == 'n':         # Resolution of datapoint grid
            n = int(parameter[1:])
        elif parameter[0] == 'r':       # Radius of circle (deg)
            r = float(parameter[1:])
        elif parameter[0:2] == 'cy':    # Centre lat (deg)
            cy = float(parameter[2:])
        elif parameter[0:2] == 'cx':    # Centre long (deg)
            cx = float(parameter[2:])
    centre = (cy, cx)

    # Create circle data for mesh generation
    circle_datapoints = gen_circle(latMin, latMax, longMin, longMax, 
                                   time=startTime, n=n,
                                   radius=r, centre=centre)
    # Create new mesh to compare against
    new_mesh = Mesh(config)
    new_mesh.add_data_points(circle_datapoints)
    new_mesh.split_to_depth(split_depth)

    # Set to JSON to make it comparable to regression_mesh
    new_mesh = new_mesh.to_json()

    return [regression_mesh, new_mesh]

#TODO calculate_v_grad_mesh
#TODO calculate_h_grad_mesh
#TODO calculate_checkerboard_mesh

def compare_cellbox_count(mesh_a, mesh_b):
    """
        Test if two provided meshes contain the same number of cellboxes

        Args:
            mesh_a (json)
            mesh_b (json)

        Throws:
            Fails if the number of cellboxes in regression_mesh and new_mesh are
            not equal
    """
    regression_mesh = mesh_a['cellboxes']
    new_mesh = mesh_b['cellboxes']

    cellbox_count_a = len(regression_mesh)
    cellbox_count_b = len(new_mesh)

    assert(cellbox_count_a == cellbox_count_b), \
        f"Incorrect number of cellboxes in new mesh. Expected :{cellbox_count_a}, got: {cellbox_count_b}"

def compare_cellbox_ids(mesh_a, mesh_b):
    """
        Test if two provided meshes contain cellboxes with the same IDs

       Args:
            mesh_a (json)
            mesh_b (json)

        Throws:
            Fails if any cellbox exists in regression_mesh that or not in new_mesh,
            or any cellbox exsits in new_mesh that is not in regression_mesh
    """
    regression_mesh = mesh_a['cellboxes']
    new_mesh = mesh_b['cellboxes']

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

def compare_cellbox_values(mesh_a, mesh_b):
    """
        Tests if the values in of all attributes in each cellbox and the
        same in both provided meshes.

        Args:
            mesh_a (json)
            mesh_b (json)

        Throws:
            Fails if any values of any attributes differ between regression_mesh
            and new_mesh
    """
    regression_mesh = mesh_a['cellboxes']
    new_mesh = mesh_b['cellboxes']

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
                    print(key, value_a, type(value_a))
                    print(key, value_b, type(value_b))
                    if not(value_a == value_b) and not(np.isnan(value_a) or np.isnan(value_b)):
                        mismatch_values.append(key)
                        mismatch_cellboxes[cellbox_a['id']] = mismatch_values

    assert(len(mismatch_cellboxes) == 0) , \
        f"Values in <{len(mismatch_cellboxes.keys())}> cellboxes in the new mesh have changed. The changes cellboxes are: {mismatch_cellboxes}"

def compare_cellbox_attributes(mesh_a, mesh_b):
    """
        Tests if the attributes of cellboxes in regression_mesh and the same as
        attributes of cellboxes in new_mesh

        Note:
            This assumes that every cellbox in mesh has the same amount
            of attributes, so only compares the attributes of the first
            two cellboxes in the mesh

        Args:
            mesh_a (json)
            mesh_b (json)

        Throws:
            Fails if the cellboxes in the provided meshes contain different
            attributes
    """
    regression_mesh = mesh_a['cellboxes']
    new_mesh = mesh_b['cellboxes']

    regression_regression_meshttributes = set(regression_mesh[0].keys())
    new_mesh_attributes = set(new_mesh[0].keys())

    missing_a_attributes = list(new_mesh_attributes - regression_regression_meshttributes)
    missing_b_attributes = list(regression_regression_meshttributes - new_mesh_attributes)

    assert(regression_regression_meshttributes == new_mesh_attributes), \
        f"Mismatch in cellbox attributes. Attributes {missing_a_attributes} have appeared in the new mesh. Attributes {missing_b_attributes} are missing in the new mesh"

def compare_neighbour_graph_count(mesh_a, mesh_b):
    """
        Tests that the neighbour_graph in the regression mesh and the newly calculated mesh have the 
        same number of nodes.

         Args:
            mesh_a (json)
            mesh_b (json)

    """
    regression_graph = mesh_a['neighbour_graph']
    new_graph = mesh_b['neighbour_graph']

    regression_graph_count = len(regression_graph.keys())
    new_graph_count = len(new_graph.keys())

    assert(regression_graph_count == new_graph_count), \
        f"Incorrect number of nodes in neighbour graph. Expected: <{regression_graph_count}> nodes, got: <{new_graph_count}> nodes."

def compare_neighbour_graph_ids(mesh_a, mesh_b):
    """
        Tests that the neighbour_graph in the regression mesh and the newly calculated mesh contain
        all the same node IDs.

        Args:
            mesh_a (json)
            mesh_b (json)
    """
    regression_graph = mesh_a['neighbour_graph']
    new_graph = mesh_b['neighbour_graph']

    regression_graph_ids = set(regression_graph.keys())
    new_graph_ids = set(new_graph.keys())

    missing_a_keys = list(new_graph_ids - regression_graph_ids)
    missing_b_keys = list(regression_graph_ids - new_graph_ids)

    assert(regression_graph_ids == new_graph_ids) , \
        f"Mismatch in neighbour graph nodes. <{len(missing_a_keys)}> nodes  have appeared in the new neighbour graph. <{len(missing_b_keys)}> nodes  are missing from the new neighbour graph."

def compare_neighbour_graph_values(mesh_a, mesh_b):
    """
        Tests that each node in the neighbour_graph of the regression mesh and the newly calculated
        mesh have the same neighbours.

        Args:
            mesh_a (json)
            mesh_b (json)

    """
    regression_graph = mesh_a['neighbour_graph']
    new_graph = mesh_b['neighbour_graph']

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

def gen_circle(latMin, latMax, longMin, longMax, time='1970-01-01', radius=1, centre=None, n=100, **kwargs):
    """
        Generates a circle within bounds of lat/long min/max.

        Args:
            latMin (float)       : Minimum latitude of mesh
            latMax (float)       : Maximum latitude of mesh
            longMin (float)      : Minimum longitude of mesh
            longMax (float)      : Maximum longitude of mesh
            radius (float)       : Radius of circle to generate, in degrees
            centre (float, float): Tuple of central coordinates of circle in form (lat(deg), long(deg)).
                                   If None, then centre of mesh is chosen
            n (int)              : Intervals to divide lat and long range into
    """

    lat  = np.linspace(latMin, latMax, n)    # Generate rows
    long = np.linspace(longMin, longMax, n)  # Generate cols
    
    # Set centre as centre of data_grid if none specified
    if centre is None:
        centre = (lat[int(n/2)], long[int(n/2)])
    
    # Create vectors for row and col idx's
    y = np.vstack(np.linspace(latMin, latMax, n))
    x = np.linspace(longMin, longMax, n)
   
    # y, x = np.ogrid[:n, :n]              
    dist_from_centre = np.sqrt((x-centre[1])**2 + (y-centre[0])**2)     # Create a 2D-array with distance from defined centre
    mask = dist_from_centre <= radius    # Create mask
    # Set up empty dataframe to populate with dummy data
    dummy_df = pd.DataFrame(columns = ['lat', 'long', 'dummy_data'])
    # For each combination of lat/long
    for i in range(n):
        for j in range(n):
            # Create a new row, adding mask value
            row = pd.DataFrame(data={'lat':lat[i], 'long':long[j], 'dummy_data':mask[i][j]}, index=[0])
            dummy_df = pd.concat([dummy_df, row], ignore_index=True)
            
    # Change boolean values to int
    dummy_df = dummy_df.replace(False, 0)
    dummy_df = dummy_df.replace(True, 1)
    
    # Fill dummy time values
    dummy_df['time'] = time

    return dummy_df

def gen_gradient(latMin, latMax, longMin, longMax, time='1970-01-01', vertical=True, n=100, **kwargs):
    """
        Generates a gradient across the map

        Args:
            latMin (float)       : Minimum latitude of mesh
            latMax (float)       : Maximum latitude of mesh
            longMin (float)      : Minimum longitude of mesh
            longMax (float)      : Maximum longitude of mesh
            vertical (bool)      : Vertical gradient (true) or horizontal gradient (false)
            n (int)              : Intervals to divide lat and long range into
    """
    lat  = np.linspace(latMin, latMax, n)    # Generate rows
    long = np.linspace(longMin, longMax, n)  # Generate cols
    #Create 1D gradient
    gradient = np.linspace(0,1,n)
    
    dummy_df = pd.DataFrame(columns = ['lat', 'long', 'dummy_data'])
    # For each combination of lat/long
    for i in range(n):
        for j in range(n):
            # Change dummy data depending on which axis to gradient
            datum = gradient[i] if vertical else gradient[j]
            # Create a new row, adding datum value
            row = pd.DataFrame(data={'lat':lat[i], 'long':long[j], 'dummy_data':datum}, index=[0])
            dummy_df = pd.concat([dummy_df, row], ignore_index=True)  
    
    # Fill dummy time values
    dummy_df['time'] = time
    
    return dummy_df

def gen_checkerboard(latMin, latMax, longMin, longMax, time='1970-01-01', n=100, gridsize=(1,1), **kwargs):
    """
        Generates a checkerboard pattern across map

        Args:
            latMin (float)       : Minimum latitude of mesh
            latMax (float)       : Maximum latitude of mesh
            longMin (float)      : Minimum longitude of mesh
            longMax (float)      : Maximum longitude of mesh
            n (int)              : Intervals to divide lat and long range into
            gridsize (int, int)  : Tuple of size of boxes in checkerboard pattern, in form (lat(deg), long(deg))
    """
    lat  = np.linspace(latMin, latMax, n, endpoint=False)    # Generate rows
    long = np.linspace(longMin, longMax, n, endpoint=False)  # Generate cols
    # Create checkerboard pattern
    horizontal = np.floor((lat - latMin)   / gridsize[1]) % 2   # Create horizontal stripes of 0's and 1's, stripe size defined by gridsize
    vertical   = np.floor((long - longMin) / gridsize[0]) % 2   # Create vertical stripes of 0's and 1's, stripe size defined by gridsize
    dummy_df = pd.DataFrame(columns = ['lat', 'long', 'dummy_data'])
    # For each combination of lat/long
    for i in range(n):
        for j in range(n):
            # Horizontal XOR Vertical should create boxes
            datum = (horizontal[i] + vertical[j]) % 2
            # Create a new row, adding datum value
            row = pd.DataFrame(data={'lat':lat[i], 'long':long[j], 'dummy_data':datum}, index=[0])
            dummy_df = pd.concat([dummy_df, row], ignore_index=True)  
    
    # Fill dummy time values
    dummy_df['time'] = time
    
    return dummy_df    