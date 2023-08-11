import pandas as pd

from polar_route.utils import round_to_sigfig

SIG_FIG_TOLERANCE = 4

# Testing mesh outputs
def test_mesh_cellbox_count(mesh_pair):
    compare_cellbox_count(mesh_pair[0], mesh_pair[1])

def test_mesh_cellbox_ids(mesh_pair):
    compare_cellbox_ids(mesh_pair[0], mesh_pair[1])

def test_mesh_cellbox_values(mesh_pair):
    compare_cellbox_values(mesh_pair[0], mesh_pair[1])

def test_mesh_cellbox_attributes(mesh_pair):
    compare_cellbox_attributes(mesh_pair[0], mesh_pair[1])

def test_mesh_neighbour_graph_count(mesh_pair):
    compare_neighbour_graph_count(mesh_pair[0], mesh_pair[1])

def test_mesh_neighbour_graph_ids(mesh_pair):
    compare_neighbour_graph_ids(mesh_pair[0], mesh_pair[1])

def test_mesh_neighbour_graph_values(mesh_pair):
    compare_neighbour_graph_values(mesh_pair[0], mesh_pair[1])


# Comparison between old and new
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
    regression_mesh = extract_cellboxes(mesh_a)
    new_mesh = extract_cellboxes(mesh_b)

    cellbox_count_a = len(regression_mesh)
    cellbox_count_b = len(new_mesh)

    assert(cellbox_count_a == cellbox_count_b), \
        f"Incorrect number of cellboxes in new mesh. "\
        f"Expected :{cellbox_count_a}, got: {cellbox_count_b}"

def compare_cellbox_ids(mesh_a, mesh_b):
    """
        Test if two provided meshes contain cellboxes with the same IDs

       Args:
            mesh_a (json)
            mesh_b (json)

        Throws:
            Fails if any cellbox exists in regression_mesh that or not in new_mesh,
            or any cellbox exists in new_mesh that is not in regression_mesh
    """
    regression_mesh = extract_cellboxes(mesh_a)
    new_mesh = extract_cellboxes(mesh_b)

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
        f"Mismatch in cellbox IDs. ID's {missing_a_ids} have appeared in the "\
        f"new mesh. ID's {missing_b_ids} are missing from the new mesh"

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
    # Retrieve cellboxes from meshes
    df_a = pd.DataFrame(extract_cellboxes(mesh_a)).set_index('geometry')
    df_b = pd.DataFrame(extract_cellboxes(mesh_b)).set_index('geometry')
    # Extract only cellboxes with same boundaries
    # Drop ID as that may be different despite same boundary
    df_a = df_a.loc[extract_common_boundaries(mesh_a, mesh_b)].drop(columns=['id'])
    df_b = df_b.loc[extract_common_boundaries(mesh_a, mesh_b)].drop(columns=['id'])
    # Ignore cellboxes with different boundaries, that will be picked up in other tests

    # For each mesh
    for df in [df_a, df_b]:
        # Round to sig figs if column contains floats
        float_cols = df.select_dtypes(include=float).columns
        for col in float_cols:
            df[col] = round_to_sigfig(df[col].to_numpy(), 
                                      sigfig=SIG_FIG_TOLERANCE)
        # Round to sig figs if column contains list, which may contain floats
        list_cols = df.select_dtypes(include=list).columns
        # Loop through columns and round any values within lists of floats
        for col in list_cols:
            round_col = list()
            for val in df[col]:
                if type(val) == list and all([type(x) == float for x in val]):
                    round_col.append(round_to_sigfig(val, sigfig=SIG_FIG_TOLERANCE))
                else:
                    round_col.append(val)

            df[col] = round_col

        
    # Find difference between the two
    diff = df_a.compare(df_b).rename({'self': 'old', 'other':'new'})

    assert(len(diff) == 0), \
        f'Mismatch between values in common cellboxes:\n'\
        f'{diff.to_string(max_colwidth=10)}'

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
    regression_mesh = extract_cellboxes(mesh_a)
    new_mesh = extract_cellboxes(mesh_b)

    regression_mesh_attributes = set(regression_mesh[0].keys())
    new_mesh_attributes        = set(new_mesh[0].keys())

    missing_a_attributes = list(new_mesh_attributes - regression_mesh_attributes)
    missing_b_attributes = list(regression_mesh_attributes - new_mesh_attributes)

    assert(regression_mesh_attributes == new_mesh_attributes), \
        f"Mismatch in cellbox attributes. Attributes {missing_a_attributes} "\
        f"have appeared in the new mesh. Attributes {missing_b_attributes} "\
        f"are missing in the new mesh"

def compare_neighbour_graph_count(mesh_a, mesh_b):
    """
        Tests that the neighbour_graph in the regression mesh and the newly calculated mesh have the 
        same number of nodes.

         Args:
            mesh_a (json)
            mesh_b (json)

    """
    regression_graph = extract_neighbour_graph(mesh_a)
    new_graph = extract_neighbour_graph(mesh_b)

    regression_graph_count = len(regression_graph.keys())
    new_graph_count = len(new_graph.keys())

    assert(regression_graph_count == new_graph_count), \
        f"Incorrect number of nodes in neighbour graph. "\
        f"Expected: <{regression_graph_count}> nodes, "\
        f"got: <{new_graph_count}> nodes."

def compare_neighbour_graph_ids(mesh_a, mesh_b):
    """
        Tests that the neighbour_graph in the regression mesh and the newly calculated mesh contain
        all the same node IDs.

        Args:
            mesh_a (json)
            mesh_b (json)
    """
    regression_graph = extract_neighbour_graph(mesh_a)
    new_graph = extract_neighbour_graph(mesh_b)

    regression_graph_ids = set(regression_graph.keys())
    new_graph_ids = set(new_graph.keys())

    missing_a_keys = list(new_graph_ids - regression_graph_ids)
    missing_b_keys = list(regression_graph_ids - new_graph_ids)

    assert(regression_graph_ids == new_graph_ids) , \
        f"Mismatch in neighbour graph nodes. <{len(missing_a_keys)}> nodes "\
        f"have appeared in the new neighbour graph. <{len(missing_b_keys)}> "\
        f"nodes are missing from the new neighbour graph."

def compare_neighbour_graph_values(mesh_a, mesh_b):
    """
        Tests that each node in the neighbour_graph of the regression mesh and the newly calculated
        mesh have the same neighbours.

        Args:
            mesh_a (json)
            mesh_b (json)

    """
    regression_graph = extract_neighbour_graph(mesh_a)
    new_graph = extract_neighbour_graph(mesh_b)

    mismatch_neighbours = dict()

    for node in regression_graph.keys():
        # Prevent crashing if node not found. 
        # This will be detected by 'test_neighbour_graph_ids'.
        if node in new_graph.keys():
            neighbours_a = regression_graph[node]
            neighbours_b = new_graph[node]

            # Sort the lists of neighbours as ordering is not important
            sorted_neighbours_a = {k:sorted(neighbours_a[k]) 
                                   for k in neighbours_a.keys()}
            sorted_neighbours_b = {k: sorted(neighbours_b[k]) 
                                   for k in neighbours_b.keys()}

            if sorted_neighbours_b != sorted_neighbours_a:
                mismatch_neighbours[node] = sorted_neighbours_b

    assert(len(mismatch_neighbours) == 0), \
        f"Mismatch in neighbour graph neighbours. "\
        f"<{len(mismatch_neighbours.keys())}> nodes have changed in the new mesh."



# Utility functions
def extract_neighbour_graph(mesh):
    """
    Extracts out the neighbour graph from a mesh
    
    Args:
        mesh (json): Complete mesh output
        
    Returns:
        dict: Neighbour graph for each cellbox
    """
    return mesh['neighbour_graph']

def extract_cellboxes(mesh):
    """
    Extracts out the cellboxes and aggregated info from a mesh
    
    Args:
        mesh (json): Complete mesh output
        
    Returns:
        list: Each cellbox as a dict/json object, in a list 
    """
    return mesh['cellboxes']

def extract_common_boundaries(mesh_a, mesh_b):
    """
    Creates a list of common boundaries between two mesh jsons

    Args:
        mesh_a (json): First mesh json to extract boundaries from
        mesh_b (json): Second mesh json to extract boundaries from

    Returns:
        list: List of common cellbox boundaries (as strings) 
    """
    bounds_a = [cb['geometry'] for cb in extract_cellboxes(mesh_a)]
    bounds_b = [cb['geometry'] for cb in extract_cellboxes(mesh_b)]

    common_bounds = [geom for geom in bounds_a if geom in bounds_b]

    return common_bounds