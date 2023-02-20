from polar_route.EnvironmentMesh import EnvironmentMesh
from polar_route.vessel_performance.VesselFactory import VesselFactory
import logging
import json

class VesselPerformanceModeller:
    """
        Class for modelling the vessel performance.
        Takes both an environmental mesh and vessel config as input in json format and modifies the input mesh to
        include vessel specifics.
    """
    def __init__(self, env_mesh_json, vessel_config):
        """

        Args:
            env_mesh_json (str): a file path pointing to an environmental mesh json file
            vessel_config (str): a file path pointing to a vessel config json file
        """
        logging.info("Initialising Vessel Performance Modeller")

        self.env_mesh = EnvironmentMesh.load_from_json(env_mesh_json)
        self.vessel = VesselFactory.get_vessel(vessel_config)

    def model_accessibility(self):
        """

        Method to determine the accessibility of cells in the environmental mesh and remove inaccessible cells from the
        neighbour graph.

        """
        for cellbox in self.env_mesh.agg_cellboxes:
            access_values = self.vessel.model_accessibility(cellbox)
            self.env_mesh.update_cellbox(cellbox.id, access_values)
        inaccessible_nodes = [c.id for c in self.env_mesh.agg_cellboxes if not c.agg_data['accessible']]
        self.env_mesh.neighbour_graph = remove_nodes(self.env_mesh.neighbour_graph, inaccessible_nodes)

    def model_performance(self):
        """

        Method to calculate the relevant vessel performance values for each cell in the environmental mesh and update
        the mesh accordingly.

        """
        for cellbox in self.env_mesh.agg_cellboxes:
            performance_values = self.vessel.model_performance(cellbox)
            self.env_mesh.update_cellbox(cellbox.id, performance_values)

    def to_json(self):
        """
            Method to return the modified mesh in json format.

            Returns:
                j_mesh (dict): a dictionary representation of the modified mesh.
        """
        j_mesh = json.loads(json.dumps(self.env_mesh))
        return j_mesh

def remove_nodes(neighbour_graph, inaccessible_nodes):
    """
        Function to remove a list of inaccessible nodes from a given neighbour graph.

        Args:
            neighbour_graph (dict): A dictionary containing indexes of cellboxes and how they are connected

            {
                'index':{
                    '1':[index,...],
                    '2':[index,...],
                    '3':[index,...],
                    '4':[index,...],
                    '-1':[index,...],
                    '-2':[index,...],
                    '-3':[index,...],
                    '-4':[index,...]
                },
                'index':{...},
                ...
            }

            inaccessible_nodes (list): A list of indexes to be removed from the neighbour_graph

        Returns:
            accessibility_graph (dict): A new neighbour graph with the inaccessible nodes removed
    """
    logging.debug(f"Removing {len(inaccessible_nodes)} nodes from the neighbour graph")
    accessibility_graph = neighbour_graph.copy()

    for node in inaccessible_nodes:
        accessibility_graph.pop(node)

    for node in accessibility_graph.keys():
        for case in accessibility_graph[node].keys():
            for inaccessible_node in inaccessible_nodes:
                if int(inaccessible_node) in accessibility_graph[node][case]:
                    accessibility_graph[node][case].remove(int(inaccessible_node))

    return accessibility_graph