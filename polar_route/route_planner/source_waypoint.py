from polar_route.route_planner.routing_info import RoutingInfo
from polar_route.route_planner.waypoint import Waypoint
import numpy as np
import logging


class SourceWaypoint(Waypoint):
    """
        Class derived from Waypoint that contains extra information for any source waypoint (routing information to
        other cellboxes and any visited cellboxes)

        Attributes:
            visited_nodes: set<int>: a set containing the indices of the visited nodes
            routing_table: dict<cellbox_indx, Routing_Info>: a dict that contains the routing information to reach
            cellbox_indx, works a routing table to reach the different cellboxes from this source waypoint
    """

    def __init__(self, source, end_wps):
        """
            Initializes a SourceWaypoint object from a Waypoint object
            Args:
                source(Waypoint): an object that encapsulates the latitude, longitude, name and cellbox_id information
                end_wps (list <Waypoint>): list of the end waypoints
        """
        super().__init__(source.get_latitude(), source.get_longitude(), name=source.get_name())
        self.cellbox_indx = source.get_cellbox_indx()
        self.end_wps = end_wps
        self.visited_nodes = set()
        self.routing_table = dict()
        # add routing information to itself, empty list of segments as distance = 0
        self.routing_table[self.cellbox_indx] = RoutingInfo(self.cellbox_indx, [])

    def update_routing_table(self, indx, routing_info):
        """
        Updates the source waypoint's routing table for a particular node with the given routing info
        Args:
            indx (str): the index of the cell to update
            routing_info (RoutingInfo): the routing info to be added
        """
        self.routing_table[indx] = routing_info

    def visit(self, cellbox_indx):
        """
        Marks the input cellbox as visited by adding its index to the set of visited nodes
        Args:
            cellbox_indx (str): the index of the visited cellbox
        """
        self.visited_nodes.add(cellbox_indx)

    def is_visited(self, indx):
        """
        Check if the node with the given index has been visited
        Args:
            indx (int): the index of the node to check
        """
        return str(indx) in self.visited_nodes
    
    def is_all_visited(self):
        """
        Check if all associated destination waypoints have been visited
        Returns:
            True if all have been visited and False if not

        """
        for wp in self.end_wps:
            if str(wp.get_cellbox_indx()) not in self.visited_nodes:
                return False
        return True

    def get_routing_info(self, _id):
        if _id not in self.routing_table.keys():
            self.routing_table[_id] = RoutingInfo(-1, None) # indicating inaccessible node and returns infinity obj
        return self.routing_table[_id]

    def get_path_nodes(self, _id):
        """
        Gets all nodes on the path from the source waypoint to the node at _id
        """
        if _id not in self.routing_table.keys():
            return []
        else:
            node_id = _id
            path_index = list()
            while node_id != self.cellbox_indx:
                node_indices = self.routing_table[node_id].get_path_nodes()
                path_index.insert(0, node_indices[1])
                node_id = node_indices[0]

            path_index.insert(0, self.cellbox_indx)


            return path_index

    def log_routing_table(self):
        logging.debug(f'Routing table of {self.cellbox_indx} source waypoint:')
        for x in self.routing_table.keys():
            logging.debug(f"To {x}, through node_idx: {self.routing_table[x].get_node_index()}")

    def log_detailed_routing_info(self):
        logging.debug(f'Routing table of {self.cellbox_indx} source waypoint:')
        for x in self.routing_table.keys():
            logging.debug(f"To {x}, through node_idx: {self.routing_table[x].get_node_index()}")
            logging.debug("using segments >> ")
            for s in self.routing_table[x].get_path():
                logging.debug(s.to_str())

    def get_obj(self, node_indx, obj):
        """
        Get the value of the objective function up to the specified node index
        Args:
            node_indx (str): the index along the path to calculate the value up to
            obj (str): the variable name corresponding to the objective function

        Returns:
            obj_value (float): the value of the objective function at the specified index along the route

        """
        if node_indx not in self.routing_table.keys():
            # This info means the node is inaccessible so the value of the objective function is infinity
            obj_value = np.inf
            return obj_value
        
        obj_value = 0
        # Sum segment values recursively until the source waypoint is reached
        for segment in self.routing_table[node_indx].get_path():
            obj_value += getattr(segment, obj)

        through_indx = self.routing_table[node_indx].node_indx
        # Search recursively and sum up the remaining segments until we reach the source waypoint
        while through_indx != self.cellbox_indx:
                for segment in self.routing_table[through_indx].get_path():
                        obj_value += getattr(segment, obj)
                through_indx = self.routing_table[through_indx].node_indx 
 
        return obj_value
