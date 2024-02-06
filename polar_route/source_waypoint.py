from polar_route.routing_info import RoutingInfo
from polar_route.waypoint import Waypoint
import numpy as np
import logging


class SourceWaypoint(Waypoint):
    """
        Class derived from Waypoint that contains extra information for any source waypoint (routing information to
        other cellboxes and any visited cellboxes)

        Attributes:
            visited_nodes: list<int>: a list contains the indices of the visited nodes
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
        self.visited_nodes = []
        self.routing_table = dict()
        # add routing information to itself, empty list of segments as distance = 0
        self.routing_table[self.cellbox_indx] = RoutingInfo (self.cellbox_indx, [])

    def update_routing_table(self, indx, routing_info):
        self.routing_table[indx] = routing_info

    def visit(self, cellbox_indx):
        self.visited_nodes.append (cellbox_indx)

    def is_visited(self, indx):
        return str(indx) in self.visited_nodes
    
    def is_all_visited(self):
        # print ("visited >>> ", self.visited_nodes)
        for wp in self.end_wps:
            if str(wp.get_cellbox_indx()) not  in self.visited_nodes:
                return False
        return True

    def get_routing_info(self, _id):
        if _id not in self.routing_table.keys():
            self.routing_table[_id] = RoutingInfo(-1, None) # indicating inaccessible node and returns infinity obj
        return self.routing_table[_id]

    def print_routing_table(self):
        logging.debug(f'Routing table of {self.cellbox_indx} source waypoint:')
        for x in self.routing_table.keys():
            logging.debug(f"To {x}, through node_idx: {self.routing_table[x].get_node_index()}")

    def print_detailed_routing_info(self):
        logging.debug(f'Routing table of {self.cellbox_indx} source waypoint:')
        for x in self.routing_table.keys():
            logging.debug(f"To {x}, through node_idx: {self.routing_table[x].get_node_index()}")
            logging.debug("using segments >> ")
            for s in self.routing_table [x].get_path():
                logging.debug(s.to_str())

    def get_obj(self, node_indx, obj):
        # print (self.node_indx)
        # print (self.path)
        if node_indx not in self.routing_table.keys(): # this info means inaccessible node so the obj is infinity
            return np.inf
        
        obj_value =0
        for segment in self.routing_table [node_indx].get_path():
            obj_value +=  getattr(segment, obj) # this should be recursive until the source wp is reached

        through_indx = self.routing_table[node_indx].node_indx
        while  through_indx!= self.cellbox_indx: # should search recursively and sum up the remaining segments until we reach the s_wp
                for segment in self.routing_table [through_indx].get_path():
                        obj_value +=  getattr(segment, obj)   
                through_indx = self.routing_table[through_indx].node_indx 
 
        return obj_value
