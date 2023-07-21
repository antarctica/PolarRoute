from polar_route.routing_info import RoutingInfo
from polar_route.waypoint import Waypoint


class SourceWaypoint (Waypoint):
    '''
        class derived from Waypoint and contains extra information for any source waypoint (routing information to other cellboxes and visited cellboxes)


        Attributes:
            visited_nodes: list<int>: a list contains the indices of the visited nodes 
            routing_table: dict<cellbox_indx, Routing_Info>: a dict that conatins teh routing information to reach cellbox_indx, works a routing table to reach the different cellboxex=s from this source waypoints

    '''

    def __init__(self, source, end_wps):
        """
            initializes a SourceWaypoint object from a Waypoint object
            Args:
                source(Waypoint): an object that encapsulates the latitude, longtitude, name and cellbox_id information
                end_wps (list <Waypoint>): list of the end waypoints
        """
        super().__init__(source.get_latitude(),source.get_longtitude(), source.get_name())
        self.cellbox_indx = source.get_cellbox_indx()
        self.end_wps = end_wps
        self.visited_nodes = []
        self.routing_table = {}
        # add routing information to itself, empty list of sepments as distance = 0
        self.routing_table[self.cellbox_indx] = RoutingInfo (self.cellbox_indx, []) 
        self.visited_nodes.append( self.cellbox_indx)


    def update_routing_table( self, indx, routing_info):
        self.routing_table[indx] = routing_info

    def visit_node( self, cellbox_indx):
        self.visited_nodes.append (cellbox_indx)

    def is_visited (self , indx): 
        return indx in  self.visited_nodes
    
    def is_all_visited(self):
        for wp in self.end_wps:
            if wp.get_cellbox_indx() not in self.visited_nodes:
                return False
        return True
    def get_routing_info(self, _id):
        if _id not in self.routing_table.keys():
            self.routing_table[_id] = RoutingInfo(-1, None) # indicating incaccessible node and returns infinity obj
            #raise ValueError ("There is no routing information ascosiated with cellbox {} ".format(_id))
        else: 
            return self.routing_table[_id]
