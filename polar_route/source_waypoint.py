from polar_route.routing_info import RoutingInfo
from polar_route.waypoint import waypoint


class SourceWaypoint (waypoint):
    '''
        class derived from Waypoint and contains extra information for any source waypoint (routing information to other cellboxes and visited cellboxes)


        Attributes:
            visited_nodes: list<int>: a list contains the indices of the visited nodes 
            routing_table: dict<cellbox_indx, Routing_Info>: a dict that conatins teh routing information to reach cellbox_indx, works a routing table to reach the different cellboxex=s from this source waypoints

    '''

    def __init__(self, source):
        """
            initializes a SourceWaypoint object from a Waypoint object
            Args:
                source(Waypoint): an object that encapsulates the latitude, longtitude, name and cellbox_id information
        """
        super().__init__(source.get_latitude(),source.get_longtitude(), source.get_name())
        self.cellbox_id = source.get_cellbox_id()
        self.visited_nodes = []
        self.routing_table = {}
        # add routing information to itself, empty list of sepments as distance = 0
        self.routing_table[self.cellbox_id] = RoutingInfo (self.cellbox_id, []) 
        self.visited_nodes.append( self.cellbox_id)


    def update_routing_table( self, indx, routing_info):
        self.routing_table[indx] = routing_info

    def visit_node( self, cellbox_id):
        self.visited_nodes.append (cellbox_id)

    def is_visited ( self , indx): 
        return indx in  self.visited_nodes
