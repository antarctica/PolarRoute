from polar_route.routing_info import RoutingInfo
from polar_route.waypoint import waypoint


class SourceWaypoint (waypoint):
    '''
        class derived from Waypoint and contains extra information for any source waypoint (routing information to other cellboxes and visited cellboxes)


        Attributes:
            visited_nodes: list<int>: a list contains the indices of the visited nodes 
            routing_table: dict<cellbox_indx, Routing_Info>: a dict that conatins teh routing information to reach cellbox_indx, works a routing table to reach the different cellboxex=s from this source waypoints

    '''

    def __init__(self, lat,long, name =None):
        super().__init__(lat,long, name )
        self.visited_nodes = []
        self.routing_table = {}
        # add routing information to itself, empty list of sepments as distance = 0
        self.routing_table[self.cellbox_indx] = RoutingInfo (self.cellbox_indx , []) 
        self.visited_nodes.append( self.cellbox_indx)


    def update_routing_table( self, indx, routing_info):
        self.routing_table[indx] = routing_info

    def visit_node( self, cellbox_indx):
        self.visited_nodes.append (cellbox_indx)

    def is_visited ( self , indx): 
        return indx in  self.visited_nodes
