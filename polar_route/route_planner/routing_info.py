import numpy as np


class RoutingInfo:
    """
        A class that represents routing information in terms of the cellbox index and the route segments that must be
        followed to reach this cellbox index.


        Attributes:
            node_indx (int): an int representing the node (cellbox) index
            path (list<Segment>): a list of segments that represents the path to reach the cellbox of the
            specified 'node_indx'
    """

    def __init__(self, indx, path):
        self.node_indx = indx
        self.path = path
    
    def get_path(self):
        """
            Returns the associated path
        """
        return self.path

    def get_path_nodes(self):
        """
            Gets a list of nodes visited along the associated path
        """
        points = []

        if len(self.path) > 0:
            points = [seg.get_end_wp().get_cellbox_indx() for seg in self.path]
            
        return points

    def get_node_index(self):
        """
            Returns the associated node index
        """
        return self.node_indx
    
    def get_obj(self, obj):
        if self.node_indx == -1: # this info means inaccessible node so the obj is infinity
            return np.inf
        
        obj_value = 0
        for segment in self.path:
            obj_value += getattr(segment, obj) # this should be recursive until the source wp
 
        return obj_value
    
    def to_str(self):
        print(f"To {self.path}, through node index: {self.node_indx}")
