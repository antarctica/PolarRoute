import numpy as np 
class RoutingInfo:
    '''
        class represents routing information in terms of cellbox index and what are the path (route's segments ) to follow to reach this cellbox indx


        Attributes:
            node_indx(int): an int representing the node (cellbox) index 
            path (list<Segment>): a list of segments that represents the path to reach the cellbox of the specified 'node_indx' 
    '''

    def __init__(self, indx , path):
        self.node_indx = indx
        self.path = path
    
    def get_path(self):
        '''
            returns path
        '''
        return self.path
    
    def get_node_index(self):
        '''
            returns node index
        '''
        return self.node_indx
    
    def get_obj (self, obj):
        obj_value =0
        if obj == "distance":
            for segment in self.path:
                obj_value += segment.get_distance()
        elif obj =="fuel":
            for segment in self.path:
                obj_value += segment.get_fuel()
        elif obj =="travel_time":
            for segment in self.path:
                obj_value += segment.get_travel_time()

        return obj_value
