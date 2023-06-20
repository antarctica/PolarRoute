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