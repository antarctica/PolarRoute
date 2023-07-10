   
class waypoint:
    '''
        class represents a waypoint in terms of its latitude, longtitude and name


        Attributes:
            lat(float): the waypoint latitiude 
            long(float): the waypoint longititude 
            name(string) (optional): the waypoint name (ex. Falklands, Rothera)
            cellbox_indx(int): the index of the cellbox that contains this waypoint
        
        Note:
        All geospatial boundaries are given in a 'EPSG:4326' projection
    '''

    def __init__(self, lat,long, name =None):
        self.lat = lat
        self.long = long 
        self.name = name
        self.cellbox_indx = -1

    def get_latitude (self):
        '''
            returns waypoint latitude
        '''
        return self.lat

    def get_longtitude (self):
        '''
            returns waypoint longtitude
        '''
        return self.long

    def get_name (self):
        '''
            returns waypoint name
        '''
        return self.name
    
    def get_cellbox_indx(self):
        '''
            returns the cellbox id that contains this waypoint 
        '''
        return self.cellbox_indx
    
    def set_cellbox_indx (self , id):
        '''
            sets the cellbox id that contains this waypoint 
        '''
        self.cellbox_indx = id
