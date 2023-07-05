   
class waypoint:
    '''
        class represents a waypoint in terms of its latitude, longtitude and name


        Attributes:
            lat(float): the waypoint latitiude 
            long(float): the waypoint longititude 
            name(string) (optional): the waypoint name (ex. Falklands, Rothera)
        
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
    
    def get_name (self):
        '''
            returns the cellbox index that contains this waypoint 
        '''
        return self.cellbox_indx
