   
class Waypoint:
    """
        Class representing a waypoint in terms of its latitude, longitude and name

        Attributes:
            lat (float): the waypoint latitude
            long (float): the waypoint longitude
            name (string) (optional): the waypoint name (ex. Falklands, Rothera)
            cellbox_indx (int): the index of the cellbox that contains this waypoint

        Note:
        All geospatial boundaries are given in a 'EPSG:4326' projection
    """
    @classmethod
    def load_from_cellbox(cls, cellbox):
        obj = Waypoint(cellbox.get_bounds().getcy(), cellbox.get_bounds().getcx())
        obj.set_cellbox_indx(str(cellbox.get_id()))
        return obj
    
    def __init__(self, lat, long, cellbox_indx=-1, name=None):
        self.lat = lat
        self.long = long 
        self.name = name
        self.cellbox_indx = cellbox_indx

    def get_latitude(self):
        """
            Returns waypoint latitude
        """
        return self.lat

    def get_longitude(self):
        """
            Returns waypoint longitude
        """
        return self.long

    def get_name(self):
        """
            Returns waypoint name
        """
        return self.name
    
    def get_cellbox_indx(self):
        """
            Returns the cellbox id that contains this waypoint
        """
        return self.cellbox_indx
    
    def set_cellbox_indx(self, i):
        """
            Sets the cellbox id that contains this waypoint
        """
        self.cellbox_indx = i

    def equals(self, wp):
        return self.lat == wp.get_latitude() and self.long == wp.get_longitude()

    def to_point(self):
        return [self.get_longitude(), self.get_latitude()]
