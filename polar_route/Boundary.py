




class Boundary:
    """
    A Boundary is a class that defines the geo-spatial/temporal
    boundaries (longtitude, latitude and time).


    Attributes:
        lat_range (float[]): array contains the start and end of latitude range 
        long_range (float[]): array contains the start and end of longtitude range 
        time_range(Date[]): array contains the start and end of time range 
        

    Note:
        All geospatial boundaries are given in a 'EPSG:4326' projection
    """
   

    def __init__(self, lat_range , long_range , time_range):
        """

            Args:
               lat_range (float[]): array contains the start and end of latitude range 
               long_range (float[]): array contains the start and end of longtitude range 
               time_range(Date[]): array contains the start and end of time range 
                
        """
        # Boundary information 
        self.lat_range = lat_range
        self.long_range = long_range
        self.time_range = time_range


    # Functions used for getting data from a cellBox
    def getcx(self):
        """
            returns x-position of the centroid of the cellbox

            Returns:
                cx (float): the x-position of the top-left corner of the CellBox
                    given in degrees longitude.
        """
        
        return self.long_range[0] + self.get_width()/2


    def getcy(self):
        """
            returns y-position of the centroid of the cellbox

            Returns:
                cy (float): the y-position of the top-left corner of the CellBox
                    given in degrees latitude.
        """
        
        return self.lat_range[0] + self.get_height()/2

    def get_height(self):
        height = self.lat_range[1] - self.lat_range[0]

    def get_width(self):
        width = self.long_range[1] - self.long_range[0]
        return width

    def getdcx(self):
        """
            returns x-distance from the edge to the centroid of the cellbox

            Returns:
                dcx (float): the x-distance from the edge of the CellBox to the 
                    centroid of the CellBox. Given in degrees longitude
        """
        return self.get_width()/2

    def getdcy(self):
        """
            returns y-distance from the edge to the centroid of the cellbox

            Returns:
                dxy (float): the y-distance from the edge of the CellBox to the
                    centroid of the CellBox. Given in degrees latitude
        """
        return self.get_height()/2

    def get_lat_min(self):
        return self.lat_range[0]

    def get_lat_max(self):
        return self.lat_range[1]   

    def get_long_min(self):
        return self.long_range[0]

    def get_long_max(self):
        return self.long_range[1]  

    def get_bounds(self):
        """
            returns the bounds of this cellbox

            Returns:
                bounds (list<tuples>): The geo-spatial boundaries of this CellBox.
        """
        bounds = [[self.lat_range[0] , self.long_range[0]],
                   [self.lat_range[1], self.long_range[0]],
                    [self.lat_range[1], self.long_range[1]],
                    [self.lat_range[0], self.long_range[1]],
                    [self.lat_range[0], self.long_range[0]]]
        return bounds
