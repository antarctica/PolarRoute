


from datetime import datetime
from datetime import timedelta

from math import cos, sin, asin, sqrt, radians

class Boundary:
    """
    A Boundary is a class that defines the geo-spatial/temporal
    boundaries (longtitude, latitude and time).


    Attributes:
        lat_range (float[]): array contains the start and end of latitude range 
        long_range (float[]): array contains the start and end of longtitude range.
          In the case of constructing a global mesh, the longtitude range should be -180:180.
        time_range(string[]): array contains the start and end of time range 
        

    Note:
        All geospatial boundaries are given in a 'EPSG:4326' projection
    """ 
    @classmethod
    def from_json(cls, config):
        """
             constructs a boundary object from json input
            Args:
               config (json): json object that contains the boundary attributes
                
        """
        long_min = config['Mesh_info']['Region']['longMin']
        long_max = config['Mesh_info']['Region']['longMax']
        lat_min = config['Mesh_info']['Region']['latMin']
        lat_max = config['Mesh_info']['Region']['latMax']
        start_time = config['Mesh_info']['Region']['startTime']
        end_time = config['Mesh_info']['Region']['endTime']
        lat_range = [lat_min, lat_max]
        long_range = [long_min , long_max]
        time_range = [start_time , end_time]
        obj = Boundary (lat_range , long_range , time_range)
        return obj



    def __init__(self, lat_range , long_range , time_range=None):
        """

            Args:
               lat_range (float[]): array contains the start and end of latitude range 
               long_range (float[]): array contains the start and end of longtitude range 
               time_range(Date[]): array contains the start and end of time range 
                
        """
        if time_range is None:
            time_range=[]
        else: 
             time_range[0] = self.parse_datetime(time_range[0])
             time_range[1] = self.parse_datetime(time_range[1])

        self.validate_bounds(lat_range , long_range , time_range)
        # Boundary information 
        self.lat_range = lat_range
        self.long_range = long_range
        self.time_range = time_range

    def parse_datetime(self, datetime_str: str):
        """
            Attempts to parse a string containing reference to system time into datetime format.
            If given the string 'TODAY', will return system time.
            special characters '+' and '-' can be used to adjust system time. e.g 'TODAY + 3' 
            will return system time + 3 days, 'TODAY - 16' will return system time - 16 days
            
            Args:
                datetime_str (String): String attempted to be parsed to datetime format. 
                    Expected input format is '%d%m%Y'
            Returns:
                date (String): date in a String format, '%Y-%m-%d'.
            Raises:
                ValueError : If given 'datetime_str' cannot be parsed, raises ValueError.
        """
        DATE_FORMAT = "%Y-%m-%d"
        
        datetime_str = datetime_str.upper()

        # check if datetime_str is valid datetime format.
        try:
            datetime_object = datetime.strptime(datetime_str, DATE_FORMAT).date()
            return datetime_object.strftime(DATE_FORMAT)
        except ValueError:
            # check if datetime_str contains reference to system-time.
            if datetime_str.strip() == "TODAY":
                today = datetime.today()
                return today.strftime(DATE_FORMAT)
            elif "TODAY" not in datetime_str:
                raise ValueError(f'Incorrect date format given. Cannot convert "{datetime_str}" to date.')
            
                # check for increment to system time.
            if "+" in datetime_str:
                increment = datetime_str.split("+")[1].strip()
                try:
                    increment = float(increment)
                    datetime_object = datetime.today() + timedelta(days = increment)
                    return datetime_object.strftime(DATE_FORMAT)
                except ValueError:
                    raise ValueError(f'Incorrect date format given. Cannot convert "{datetime_str}" to date. ' + \
                        f'Time increment "{increment}" cannot be cast to float')
                
                # check for decrement to system time.
            elif "-" in datetime_str:
                decrement = datetime_str.split("-")[1].strip()
                try:
                    decrement = float(decrement)
                    datetime_object = datetime.today() - timedelta(days = decrement)
                    return datetime_object.strftime(DATE_FORMAT)
                except ValueError:
                    raise ValueError(f'Incorrect date format given. Cannot convert "{datetime_str}" to date. ' + \
                            f'Time decrement "{decrement}" cannot be cast to float')
            else:
                raise ValueError(f'Incorrect date format given. Cannot convert "{datetime_str}" to date')
        

    def validate_bounds (self, lat_range , long_range , time_range):
        """
            method to check the bounds are valid
            Args:
               lat_range (float[]): array contains the start and end of latitude range 
               long_range (float[]): array contains the start and end of longtitude range 
               time_range(Date[]): array contains the start and end of time range 
                
        """
        if len(lat_range) < 2 or len (long_range)<2 :
            raise ValueError('Boundary: range should contain two values')
        if lat_range[0] > lat_range [1]:
             raise ValueError('Boundary: Latitude start range should be smaller than range end')
        if long_range[0] > long_range [1]:
             raise ValueError('Boundary: Longtitude start range should be smaller than range end')
        if long_range[0] < -180 or long_range[1] > 180:
            raise ValueError('Boundary: Longtitude range should be within -180:180')
        if len (time_range) > 0:
            if datetime.strptime(time_range[0], '%Y-%m-%d') >= datetime.strptime(time_range[1], '%Y-%m-%d'):
                     raise ValueError('Boundary: Start time range should be smaller than range end')

    # Functions used for getting data from a cellbox
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
        """
            returns height of the cellbox

            Returns:
                height (float): the height of the CellBox
                    given in degrees latitude.
        """
        height = self.lat_range[1] - self.lat_range[0]
        return height
    def get_width(self):
        """
            returns width of the cellbox

            Returns:
                width (float): the width of the CellBox
                    given in degrees longtitude.
        """
        width = self.long_range[1] - self.long_range[0]
        return width
    def get_time_range (self):
        """
            returns the time range
        """
        return self.time_range
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
        """
            returns the min latitude
        """
        return self.lat_range[0]
    def get_lat_max(self):
        """
            returns the max latitude
        """
        return self.lat_range[1]   
    def get_long_min(self):
        """
            returns the min longtitude
        """
        return self.long_range[0]
    def get_long_max(self):
        """
            returns the max longtitude
        """
        return self.long_range[1]  
    def get_time_min(self):
        """
            returns the min of time range
        """
        return self.time_range[0]

    def get_time_max(self):
        """
            returns the max of time range
        """

        return self.time_range[1] 
    def get_bounds(self):
        """
            returns the bounds of this cellbox

            Returns:
                bounds (list<tuples>): The geo-spatial boundaries of this CellBox.
        """
        bounds = [[ self.long_range[0], self.lat_range[0] ],
                   [ self.long_range[0], self.lat_range[1]],
                    [ self.long_range[1], self.lat_range[1]],
                    [ self.long_range[1], self.lat_range[0]],
                    [self.long_range[0], self.lat_range[0], ]]
        return bounds
    
    def calc_size(self):
        """
        Calculate the great circle distance (in meters) between 
        two points on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [self.get_long_min(), 
                                               self.get_lat_min(),
                                               self.get_long_max(),
                                               self.get_lat_max()])
        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        # Get diagonal length
        m = (6371 * c * 1000)
        # Divide by sqrt(2) to get 'square' side length
        return m / sqrt(2)

    def __str__(self):


        lat_range = "lat range :[" + str(self.get_lat_min()) + \
              "," + str(self.get_lat_max()) + "]"
        long_range = "long range :[" + str(self.get_long_min()) + \
              "," + str(self.get_long_max()) + "]"
        time_range = "time range :" + str(self.get_time_range())

        return "{"+ lat_range + ", " + long_range + ", " + time_range + "}"


