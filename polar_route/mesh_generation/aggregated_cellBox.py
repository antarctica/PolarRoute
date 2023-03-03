from shapely.geometry import Polygon
from polar_route.mesh_generation.boundary import Boundary
import shapely.wkt


class AggregatedCellBox:
    """
    a class represnts an aggrgated information within a geo-spatial/temporal boundary. 


    Attributes:
      

    Note:
        All geospatial boundaries of a CellBox are given in a 'EPSG:4326' projection
    """
    @classmethod
    def from_json(cls, cellbox_json):
        """

            Args:
                cellbox_json(Json): json object that encapsulates boundary, agg_data and id of the CellBox
        """
        cellbox_id = cellbox_json ['id']
        def load_boundary (cellbox_json):
        
            shape = shapely.wkt.loads (cellbox_json ["geometry"])
            bounds = shape.bounds
            lat_range = [bounds[1] , bounds[3]]
            long_range = [bounds [0], bounds [2]]
            return Boundary (lat_range , long_range)

        def load_agg_data (cellbox_json):
            dict_obj = {}
            for key in cellbox_json:
                if key  not in [  "geometry","cx", "cy", "dcx", "dcy"]:
                    dict_obj[key] = cellbox_json[key]

            return dict_obj
        
        boundary = load_boundary( cellbox_json)
        agg_data = load_agg_data( cellbox_json)
        obj = AggregatedCellBox(boundary , agg_data ,cellbox_id )
        return obj




    def __init__(self, boundary , agg_data , id):
        """

            Args:
                boundary(Boundary): encapsulates latitude, longtitude and time range of the CellBox
                agg_data (dict): a dictionary that contains data_names and agg values
                id (string): a string represents cellbox id
        """
        # Box information relative to bottom left
        self.boundary = boundary
        self.agg_data = agg_data
        self.id = id
        
######## setters and getters ########
    def set_boundary(self, boundary):
        """
            set the boundary of the CellBox
        """
        self.boundary = boundary

    def set_agg_data (self, agg_data):
        """
        sets the agg_data
        """
        self.agg_data = agg_data
    
    def set_id (self, id):
        """
        sets the cellbox id
        """
        self.id = id
    
    def get_boundary(self):
        """
            get the boundary of the CellBox
        """
        return self.boundary 

    def get_agg_data (self):
        """
        returns the agg_data
        """
        return self.agg_data

    def get_id (self):
        """
        returns the id
        """
        return self.id


    def to_json(self):
        '''
            convert cellBox to JSON

            The returned object is of the form -

                {\n
                    "geometry" (String): POLYGON(...),\n
                    "cx" (float): ...,\n
                    "cy" (float): ...,\n
                    "dcx" (float): ...,\n
                    "dcy" (float): ..., \n
                    \n
                    "agg_value_1" (float): ...,\n
                    ...,\n
                    "agg_value_n" (float): ...\n
                }\n
            Returns:
                cell_json (dict): A JSON parsable dictionary representation of this AggregatedCellBox
        '''
        cell_json = {
            "geometry": str(Polygon(self.get_boundary().get_bounds())),
            'cx': float(self.get_boundary().getcx()),
            'cy': float(self.get_boundary().getcy()),
            'dcx': float(self.get_boundary().getdcx()),
            'dcy': float(self.get_boundary().getdcy()),
           
        }

        cell_json.update(self.get_agg_data())
        cell_json['id'] = self.get_id()
        
        return cell_json



    def contains_point(self, lat, long):
        """
            Returns true if a given lat/long coordinate is contained within this cellBox.

            Args:
                lat (float): latitude of a given point
                long (float): longitude of a given point

            Returns:
                contains_points (bool): True if this CellBox contains a point given by
                    parameters (lat, long)
        """
        if (lat >= self.boundary.get_lat_min()) & (lat <= self.boundary.get_lat_max()):
            if (long >= self.boundary.get_long_min()) & (long <= self.boundary.get_long_max()):
                return True
        return False
