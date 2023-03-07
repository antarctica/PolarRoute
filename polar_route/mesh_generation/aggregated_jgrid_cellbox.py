
from polar_route.mesh_generation.aggregated_cellBox import AggregatedCellBox

class JGridAggregatedCellBox (AggregatedCellBox):
    """
    a class represnts an aggrgated information within a geo-spatial/temporal boundary. 


    Attributes:
      

    Note:
        All geospatial boundaries of a CellBox are given in a 'EPSG:4326' projection
    """
    def __init__(self, boundary , agg_data , id):
        """

            Args:
                boundary(Boundary): encapsulates latitude, longtitude and time range of the CellBox
        """
        # Box information relative to bottom left
        self.boundary = boundary
        self.agg_data = agg_data
        self.id = id
        self.node_string =""
        
######## setters and getters ########
    def set_node_string (self , str):
       """
          sets node string
       """
       self.node_string = str

    def get_node_string (self ):
       """
       returns node string
       """
       return self.node_string

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
        sets id
        """
        self.id = id
    
    def get_boundary(self):
        """
            get the boundary of the CellBox
        """
        return self.boundary 

    def get_agg_data (self):
        """
        returns agg_data
        """
        return self.agg_data

    def get_id (self):
        """"
        returns the id
        """
        return self.id

def mesh_dump(self):
        """
            returns a string representing all the information stored in the mesh
            of this cellbox

            for use in j_grid regression testing
        """
        number_of_points=0
        mesh_dump = ""
        mesh_dump += self.get_node_string() + "; "  # add node string
        mesh_dump += "0 "
        ice_area = 0
        uc = None
        vc = None
        mesh_dump += str(self.bounds.getcy()) + ", " + str(self.bounds.getcx()) + "; "  # add lat,long

        for source in self.get_data_sources():
            loader = source.get_data_loader()
            if loader.get_data_name() =='SIC':
                value = self.agg_data ['SIC']
                # call the data_loader with COUNT as the agg_type to get the number of datapoints
                number_of_points = loader.get_value (self.bounds , "COUNT")['SIC']
                if value is not None:
                   ice_area = value
      
        uc = self.agg_data['uC']
        vc = self.agg_data['vC']

        mesh_dump += str(ice_area) + "; "  # add ice area
        # add uc, uv
        if (uc is None): 
            uc = 0
        if (vc is None):
             vc = 0
        
        mesh_dump += str(uc) + ", " + str(vc) + ", "
        mesh_dump += str(number_of_points) 
        mesh_dump += "\n"

        return mesh_dump
