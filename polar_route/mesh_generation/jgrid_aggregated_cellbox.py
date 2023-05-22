

from shapely.geometry import Polygon
import numpy as np
import pandas as pd
from polar_route.mesh_generation.boundary import Boundary
from polar_route.mesh_generation.aggregated_cellbox import AggregatedCellBox
import logging
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
       self.node_string = str

    def get_node_string (self ):
       return self.node_string

    def set_Boundary(self, boundary):
        """
            set the boundary of the CellBox
        """
        self.boundary = boundary

    def set_agg_data (self, agg_data):
      
        self.agg_data = agg_data
    
    def set_id (self, id):
      
        self.id = id
    
    def get_boundary(self):
        """
            get the boundary of the CellBox
        """
        return self.boundary 

    def get_agg_data (self):
      
        return self.agg_data

    def get_id (self):
      
        return self.id

    def mesh_dump(self):
        logging.info ("dumping the mesh ...")
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
        mesh_dump += str(self.boundary.getcy()) + ", " + str(self.boundary.getcx()) + "; "  # add lat,long
     
        value = self.agg_data ['SIC']      
        number_of_points = self.agg_data["SIC_COUNT"]
        if value != None:
            ice_area = value 
      
        uc = self.agg_data['uC']*3.6  #unit conversion to match the Java code
        vc = self.agg_data['vC']*3.6 #unit conversion to match the Java code 

        mesh_dump += str(ice_area) + "; "  # add ice area
        # add uc, uv
        if np.isnan(uc) or np.isnan(vc): 
            uc = 0.0
            vc = 0.0 
        
        mesh_dump += str(uc) + ", " + str(vc) + ", "
        mesh_dump += str(number_of_points) 
        mesh_dump += "\n"

        return mesh_dump