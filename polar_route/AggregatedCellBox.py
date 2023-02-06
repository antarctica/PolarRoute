from shapely.geometry import Polygon
import numpy as np
import pandas as pd
from polar_route.Boundary import Boundary


class AggregatedCellBox:
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
        
######## setters and getters ########
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


    def to_json(self):
        '''
            convert cellBox to JSON

            The returned object is of the form -

                {
                    "geometry" (String): POLYGON(...),\n
                    "cx" (float): ...,\n
                    "cy" (float): ...,\n
                    "dcx" (float): ...,\n
                    "dcy" (float): ..., \n
                    \n
                    "agg_value_1" (float): ...,\n
                    ...,\n
                    "agg_value_n" (float): ...
                }
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


   
