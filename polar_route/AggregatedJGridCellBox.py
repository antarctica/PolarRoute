

from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import Boundary
from polar_route.AggregatedCellBox import AggregatedCellBox

class AggregatedJGridCellBox (AggregatedCellBox):
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

#TODO: edit the logic to be based on the differnt boundaried for reading SIC and uc, vc
    def aggregate(self):
        '''
            aggregates CellBox data using the associated data_source's aggregate type and returns AggregatedCellBox object
            
        '''
     
        agg_dict = {}
        for source in self.get_data_source():
            agg_type = source.get_aggregate_type()
            agg_value = source.get_data_loader().get_value( self.bounds) # get the aggregated value from the associated DataLoader
            data_name = source.get_data_loader()._get_data_name()
            if (agg_value[data_name] == None and source.get_value_fill_type()=='parent'):  #if the agg_value empty and get_value_fill_type is parent, then use the parent bounds
               agg_value = source.get_data_loader().get_value( self.get_parent().bounds) 
            elif (agg_value[data_name] == None and source.get_value_fill_type()=='zero'): #if the agg_value empty and get_value_fill_type is 0, then set agg_value to 0
                agg_value = 0  
            else:
                 agg_value = np.nan
            agg_dict.update (agg_value) # combine the aggregated values in one dict 

        agg_cellbox = AggregatedCellBox (self.bounds , agg_dict , self.get_id())

        return agg_cellbox  

#TODO: check if the current data is empty and use the parent bounds??, where is_land  used?? , how the loader would work?? what is the diff between AggCellBox and JGridAggCellBox
#TODO: create a seperate ois_land data loader that specify if a land is in certain boundary based on cellbox set_land
def mesh_dump(self):
        """
            returns a string representing all the information stored in the mesh
            of this cellbox

            for use in j_grid regression testing
        """
        number_of_points=0
        mesh_dump = ""
        mesh_dump += self.node_string() + "; "  # add node string
        mesh_dump += "0 "
        ice_area = 0
        uc = None
        vc = None
        mesh_dump += str(self.bounds.getcy()) + ", " + str(self.bounds.getcx()) + "; "  # add lat,long
        # for source in self.get_data_sources():
        #     loader = source.get_data_loader()
        #     if loader.get_data_name() =='SIC':
        #         value = loader.get_value(self.bounds)
        #         number_of_points = loader.get_datapoints (self.bounds).size
        #         if value['SIC']!= None:
        #            ice_area = value['SIC']
        #     if loader.get_data_name() == 'current':
        #         current_data = loader.get_value (self.bounds)
        #         uc = current_data['uc']
        #         vc = current_data['vc']


        for source in self.get_data_sources():
            loader = source.get_data_loader()
            if loader.get_data_name() =='SIC':
                value = self.agg_data ['SIC']
                number_of_points = loader.get_datapoints (self.bounds).size
                if value != None:
                   ice_area = value
      
        uc = self.agg_data['uc']
        vc = self.agg_data['vc']

        mesh_dump += str(ice_area) + "; "  # add ice area
        # add uc, uv
        if (uc == None): 
            uc = 0
        if (vc == None):
             vc = 0
        
        mesh_dump += str(uc) + ", " + str(vc) + ", "
        mesh_dump += str(number_of_points) # TODO: double check 
        mesh_dump += "\n"

        return mesh_dump
