"""
Outlined in this section we will discuss the usage of the CellBox functionality
of the PolarRoute package. In this series of class distributions we house our discrete
representation of input data. In each CellBox, we represent a way of accessing the information governing our numerical world, this includes and is not limited to:
Ocean Currents, Sea Ice Concentration and Bathymetric depth.

Example:
    An example of running this code can be executed by running the following
    in a ipython/Jupyter Notebook::

        from polar_route import cellbox
        ....

Note:
    CellBoxes are intended to be constructed by and used within a Mesh
    object. The methods provided are to extract information for CellBoxes
    contained within a Mesh.
"""
from memory_profiler import profile
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
from polar_route.Boundary import Boundary
from polar_route.AggregatedCellBox import AggregatedCellBox


class CellBox:
    """
    A CellBox represnts a geo-spatial/temporal boundary that enables projecting to information within. Information about any given value of a CellBox is calculated from 
    the mean of all data points of that type within those bounds. CellBoxes may
    be split into smaller CellBoxes and the data points within distributed
    between the newly created CellBoxes so as to construct a non-uniform mesh
    of CellBoxes, such as within a Mesh.


    Attributes:
        Bounds (Boundary): object that contains the latitude and logtitute range and the time range

    Note:
        All geospatial boundaries of a CellBox are given in a 'EPSG:4326' projection
    """
    

    def __init__(self, bounds , id ):
        """

            Args:
                bounds(Boundary): encapsulates latitude, longtitude and time range of the CellBox
        """
        # Box information relative to bottom left
        self.bounds = bounds
        self.parent = None
        self.minimum_datapoints = 10
        self.split_depth = 0
        self.data_source = None
        self.id = id
        
######## setters and getters ########
    def set_minimum_datapoints(self, minimum_datapoints):
        """
            set the minimum number of data contained within CellBox boundaries
        """
        if (minimum_datapoints < 0):
             raise ValueError(f'CellBox: minimum number of data contained can not be negative')
        self.minimum_datapoints = minimum_datapoints

    def set_data_source (self, data_source):
        """
            a method that sets the data source of the cellbox (the data loaders, splitting conditions and aggregation type)
            Args:
            data_source (List <MetaData>): a list of MetaData objects, each object represents a source of this CellBox data (where the data comes from, how it is spitted and aggregated)  
        """
        self.data_source = data_source
    

    def set_parent(self, parent):
        """
            set the parent CellBox, which is the bigger CellBox that conains this CellBox
        """
        self.parent = parent


    def set_split_depth(self, split_depth):
        """
            set the split depth of a CellBox, which represents is the number of times the CellBox has been split to reach it's current size.
        """
        if (split_depth < 0):
             raise ValueError(f'CellBox: split depth can not be negative')
        self.split_depth = split_depth

    def set_id (self , id):
         self.id = id

    def get_id (self ):
         return self.id 

    def get_minimum_datapoints(self):
        """
            get the minimum number of data contained within CellBox boundaries
        """
        return self.minimum_datapoints

    def get_data_source (self):
        """
            a method that gets the data source of the cellbox (the data loaders, splitting conditions and aggregation type)
            returns:
            data_source (List <MetaData>): a list of MetaData objects, each object represents a source of this CellBox data (where the data comes from, how it is spitted and aggregated)  
        """
        return self.data_source
    

    def get_parent(self):
        """
            get the parent CellBox, which is the bigger CellBox that conains this CellBox
        """
        return self.parent


    def get_split_depth(self):
        """
            get the split depth of a CellBox, which represents is the number of times the CellBox has been split to reach it's current size.
        """
        return self.split_depth

######################################################
# methods used for splitting a cellbox


    def should_split(self):
        """
            determines if a cellbox should be split based on the homogeneity
            condition of each data type contained within. The homogeneity condition
            of values within this cellbox is calculated using the method
            'get_hom_cond' in each DataLoader object inside CellBox's metadata

            if ANY data returns 'HOM':
                do not split
            if ANY data returns 'MIN':
                do not split
            if ALL data returns 'CLR':
                do not split
            else (mixture of CLR & HET):
                split

            Returns:
                should_split (bool): True if the splitting_conditions of this CellBox
                    will result in the CellBox being split.
        """
        hom_conditions = []
        for current_data_source in self.data_source:
            data_loader = current_data_source.get_data_loader()
            for splitting_cond in current_data_source.get_splitting_conditions():
               hom_cond = data_loader.get_hom_condition(self.bounds, splitting_cond)
               hom_conditions.append(hom_cond )
      
        if "HOM" in hom_conditions:
            return False
        if "MIN" in hom_conditions:
            return False
        if hom_conditions.count("CLR") == len(hom_conditions):
            return False

        return True

    def split(self , start_id):
        """
            splits the current cellbox into 4 corners, returns as a list of cellbox objects.
            args
            start_id : represenst the start of the splitted cellboxes ids, usuallly it is the number of the existing cellboxes
            Returns:
                split_boxes (list<CellBox>): The 4 corner cellboxes generated by splitting
                 the cellbox uniformly.
        """

        split_boxes = self.create_splitted_cell_boxes(start_id)

        # set CellBox split_depth, data_source and parent
        for split_box in split_boxes:
            split_box.set_split_depth ( self.get_split_depth() + 1)
            split_box.set_data_source (self.get_data_source())
            split_box.set_parent (self)

        return split_boxes

    def create_splitted_cell_boxes(self, index):
        half_width = self.bounds.get_width() / 2
        half_height = self.bounds.get_height() / 2

        # create 4 new cellBoxes
        time_range = self.bounds.get_time_range()
        lat = self.bounds.get_lat_min()
        lat_range = [lat + half_height , lat + self.bounds.get_height() ] 
        long = self.bounds.get_long_min()
        long_range = [long , long + half_width]
        boundary = Boundary (lat_range , long_range , time_range)
        north_west = CellBox(boundary , str(index) )

        lat_range = [lat + half_height , lat + self.bounds.get_height() ] 
        long_range = [long + half_width , long + self.bounds.get_width()]
        boundary = Boundary (lat_range , long_range , time_range)
        index +=1
        north_east = CellBox(boundary , str(index))

        lat_range = [lat, lat + half_height ] 
        long_range = [long, long + half_width]
        boundary = Boundary (lat_range , long_range , time_range)
        index +=1
        south_west = CellBox(boundary , str(index))

        lat_range = [lat, lat + half_height ] 
        long_range = [long + half_width, long + self.bounds.get_width()]
        boundary = Boundary (lat_range , long_range , time_range)
        index +=1
        south_east = CellBox(boundary , str(index))

        split_boxes = [north_west, north_east, south_west, south_east]
        return split_boxes
    
    def aggregate(self):
        '''
            aggregates CellBox data using the associated data_source's aggregate type and returns AggregatedCellBox object
            
        '''
     
        agg_dict = {}
        for source in self.get_data_source():
            loader = source.get_data_loader()
            agg_value = loader.get_value( self.bounds) # get the aggregated value from the associated DataLoader
            data_name = loader.data_name
            if ',' in data_name: # check if the data name has many entries (ex. uC,uV)
               data_name_list =  data_name.split(',')
               for name in data_name_list:
                  parent = self.get_parent() 
                  if np.isnan(agg_value[name]) and source.get_value_fill_type()=='zero': #if the agg_value empty and get_value_fill_type is 0, then set agg_value to 0
                        agg_value[name] = 0  
                  elif np.isnan(agg_value[name]) and source.get_value_fill_type()=='parent': 
                        while np.isnan(agg_value[name]): #if the agg_value empty and get_value_fill_type is parent, then use the parent bounds
                            agg_value [name] = loader.get_value( parent.bounds)[name]
                            parent = parent.get_parent()     

            elif np.isnan(agg_value [data_name]) and source.get_value_fill_type()=='zero': #if the agg_value empty and get_value_fill_type is 0, then set agg_value to 0
                agg_value[data_name] = 0 
            elif np.isnan(agg_value [data_name]) and source.get_value_fill_type()=='parent': 
                while parent !=None and np.isnan(agg_value[data_name]):  #if the agg_value empty and get_value_fill_type is parent, then use the parent bounds
                    agg_value = loader.get_value( parent.bounds) 
                    parent = parent.get_parent()
               
                 
           
            
            agg_dict.update (agg_value) # combine the aggregated values in one dict 

        agg_cellbox = AggregatedCellBox (self.bounds , agg_dict , self.get_id())
        # free the memory space used by the cellbox
        self.deallocate_cellbox()
        return agg_cellbox  

# Method to free up the memory space allocated by the cellbox
    
    def deallocate_cellbox(self):
        for source in self.get_data_source():
            loader = source.get_data_loader()
            del loader
            del source
        parent = self.parent
        # free up the memory space used by the parent cellboxes chain
        while isinstance(parent, CellBox):
            for source in self.parent.get_data_source():
                loader = source.get_data_loader()
                del loader
                del source
            grandparent = parent.parent
            del parent
            parent = grandparent
        del self