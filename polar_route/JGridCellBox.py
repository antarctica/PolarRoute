

from shapely.geometry import Polygon
import numpy as np
import pandas as pd
from polar_route.Boundary import Boundary
from polar_route.AggregatedJGridCellBox import AggregatedJGridCellBox
from polar_route.cellbox import CellBox

class JGridCellBox (CellBox):
    """
    A JGridCellBox represnts a subclass of CellBox tailored to guarantee compatability with the earlier Java implemnation (which is based on coordinates and focus).


    Attributes:
        Bounds (Boundary): object that contains the latitude and logtitute range and the time range

    Note:
        All geospatial boundaries of a CellBox are given in a 'EPSG:4326' projection
    """
    

    def __init__(self, bounds , id  ):
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
        self.x_coord = 0
        self.y_coord = 0
        self.focus = ""
        self.land_locked = False
        self.initial_bounds = None
 

######################################################
# methods used for splitting and aggregating JGridCellBox

    def split(self , start_id):
        """
            splits the current cellbox into 4 corners, returns as a list of cellbox objects.
            args
            start_id : represenst the start of the splitted cellboxes ids, usuallly it is the number of the existing cellboxes
            Returns:
                split_boxes (list<CellBox>): The 4 corner cellboxes generates by splitting
                    this current cellboxes and dividing the data_points contained between.
        """
        # split using the CellBox method then perform the extra JGridCellBox logic
        split_boxes = CellBox.split(self) 

        # set CellBox split_depth, data_source and parent
        for split_box in split_boxes:
        # if parent box is land, all child boxes are considered land
          if self.land_locked:
                split_box.land_locked = True

          # set gridCoord of split boxes equal to parent.
          split_box.set_grid_coord(self.x_coord, self.y_coord)

        # create focus for split boxes.
          split_box.set_focus(self.get_focus().copy())
          split_box.add_to_focus(split_boxes.index(split_box))
        return split_boxes

#TODO: figure out the difference between this and cellbox aggregate??? use of Land_loader?
    def aggregate(self):
        '''
            aggregates JGridCellBox data using the associated data_source's aggregate type and returns AggregatedJGridCellBox object
            
        '''
     
        agg_dict = {}
        for source in self.get_data_source():
            loader = source.get_data_loader()
            agg_value = loader.get_value( self.bounds) # get the aggregated value from the associated DataLoader
            data_name = loader.data_name
            parent = self.get_parent()
            if ',' in data_name: # check if the data name has many entries (ex. uC,uV)
               agg_value = self.check_vector_data(source, loader, agg_value, data_name)     

            elif np.isnan(agg_value [data_name]) and source.get_value_fill_type()=='zero': #if the agg_value empty and get_value_fill_type is 0, then set agg_value to 0
                agg_value[data_name] = 0 
            elif np.isnan(agg_value [data_name]) and source.get_value_fill_type()=='parent': 
                while parent !=None and np.isnan(agg_value[data_name]):  #if the agg_value empty and get_value_fill_type is parent, then use the parent bounds
                    agg_value = loader.get_value( parent.bounds) 
                    parent = parent.get_parent()
               
             
            agg_dict.update (agg_value) # combine the aggregated values in one dict 

        agg_cellbox = AggregatedJGridCellBox (self.bounds , agg_dict , self.get_id())
        agg_cellbox.set_node_string(self.node_string())

        return agg_cellbox  

def set_grid_coord(self, xpos, ypos):
        """
            sets up initial grid-coordinate when creating a j_grid

            for use in j_grid impl
        """
        self.x_coord = xpos
        self.y_coord = ypos

def get_focus(self):
        """
            returns the focus of this cellbox

            for use in j_grid impl
        """
        return self.focus

def add_to_focus(self, focus):
        """
            append additional information to the focus of this cellbox
            to be used when splitting.

            for use in j_grid impl
        """
        self.focus.append(focus)

def grid_coord(self):
        """
            returns a string representation of the grid_coord of this cellbox

            for use in j_grid regression testing
        """
        return "(" + str(int(self.x_coord)) + "," + str(int(self.y_coord)) + ")"

def node_string(self):
        """
            returns a string representing the node of this cellbox


            for use in j_grid regression testing
        """
        node_string = self.grid_coord() + " F:" + str(len(self.get_focus()))

        focus_string = "["
        for focus in self.get_focus():
            focus_string += str(focus) + " "
        focus_string += "]"
        return node_string + " " + focus_string

def set_initial_bounds(self, parent):
     self.initial_bounds= parent


