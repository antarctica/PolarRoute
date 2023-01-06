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

from shapely.geometry import Polygon
import numpy as np
import pandas as pd
from polar_route.Boundary import Boundary
from polar_route.AggregatedCellBox import AggregatedCellBox
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



    def aggregate(self):
        '''
            aggregates CellBox data using the associated data_source's aggregate type and returns AggregatedCellBox object
            
        '''
     
        agg_dict = {}
        for source in self.get_data_source():
            agg_type = source.get_aggregate_type()
            agg_value = source.get_data_loader().get_value( self.bounds) # get the aggregated value from the associated DataLoader
            data_name = source.get_data_loader()._get_data_name()
            #TODO: replace None with something else after discussing with Harry the return of get_value if there is no data wthin bounds
            if (agg_value[data_name] == None and source.get_value_fill_type()=='parent'):  #if the agg_value empty and get_value_fill_type is parent, then use the parent bounds
               agg_value = source.get_data_loader().get_value( self.get_parent().bounds) 
            elif (agg_value[data_name] == None and source.get_value_fill_type()=='zero'): #if the agg_value empty and get_value_fill_type is 0, then set agg_value to 0
                agg_value = 0  
            else:
                 agg_value = np.nan
            agg_dict.update (agg_value) # combine the aggregated values in one dict 

        agg_cellbox = AggregatedCellBox (self.bounds , agg_dict , self.get_id())

        return agg_cellbox  

def set_grid_coord(self, xpos, ypos):
        """
            sets up initial grid-coordinate when creating a j_grid

            for use in j_grid regression testing
        """
        self.x_coord = xpos
        self.y_coord = ypos

def get_focus(self):
        """
            returns the focus of this cellbox

            for use in j_grid regression testing
        """
        return self.focus

def add_to_focus(self, focus):
        """
            append additional information to the focus of this cellbox
            to be used when splitting.

            for use in j_grid regression testing
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

#TODO: check if the current data is empty and use the parent bounds
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
        for source in self.get_data_sources():
            loader = source.get_data_loader()
            if loader.get_data_name() =='SIC':
                value = loader.get_value(self.bounds)
                number_of_points = loader.get_datapoints (self.bounds).size
                if value['SIC']!= None:
                   ice_area = value['SIC']
            if loader.get_data_name() == 'current':
                current_data = loader.get_value (self.bounds)
                uc = current_data['uc']
                vc = current_data['vc']

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
