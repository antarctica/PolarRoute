
from polar_route.mesh_generation.boundary import Boundary
from polar_route.mesh_generation.jgrid_aggregated_cellbox import JGridAggregatedCellBox
from polar_route.mesh_generation.cellbox import CellBox
import numpy as np


class JGridCellBox (CellBox):
    """
    A JGridCellBox represnts a subclass of CellBox tailored to guarantee compatability with the earlier Java implemnation (which is based on coordinates and focus).


    Attributes:
        Bounds (Boundary): object that contains the latitude and logtitute range and the time range

    Note:
        All geospatial boundaries of a CellBox are given in a 'EPSG:4326' projection
    """

    def __init__(self, bounds, id):
        """

            Args:
                bounds(Boundary): encapsulates latitude, longtitude and time range of the CellBox
        """
        # Box information relative to bottom left
        self.bounds = bounds
        self.parent = None
        self.minimum_datapoints = 3000
        self.split_depth = 0
        self.data_source = None
        self.id = id
        self.x_coord = 0
        self.y_coord = 0
        self.focus = []
        self.land_locked = False
        self.initial_bounds = None

    @classmethod
    def init_from_cellbox(cls , cellbox):
        obj = JGridCellBox (cellbox.bounds , cellbox.id)
        obj.parent = cellbox.parent
        obj.split_depth = cellbox.split_depth
        obj.data_source = cellbox.data_source
        obj.id = cellbox.id
        return obj

    def split(self, start_id):
        """
            splits the current cellbox into 4 corners, returns as a list of cellbox objects.
            args
            start_id : represenst the start of the splitted cellboxes ids, usuallly it is the number of the existing cellboxes
            Returns:
                split_boxes (list<CellBox>): The 4 corner cellboxes generates by splitting
                    this current cellboxes and dividing the data_points contained between.
        """
        # split using the CellBox method then perform the extra JGridCellBox logic
        split_boxes = CellBox.split(self, start_id)
        jgrid_split_boxes = []
        # set CellBox split_depth, data_source and parent
        for split_box in split_boxes:
            index = split_boxes.index(split_box)
            jgrid_split_box = JGridCellBox.init_from_cellbox(split_box)
            # set gridCoord of split boxes equal to parent.
            jgrid_split_box.set_grid_coord(self.x_coord, self.y_coord)
            # create focus for split boxes.
            jgrid_split_box.set_focus(self.get_focus().copy())
            jgrid_split_box.add_to_focus(index)
            jgrid_split_box.set_initial_bounds (self.initial_bounds)
            jgrid_split_boxes.append(jgrid_split_box)
        return jgrid_split_boxes

    def aggregate(self):
        '''
            aggregates JGridCellBox data using the associated data_source's aggregate type and returns AggregatedJGridCellBox object

        '''
        agg_dict = {}
        for source in self.get_data_source():
            loader = source.get_data_loader()

            data_name = loader.data_name
            parent = self.get_parent()
            # check if the data name has many entries (ex. uC,vC)
            if ',' in data_name:
               
                agg_value = loader.get_value(self.initial_bounds)
            else:
                # get the aggregated value from the associated DataLoader
                agg_value = loader.get_value(self.bounds)
                if np.isnan(agg_value[data_name]):
                     if source.get_value_fill_type() == 'parent':
                  		# if the agg_value empty and get_value_fill_type is parent, then use the parent bounds
                        while parent is not None and np.isnan(agg_value[data_name]):
                                     agg_value = loader.get_value(
                                         parent.bounds)
                                     parent = parent.get_parent()
                     else:# not parent, so either float or Nan so set the agg_Data to value_fill_type
                         agg_value[data_name] = source.get_value_fill_type()
            if data_name == "SIC":
                number_of_points = loader.get_value (self.bounds , "COUNT")['SIC']
                agg_dict.update ({"SIC_COUNT": number_of_points})
            agg_dict.update (agg_value) # combine the aggregated values in one dict 

        agg_cellbox = JGridAggregatedCellBox (self.bounds , agg_dict , self.get_id())
   
        agg_cellbox.set_node_string(self.get_node_string())
        # print (self.get_node_string())

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

    def get_node_string(self):
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

    def set_initial_bounds(self, bounds):
        """
                returns a string representation of the grid_coord of this cellbox

                for use in j_grid regression testing
        """
        self.initial_bounds= bounds

    def is_land(self):
        """
        checks if the current cellbox is land using the loader
        """
        is_land = False
        for source in self.data_source:
            loader = source.get_data_loader()
            data_name = 'is_land'
            if loader.data_name == data_name:
                is_land = loader.get_value (self.bounnds)[data_name]  
        return is_land
    
    def set_focus(self, focus):
        """
            initialize the focus of this cellbox
            for use in j_grid regression testing
        """
        self.focus = focus

        
    def contains_point(self, lat, long):
        """
            Returns true if a given lat/long coordinate is contained within this cellbox.

            Args:
                lat (float): latitude of a given point
                long (float): longitude of a given point

            Returns:
                contains_points (bool): True if this CellBox contains a point given by
                    parameters (lat, long)
        """
        if (lat >= self.bounds.get_lat_min()) & (lat <= self.bounds.get_lat_max()):
            if (long >= self.bounds.get_long_min()) & (long <= self.bounds.get_long_max()):
                return True
        return False
