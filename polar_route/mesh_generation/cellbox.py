"""
Outlined in this section we will discuss the usage of the CellBox functionality of the PolarRoute package.
In this series of class distributions we house our discrete representation of input data. In each CellBox,
we represent a way of accessing the information governing our numerical world,
this includes and is not limited to: Ocean Currents, Sea Ice Concentration and Bathymetric depth.\n


    Example:\n
    An example of running this code can be executed by running the following in a ipython/Jupyter Notebook:: \n

            from polar_route.mesh_generation.cellbox import cellbox \n
            .... \n

    Note:\n
        CellBoxes are intended to be constructed by and used within a Mesh object.
          The methods provided are to extract information for CellBoxes contained within a Mesh. \n

"""


import numpy as np
from polar_route.mesh_generation.boundary import Boundary
from polar_route.mesh_generation.aggregated_cellbox import AggregatedCellBox


class CellBox:
    """
    A CellBox represnts a geo-spatial/temporal boundary that enables projecting to information within.
    Information about any given value of a CellBox is calculated from aggregating all data points of within those bounds.
    CellBoxes may  be split into smaller CellBoxes and the data points within distributed  between the newly created
    CellBoxes so as to construct a non-uniform mesh of CellBoxes, such as within a Mesh.\n

    Attributes:
        Bounds (Boundary): object that contains the latitude and logtitute range and the time range \n
        id (int):  the id of the cellbox \n

    """

    def __init__(self, bounds, id):
        """

            Args:
                bounds(Boundary): encapsulates latitude, longtitude and time range of the CellBox\n
                id (int):  the id of the cellbox \n
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
        if minimum_datapoints < 0:
            raise ValueError(
                f'CellBox: minimum number of data contained can not be negative')
        self.minimum_datapoints = minimum_datapoints

    def set_data_source(self, data_source):
        """
            a method that sets the data source of the cellbox ( which includes the data loaders,
              splitting conditions and aggregation type)

            Args:
                data_source (List <MetaData>): a list of MetaData objects, each object represents a source of this CellBox data
                  (where the data comes from, how it is spitted and aggregated)
        """
        self.data_source = data_source

    def set_parent(self, parent):
        """
            set the parent CellBox, which is the bigger CellBox that conains this CellBox
            Args:
                CellBox: the bigger Cellbox object that got splitted to produce this cellbox
        """
        self.parent = parent

    def set_split_depth(self, split_depth):
        """
            set the split depth of a CellBox, which represents is the number of times the CellBox
            has been split to reach it's current size.
        """
        if split_depth < 0:
            raise ValueError(f'CellBox: split depth can not be negative')
        self.split_depth = split_depth

    def set_id(self, id):
        """
        method ssts cellbox id
        """
        self.id = id

    def get_id(self):
        """
        method returns cellbox cell id
        """
        return self.id

    def get_minimum_datapoints(self):
        """
            get the minimum number of data contained within CellBox boundaries
        """
        return self.minimum_datapoints

    def get_data_source(self):
        """
            a method that gets the data source of the cellbox
              (the data loaders, splitting conditions and aggregation type)
            returns:
            data_source (List <MetaData>): a list of MetaData objects, each object represents a source
            of this CellBox data (where the data comes from, how it is spitted and aggregated)
        """
        return self.data_source

    def get_parent(self):
        """
            get the parent CellBox, which is the bigger CellBox that conains this CellBox
        """
        return self.parent

    def get_bounds(self):
        """
            get the spatial and temporal bounds (lat range, long range and time range) of this cellbox
        """
        return self.bounds

    def get_split_depth(self):
        """
            get the split depth of a CellBox, which represents is the number of times the CellBox
              has been split to reach it's current size.
        """
        return self.split_depth

######################################################
# methods used for splitting a cellbox

    def should_split(self, stop_index):
        """
            determines if a cellbox should be split based on the homogeneity
            condition of each data type contained within. The homogeneity condition
            of values within this cellbox is calculated using the method
            'get_hom_cond' in each DataLoader object inside CellBox's metadata\n

            if ANY data returns 'HOM':
                do not split
            if ANY data returns 'MIN':
                do not split
            if ALL data returns 'CLR':
                do not split
            else (mixture of CLR & HET):
                split\n
            Args:
                stop_index: the index of the data source at which checking the splitting conditions stops.
                Implemented like this to perform depth-first splitting.
                Should be deprecated once we switch to breadth-first splitting
            Returns:
                 bool: True if the splitting_conditions of this CellBox
                    will result in the CellBox being split.
        """
        hom_conditions = []

        current_data_source = None
        for index in range(0, stop_index):
            current_data_source = self.get_data_source()[index]
            data_loader = current_data_source.get_data_loader()
            for splitting_cond in current_data_source.get_splitting_conditions():
                hom_cond = data_loader.get_hom_condition(
                    self.bounds, splitting_cond)
                hom_conditions.append(hom_cond)

        if "HOM" in hom_conditions:
            return False
        if "MIN" in hom_conditions:
            return False
        if hom_conditions.count("CLR") == len(hom_conditions):
            return False

        return True

    def should_split_breadth_first(self):
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
                hom_cond = data_loader.get_hom_condition(
                    self.bounds, splitting_cond)
                hom_conditions.append(hom_cond)

        if "HOM" in hom_conditions:
            return False
        if "MIN" in hom_conditions:
            return False
        if hom_conditions.count("CLR") == len(hom_conditions):
            return False

        return True

    def split(self, start_id):
        """
            splits the current cellbox into 4 corners, returns as a list of cellbox objects.

            Args:
                start_id : represenst the start of the splitted cellboxes ids,
                  usuallly it is the number of the existing cellboxes
            Returns:
                 list<CellBox>: The 4 corner cellboxes generated by splitting
                 the cellbox uniformly.
        """
        split_boxes = self.create_splitted_cell_boxes(start_id)

        # set CellBox split_depth, data_source and parent
        for split_box in split_boxes:
            split_box.set_split_depth(self.get_split_depth() + 1)
            split_box.set_data_source(self.get_data_source())
            split_box.set_parent(self)

        return split_boxes

    def create_splitted_cell_boxes(self, index):
        """
        method that creates 4 splitted cellbox
        """
        half_width = self.bounds.get_width() / 2
        half_height = self.bounds.get_height() / 2

        # create 4 new cellboxes
        time_range = self.bounds.get_time_range()
        lat = self.bounds.get_lat_min()
        lat_range = [lat + half_height, lat + self.bounds.get_height()]
        long = self.bounds.get_long_min()
        long_range = [long, long + half_width]
        boundary = Boundary(lat_range, long_range, time_range)
        north_west = CellBox(boundary, str(index))

        lat_range = [lat + half_height, lat + self.bounds.get_height()]
        long_range = [long + half_width, long + self.bounds.get_width()]
        boundary = Boundary(lat_range, long_range, time_range)
        index += 1
        north_east = CellBox(boundary, str(index))

        lat_range = [lat, lat + half_height]
        long_range = [long, long + half_width]
        boundary = Boundary(lat_range, long_range, time_range)
        index += 1
        south_west = CellBox(boundary, str(index))

        lat_range = [lat, lat + half_height]
        long_range = [long + half_width, long + self.bounds.get_width()]
        boundary = Boundary(lat_range, long_range, time_range)
        index += 1
        south_east = CellBox(boundary, str(index))

        split_boxes = [north_west, north_east, south_west, south_east]
        return split_boxes

    def aggregate(self):
        '''
            aggregates CellBox data using the associated data_sources' aggregate type (ex. MEAN, MAX)
            and returns AggregatedCellBox object

            Returns:
                AggregatedCellbox: object contains the aggregated data within cellbox bounds.
        '''
        agg_dict = {}
        for source in self.get_data_source():
            loader = source.get_data_loader()
            # get the aggregated value from the associated DataLoader
            agg_value = loader.get_value(self.bounds)
            data_name = loader.data_name
            parent = self.get_parent()
            # check if the data name has many entries (ex. uC,uV)
            if ',' in data_name:
                agg_value = self.check_vector_data(
                    source, loader, agg_value, data_name)
            elif np.isnan(agg_value[data_name]):
                if source.get_value_fill_type() == 'parent':
                    # if the agg_value empty and get_value_fill_type is parent, then use the parent bounds
                    while parent is not None and np.isnan(agg_value[data_name]):
                        agg_value = loader.get_value(parent.bounds)
                        parent = parent.get_parent()
                else:  # not parent, so either float or Nan so set the agg_Data to value_fill_type
                    agg_value[data_name] = source.get_value_fill_type()

            # combine the aggregated values in one dict
            agg_dict.update(agg_value)

        agg_cellbox = AggregatedCellBox(self.bounds, agg_dict, self.get_id())
        # free the memory space used by the cellbox
        self.deallocate_cellbox()
        return agg_cellbox

    def check_vector_data(self, source, loader, agg_value, data_name):
        """
        method that checks if the vector data is None and calls the parent get value
        """
        data_name_list = data_name.split(',')
        for name in data_name_list:
            parent = self.get_parent()
            if np.isnan(agg_value[name]):
                if source.get_value_fill_type() == 'parent':
                    # if the agg_value empty and get_value_fill_type is parent, then use the parent bounds
                    while parent is not None and np.isnan(agg_value[name]):
                        agg_value[name] = loader.get_value(parent.bounds)[name]
                        parent = parent.get_parent()
                else:  # not parent, so either float or Nan so set the agg_Data to value_fill_type
                    agg_value[data_name] = source.get_value_fill_type()
        return agg_value

    def deallocate_cellbox(self):
        """
        Method to free up the memory space allocated by the cellbox
        """
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
