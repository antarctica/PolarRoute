"""
Outlined in this section we will discuss the usage of the CellBox functionality
of the PolarRoute package. In this series of class distributions we house our discrete
representation of input data. In each CellBox we determine the mean and variance of
the information governing our numerical world, this includes and is not limited to:
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


class CellBox:
    """
    A CellBox is a collection of data-points contained within a given geo-spatial/temporal
    boundary. Information about any given value of a CellBox is calculated from 
    the mean of all data points of that type within those bounds. CellBoxes may
    be split into smaller CellBoxes and the data points within distributed
    between the newly created CellBoxes so as to construct a non-uniform mesh
    of CellBoxes, such as within a Mesh.


    Attributes:
        lat (float): The latitude of the top-left corner of the CellBox
        long (float): The longitude of the top-left corner of the CellBox
        width (float): The width of the CellBox, given in degrees longitude
        height (float): The height of the CellBox, given in degrees latitude

    Note:
        All geospatial boundaries of a CellBox are given in a 'EPSG:4326' projection
    """
    split_depth = 0

    def __init__(self, lat, long, width, height, splitting_conditions=[], j_grid=False):
        """

            Args:
                lat (float): The latitude of the top-left corner of the CellBox
                long (float): The longitude of the top-left corner of the CellBox
                width (float): The width of the CellBox, given in degrees longitude
                height (float): The height of the CellBox, given in degrees latitude
                splitting_conditions (list<(dict)>): conditions in which the CellBox
                    will be split into 4 smaller CellBoxes.

                    splitting_conditions are of the form -
                        {
                            <value>:{
                                "threshold" (float):,\n
                                "upperBound" (float):,\n
                                "lowerBound" (float)
                            }
                        }

                j_grid (bool): True if the CellBox should be constructed using the
                    format of the original Java codebase
        """
        # Box information relative to bottom left
        self.lat = lat
        self.long = long
        self.width = width
        self.height = height

        self._data_points = pd.DataFrame()

        self.minimum_datapoints = 10
        self._splitting_conditions = splitting_conditions
        self._value_fill_types = dict()

        self._value_out_types = dict()

        # For initial implementation of land based from Java codebase.
        self._j_grid = j_grid
        self.land_locked = False
        self.grid_uc = 0
        self.grid_vc = 0
        self.x_coord = 0
        self.y_coord = 0
        self.focus = ""
        self.min_depth = 10
        self._current_points = pd.DataFrame()

    # Functions used for getting data from a cellBox
    def getcx(self):
        """
            returns x-position of the centroid of the cellbox

            Returns:
                cx (float): the x-position of the top-left corner of the CellBox
                    given in degrees longitude.
        """
        return self.long + self.width/2

    def getcy(self):
        """
            returns y-position of the centroid of the cellbox

            Returns:
                cy (float): the y-position of the top-left corner of the CellBox
                    given in degrees latitude.
        """
        return self.lat + self.height/2

    def getdcx(self):
        """
            returns x-distance from the edge to the centroid of the cellbox

            Returns:
                dcx (float): the x-distance from the edge of the CellBox to the 
                    centroid of the CellBox. Given in degrees longitude
        """
        return self.width/2

    def getdcy(self):
        """
            returns y-distance from the edge to the centroid of the cellbox

            Returns:
                dxy (float): the y-distance from the edge of the CellBox to the
                    centroid of the CellBox. Given in degrees latitude
        """
        return self.height/2

    def get_data_names(self):
        """
            Returns the data names of all values which have been added to this CellBox

            Returns:
                data_names (list<(String)>): A list of all the names of data types which
                    have been added to this CellBox
        """
        data_names = list(self._data_points.columns)

        to_remove = ['lat', 'long', 'time']
        for item in to_remove:
            data_names.remove(item)

        return data_names

    # TODO: getter / setter
    def get_data_points(self, values=[]):
        """
            Returns a dataframe of containing values specified in parameter 'values'.
            If values is empty, return a dataframe containing all datapoints within
            the CellBox.

            Args:
                values (list<string>): datapoints within the CellBox to be included
                    in the returned dataframe

            Returns:
                data_points (Dataframe): a dataframe of datapoints within the CellBox.
                The dataframe is of the form -

                    long | lat | time | value_1 | ... | value_n
        """
        if len(values) == 0:
            return self._data_points
        else:
            data_points = pd.DataFrame()

            # TODO: review, this looks rather inefficient through recursions
            for value in values:
                data_points = pd.concat(
                    [data_points, self.get_data_points().dropna(subset=[value])], axis=0)

            columns = ['lat', 'long', 'time'] + values
            return data_points[columns]

    # TODO: getter / setter
    # TODO: caller can aggregate, this looks like an anti-pattern
    def get_value(self, value_name, value_type="MEAN"):
        """
            returns the mean value of the datapoints within this cellbox
            specified by the parameter 'value'.

            Args:
                value_name (string): The value type requested

                value_type (string): The output type of a value requested.
                    value_type may be < MEAN | MIN | MAX >. If none is given
                    a default of MIN is used.

            Returns:
                value (float): The mean of all data_points of type 'value'
                    within this CellBox
        """
        data_frame = self.get_data_points(values=[value_name])

        if value_type == "MIN":
            value = data_frame[value_name].min()
        elif value_type == "MAX":
            value = data_frame[value_name].max()
        else:  # value_type == MEAN
            value = data_frame[value_name].mean()

        return value

    def get_bounds(self):
        """
            returns the bounds of this cellbox

            Returns:
                bounds (list<tuples>): The geo-spatial boundaries of this CellBox.
        """
        bounds = [[self.long, self.lat],
                    [self.long, self.lat + self.height],
                    [self.long + self.width, self.lat + self.height],
                    [self.long + self.width, self.lat],
                    [self.long, self.lat]]
        return bounds

    def get_value_out_types(self):
        """
            TODO
        """
        return dict(self._value_out_types)

    def get_value_fill_types(self):
        """
            TODO
        """
        return dict(self._value_fill_types)

    def set_minimum_datapoints(self, minimum_datapoints):
        """
            TODO
        """
        self.minimum_datapoints = minimum_datapoints

    # Functions used for adding data to a cellbox
    def add_data_points(self, new_data_points):
        """
            adds a dataframe containing datapoints to the dataframe
            of datapoints within this CellBox

            Args:
                new_data_points (Dataframe): A dataframe of data_points to be added
                to this CellBox. new_data_points must be of the format -

                    lat | long | (time)* | value_1 | ... | value_n
        """
        self._data_points = pd.concat([self._data_points, new_data_points], axis=0)

    def add_splitting_condition(self, splitting_condition):
        """
            adds a dictionary containing a splitting condition to the
            list of splitting conditions contained within this CellBox

            Args:
                splitting_condition (dict): a splitting condition to be added to
                this CellBox. splitting_condition must be of the form -

                splitting condition is of form:
                {<value>: {
                    'threshold': (float) ...,
                    'upperbound': (float) ...,
                    'lowerbound' (float) ...:
                }}
        """
        self._splitting_conditions = self._splitting_conditions + [splitting_condition]

    def add_value_output_type(self, value_out_type):
        """
            appends a dictionary mapping values in a cellbox to there output types to the cellboxes
            internal memory of values to output type mappings.

            Args:
                value_out_type (string): A dictionary containing a mapping of a value held within
                the cellbox to its output type. An output type may be either MEAN, MIN or MAX.
                If no output type is defined for a value in a cellbox, this defaults as MEAN.

                {
                    <value>: < MEAN | MIN | MAX >,
                    ...
                }
        """
        self._value_out_types.update(value_out_type)

    def set_value_fill_types(self, value_fill_types):
        """
            TODO
        """
        self._value_fill_types = value_fill_types


    # Functions used for splitting a cellbox
    def value_should_be_split(self, value, threshold, lowerbound, upperbound):
        """
            returns true or false depending on whether a splitting condition associated
            with parameter 'value' should cause the cellbox to be split according to the
            parameters 'threshold', 'upperbound' and 'lowerbound'.

            Args:
                value (string): the name of a value a splitting condition is checked against.
                threshold (float): The threshold at which data_points of type 'value' within
                    this CellBox are checked to be either above or below.
                lowerbound (float): The lowerbound of acceptable percentage of data_points of
                    type value within this CellBox that are above 'threshold'.
                upperbound (float): the upperbound of acceptable percentage of data_points of
                    type value within this CellBox that are above 'threshold'.

            Returns:
                should_be_split (bool): True if the splitting_condition given would result in
                    this CellBox being split.
        """
        data_limit = self.minimum_datapoints

        data_points = self.get_data_points(values=[value])

        if data_points.shape[0] < data_limit:
            return False

        prop_over = data_points.loc[data_points[value] > threshold]

        proportion_over_x_percent = prop_over.shape[0] / data_points.shape[0]
        return lowerbound < proportion_over_x_percent < upperbound

    def value_hom_condition(self, value, threshold, lowerbound, upperbound):
        """
            returns 'CLR', 'HET' or 'HOM' dependant on the distribution of
            datapoints contained within.

            Args:
                value (string): the name of a value a splitting condition is checked against.
                threshold (float): The threshold at which data points of type 'value' within
                    this CellBox are checked to be either above or below
                lowerbound (float): The lowerbound of acceptable percentage of data_points of
                    type value within this CellBox that are above 'threshold'
                upperbound (float): the upperbound of acceptable percentage of data_points of
                    type value within this CellBox that are above 'threshold'

            Returns:
                hom_condition (string): The homogeniety condtion of this CellBox by given parameters
                    hom_condition is of the form -

                CLR = the proportion of data points within this cellbox over a given
                    threshold is lower than the lowerbound
                HOM = the proportion of data points within this cellbox over a given
                    threshold is higher than the upperbound
                MIN = the cellbox contains less than a minimum number of data points

                HET = the proportion of data points within this cellbox over a given
                    threshold if between the upper and lower bound
        """
        data_limit = 4
        if self._j_grid:
            data_limit = 3000

        data_points = self.get_data_points([value])

        if data_points.shape[0] < data_limit:
            return "MIN"

        over_threshold = data_points.loc[data_points[value] > threshold]

        prop_over = over_threshold.shape[0] / data_points.shape[0]
        if prop_over <= lowerbound:
            return "CLR"
        if prop_over >= upperbound:
            return "HOM"
        return "HET"

    def hom_condition(self):
        """
            The total homogeneity condition of this CellBox, determined by
            all splitting_conditions within this CellBox.

            Returns:
                hom_condition (string): The homogeneity condition of this CellBox.
                    hom_condition is of the form -

                    CLR = the proportion of datapoints within this CellBox over a given
                        threshold is lower than the lowerbound
                    HOM = the proportion of datapoints within this cellbox over a given
                        threshold is higher than the upperbound
                    MIN = the cellbox contains less than a minimum number of datapoints

                    HET = the proportion of datapoints within this cellbox over a given
                        threshold if between the upper and lower bound

        """
        hom_conditions = []

        for splitting_condition in self._splitting_conditions:
            value = list(splitting_condition.keys())[0]
            threshold = float(splitting_condition[value]['threshold'])
            upperbound = float(splitting_condition[value]['upperBound'])
            lowerbound = float(splitting_condition[value]['lowerBound'])

            hom_conditions.append(self.value_hom_condition(value, threshold, lowerbound, upperbound))

        if "HOM" in hom_conditions:
            return "HOM"
        if "MIN" in hom_conditions:
            return "MIN"
        if "HET" in hom_conditions: 
            return "HET"
        if hom_conditions.count("CLR") == len(hom_conditions):
            return "CLR"

        return "ERR"

    def should_be_split(self):
        """
            returns true or false depending on if any of the splitting conditions
            on values contained within this cellbox dictate that the cellbox
            should be split

            DEPRECATED - use should split instead.
        """

        # if a j_grid has been generated, use a different function to determine splitting
        if self._j_grid:
            splitting_percentage = 0.12
            split_min_prop = 0.05
            split_max_prop = 0.85
            return self.should_we_split(splitting_percentage, split_min_prop, split_max_prop)

        split = False
        for splitting_condition in self._splitting_conditions:
            value = list(splitting_condition.keys())[0]
            threshold = float(splitting_condition[value]['threshold'])
            upperbound = float(splitting_condition[value]['upperBound'])
            lowerbound = float(splitting_condition[value]['lowerBound'])
            split = split or self.value_should_be_split(value, threshold, lowerbound, upperbound)
        return split

    def should_split(self):
        """
            determines if a cellbox should be split based on the homogeneity
            condition of each data type contained within. The homogeneity condition
            of values within this cellbox is calculated using the function
            'value_hom_condition()'

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

        for splitting_condition in self._splitting_conditions:
            value = list(splitting_condition.keys())[0]
            threshold = float(splitting_condition[value]['threshold'])
            upperbound = float(splitting_condition[value]['upperBound'])
            lowerbound = float(splitting_condition[value]['lowerBound'])

            hom_conditions.append(self.value_hom_condition(value,threshold,lowerbound,upperbound))

        if "HOM" in hom_conditions:
            return False
        if "MIN" in hom_conditions:
            return False
        if hom_conditions.count("CLR") == len(hom_conditions):
            return False

        return True

    def split(self):
        """
            splits the current cellbox into 4 corners, returns as a list of cellbox objects.

            Returns:
                split_boxes (list<CellBox>): The 4 corner cellboxes generates by splitting
                    this current cellboxes and dividing the data_points contained between.
        """

        # split_boxes = [{}, {}, {}, {}]

        half_width = self.width / 2
        half_height = self.height / 2

        # create 4 new cellBoxes
        north_west = CellBox(self.lat + half_height, self.long, half_width, half_height,
                             splitting_conditions=self._splitting_conditions, j_grid=self._j_grid)
        north_east = CellBox(self.lat + half_height, self.long + half_width, half_width, half_height,
                             splitting_conditions=self._splitting_conditions, j_grid=self._j_grid)
        south_west = CellBox(self.lat, self.long, half_width, half_height,
                             splitting_conditions=self._splitting_conditions, j_grid=self._j_grid)
        south_east = CellBox(self.lat, self.long + half_width, half_width, half_height,
                             splitting_conditions=self._splitting_conditions, j_grid=self._j_grid)

        split_boxes = [north_west, north_east, south_west, south_east]

        for split_box in split_boxes:
            split_box.split_depth = self.split_depth + 1

            # Split dataPoints per box
            long_loc = self._data_points.loc[(self._data_points['long'] > split_box.long) &
                                             (self._data_points['long'] <= (split_box.long + split_box.width))]
            lat_long_loc = long_loc.loc[(long_loc['lat'] > split_box.lat) &
                                        (long_loc['lat'] <= (split_box.lat + split_box.height))]

            split_box.add_data_points(lat_long_loc)

            split_box.add_value_output_type(self.get_value_out_types())
            split_box.set_value_fill_types(self.get_value_fill_types())
            split_box.set_minimum_datapoints(self.minimum_datapoints)

            for value in split_box.get_value_fill_types().keys():
                if value in split_box.get_value_out_types().keys():
                    value_output_type = split_box.get_value_out_types()['value']
                else:
                    value_output_type = "MEAN"

                if np.isnan(split_box.get_value(value)):
                    if split_box.get_value_fill_types()[value] == "zero":
                        fill_value = 0
                    elif split_box.get_value_fill_types()[value] == "parent":
                        fill_value = self.get_value(value, value_output_type)
                    else:
                        fill_value = np.nan
              
                    if not np.isnan(fill_value):
                        datapoint = [[split_box.getcy(), split_box.getcx(), fill_value]]
                        fill_df = pd.DataFrame(datapoint, columns=['lat','long',value])

                        split_box.add_data_points(fill_df)

            # if parent box is land, all child boxes are considered land
            if self.land_locked:
                split_box.land_locked = True

            if self._j_grid:
                split_box.grid_uc = self.grid_uc
                split_box.grid_vc = self.grid_vc

                # set gridCoord of split boxes equal to parent.
                split_box.set_grid_coord(self.x_coord, self.y_coord)

                # create focus for split boxes.
                split_box.set_focus(self.get_focus().copy())
                split_box.add_to_focus(split_boxes.index(split_box))

        return split_boxes

    # Misc

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
        if (lat >= self.lat) & (lat <= self.lat + self.height):
            if (long >= self.long) & (long <= self.long + self.width):
                return True
        return False

    def __str__(self):
        '''
            Converts a cellBox to a String which may be printed to console for debugging purposes
            TODO
        '''
        cellbox_str = "TODO"
        return cellbox_str

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
                    "value_1" (float): ...,\n
                    ...,\n
                    "value_n" (float): ...
                }
            Returns:
                cell_json (dict): A JSON parsable dictionary representation of this CellBox
        '''
        cell_json = {
            "geometry": str(Polygon(self.get_bounds())),
            'cx': float(self.getcx()),
            'cy': float(self.getcy()),
            'dcx': float(self.getdcx()),
            'dcy': float(self.getdcy())
        }

        if self._j_grid:
            cell_json['uC'] = self.grid_uc
            cell_json['vC'] = self.grid_vc
            cell_json['SIC'] = self.get_value('SIC')
            cell_json['elevation'] = 0 if self.is_land_m() else -100
            cell_json['thickness'] = 2
            cell_json['density'] = 875
        else:
            for value in self.get_data_names():
                if value in self.get_value_out_types().keys():
                    cell_json[value] = float(self.get_value(value, 
                        self.get_value_out_types()[value]))
                else:
                    cell_json[value] = float(self.get_value(value))

        return cell_json

    def contains_land(self):
        """
            DEPRECATED - Land mask are now calculated based on the average depth of a cell.

            Returns True if any icepoint within the cell has a
            depth less than the specified minimum depth.
        """

        if self._j_grid:
            return self.is_land_m()

        depth_list = self._data_points.dropna(subset=['depth'])['depth']

        if (depth_list > self.min_depth).any():
            return True
        return False

    def is_land(self):
        """
            DEPRECATED - land masked are now calculated based on the average depth of a cell

            Returns True if all icepoints within the cell have a
            depth less than the specified minimum depth.
        """
        if self._j_grid:
            return self.is_land_m()

        depth_list = self._data_points.dropna(subset=['depth'])['depth']
        if (depth_list > self.min_depth).all():
            return True
        return False

    # Functions used for j_grid regression testing.
    def set_grid_coord(self, xpos, ypos):
        """
            sets up initial grid-coordinate when creating a j_grid

            for use in j_grid regression testing
        """
        self.x_coord = xpos
        self.y_coord = ypos

    def set_focus(self, focus):
        """
            initialize the focus of this cellbox

            for use in j_grid regression testing
        """
        self.focus = focus

    def add_to_focus(self, focus):
        """
            append additional information to the focus of this cellbox
            to be used when splitting.

            for use in j_grid regression testing
        """
        self.focus.append(focus)

    def get_focus(self):
        """
            returns the focus of this cellbox

            for use in j_grid regression testing
        """
        return self.focus

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

    def mesh_dump(self):
        """
            returns a string representing all the information stored in the mesh
            of this cellbox

            for use in j_grid regression testing
        """
        mesh_dump = ""
        mesh_dump += self.node_string() + "; "  # add node string
        mesh_dump += "0 "
        mesh_dump += str(self.getcy()) + ", " + str(self.getcx()) + "; "  # add lat,lon
        ice_area = self.get_value('SIC')
        if np.isnan(ice_area):
            ice_area = 0
        mesh_dump += str(ice_area) + "; "  # add ice area
        if np.isnan(self.grid_uc):
            mesh_dump += str(0) + ", " + str(0) + ", "
        else:
            mesh_dump += str(self.grid_uc) + ", " + str(self.grid_vc) + ", "
        mesh_dump += str(self.get_data_points(['SIC']).shape[0])
        mesh_dump += "\n"

        return mesh_dump

    def add_current_points(self, current_points):
        """
            updates the current points contained within this cellBox to a pandas
            dataframe provided by parameter currentPoints.

            Required for j_grid creation

            for use in j_grid regression testing
        """
        self._current_points = current_points.dropna()
        self.grid_uc = self._current_points['uC'].mean()
        self.grid_vc = self._current_points['vC'].mean()

        self._data_points = pd.concat([self._data_points, current_points], axis=0)

    def set_land(self):
        """
            sets attribute 'land_locked' of a cellBox based on the proportion
            of current vectors contained within it that are not empty.

            Only to be used on un-split cells

            for use in j_grid regression testing

            TODO requires changing to deem a cell land if the datapoint closest
            to the centre of the cell is nan
        """
        if self.split_depth == 0:  # Check if a cell has not been split
            total_currents = self._current_points.dropna()
            watermin = 112.5

            if total_currents.shape[0] < watermin:
                self.land_locked = True

    def is_land_m(self):
        """
            returns true/false dependent on if this cellbox is considered land

            used for j_grid regression testing
        """
        return self.land_locked

    def should_we_split(self, splitting_percentage, split_min_prop, split_max_prop):
        """
            returns true/false dependent on if this cellbox should be split

            used for j_grid regression testing
        """
        if not self._j_grid:
            return self.should_be_split()
        data_limit = 3000

        ice_points = self._data_points.dropna(subset=['iceArea'])

        if ice_points.shape[0] < data_limit:
            return False

        prop_over = ice_points.loc[ice_points['iceArea'] > splitting_percentage]

        proportion_over_x_percent = prop_over.shape[0] / ice_points.shape[0]
        return split_min_prop < proportion_over_x_percent < split_max_prop
