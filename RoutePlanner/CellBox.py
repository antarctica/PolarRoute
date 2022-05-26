"""
Outlined in this section we will discuss the usage of the CellBox functionallity
of the pyRoutePlanner. In this series of class distributions we house our discrete
representation of input data. In each CellBox we determine the mean and variance of
the information goverining our nemerical world, this includes and is not limited to:
Ocean Currents, Sea Ice Concentration, Bathemetric depth, whether on land.

Example:
    An example of running this code can be executed by running the following
    in a ipython/Jupyter Notebook::

        from RoutePlanner import CellBox
        ....

Additional information on constructing document strings using the Google
DocString method can be found at
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

Attributes:
    Some of the key attributes that the CellBox comprises are ...

Todo:
    * Adding the addition of ...


"""

from matplotlib.patches import Polygon
import math
import numpy as np
import pandas as pd

class CellBox:
    """Exceptions are documented in the same way as classes.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        lat    (:obj:`float`): ...
        long   (:obj:`float`): ...
        width  (:obj:`float`): ...
        height (:obj:`float`): ...
    Attributes:
        ...
    """
    split_depth = 0

    def __init__(self, lat, long, width, height, splitting_conditions = [], j_grid=False):
        # Box information relative to bottom left
        self.lat = lat
        self.long = long
        self.width = width
        self.height = height

        # Minimum Depth to be used in the land mask
        self.min_depth = -10

        self._data_points = pd.DataFrame()

        self._splitting_conditions = splitting_conditions

        # For initial implementation of land based from Java codebase.
        self._j_grid = j_grid
        self.land_locked = False
        self.grid_uc = 0
        self.grid_vc = 0
        self.x_coord = 0
        self.y_coord = 0
        self.focus = ""

    # Functions used for getting data from a cellBox
    def getcx(self):
        """
            returns x-position of the centroid of the cellbox
        """
        return self.long + self.width/2

    def getcy(self):
        """
            returns y-position of the centroid of the cellbox
        """
        return self.lat + self.height/2

    def getdcx(self):
        """
            returns x-distance from the edge to the centroid of the cellbox
        """
        return self.width/2

    def getdcy(self):
        """
            returns y-distance from the edge to the centroid of the cellbox
        """
        return self.height/2

    def get_data_points(self, values = []):
        """
            Returns a dataframe of containing values specifed in parameter 'values'.
            If values is empty, return full dataframe.
        """
        if len(values) == 0:
            return self._data_points
        else:
            data_points = pd.DataFrame()
            for value in values:
                data_points = pd.concat(
                    [data_points, self.get_data_points().dropna(subset = [value])], axis = 0)

            columns =  ['lat', 'long', 'time'] + values
            return data_points[columns]

    def get_value(self, value):
        """
            returns the mean value of the datapoints within this cellbox
            specifed by the parameter 'value'
        """
        data_frame = self.get_data_points(values = [value])

        value = data_frame[value].mean()
        # temporary fix to avoid crashing - should be changed!
        if np.isnan(value):
            value = 0
        return value

    def get_bounds(self):
        """
            returns the bounds of this cellbox
        """
        bounds = [[self.long, self.lat],
                    [self.long, self.lat + self.height],
                    [self.long + self.width, self.lat + self.height],
                    [self.long + self.width, self.lat],
                    [self.long, self.lat]]
        return bounds

    # Functions used for adding data to a cellbox
    def add_data_points(self, new_data_points):
        """
            adds a dataframe containing datapoints to the dataframe
            of datapoints within this cellbox
        """
        self._data_points = pd.concat([self._data_points, new_data_points], axis=0)

    def add_splitting_condition(self, splitting_condition):
        """
            adds a dictionary containing a splitting condition to the
            list of splitting conditions contained within this cellbox

            splitting condition is of form:
            {'': {
                'threshold':,
                'upperbound':,
                'lowerbound':
            }}
        """
        self._splitting_conditions = self._splitting_conditions + [splitting_condition]

    # Functions used for splitting a cellbox
    def value_should_be_split(self, value, threshold, lowerbound, upperbound):
        """
            returns true or false dependant of wether a splitting condition associated
            with parameter 'value' should cause the cellbox to be split dependant on
            parameters 'threshold', 'upperbound' and 'lowerbound
        """
        data_limit = 4

        data_points = self.get_data_points(values = [value])

        if data_points.shape[0] < data_limit:
            return False

        prop_over = data_points.loc[data_points[value] > threshold]

        proportion_over_x_percent = prop_over.shape[0] / data_points.shape[0]
        return (proportion_over_x_percent > lowerbound and proportion_over_x_percent < upperbound)

    def should_be_split(self):
        """
            returns true or false dependant on if any of the splitting condtions
            on values contained within this cellbox dictate that the cellbox
            should be split
        """

        # if a j_grid has been generated, use a different function to determin splitting
        if self._j_grid:
            splitting_percentage = 0.12
            split_min_prop = 0.05
            split_max_prop = 0.85
            return self.shouldWeSplit(splitting_percentage, split_min_prop, split_max_prop)

        split = False
        for splitting_condition in self._splitting_conditions:
            value = list(splitting_condition.keys())[0]
            threshold = float(splitting_condition[value]['threshold'])
            upperbound = float(splitting_condition[value]['upperBound'])
            lowerbound = float(splitting_condition[value]['lowerBound'])
            split = split or self.value_should_be_split(value, threshold, lowerbound, upperbound)
        return split

    def split(self):
        '''
            splits the current cellbox into 4 corners, returns as a list of cellbox objects.
        '''

        # split_boxes = [{}, {}, {}, {}]

        half_width = self.width / 2
        half_height = self.height / 2

        # create 4 new cellBoxes
        north_west = CellBox(self.lat + half_height, self.long, half_width, half_height,
                            splitting_conditions = self._splitting_conditions, j_grid=self._j_grid)
        north_east = CellBox(self.lat + half_height,self.long + half_width,half_width,half_height,
                            splitting_conditions = self._splitting_conditions, j_grid=self._j_grid)
        south_west = CellBox(self.lat, self.long, half_width, half_height, 
                            splitting_conditions = self._splitting_conditions, j_grid=self._j_grid)
        south_east = CellBox(self.lat, self.long + half_width, half_width, half_height,
                            splitting_conditions = self._splitting_conditions, j_grid=self._j_grid)


        split_boxes = [north_west, north_east, south_west, south_east]


        for split_box in split_boxes:
            split_box.split_depth = self.split_depth + 1

            #Split dataPoints per box
            long_loc = self._data_points.loc[(self._data_points['long'] > split_box.long) &
                            (self._data_points['long'] <= (split_box.long + split_box.width))]
            lat_long_loc = long_loc.loc[(long_loc['lat'] > split_box.lat) &
                            (long_loc['lat'] <= (split_box.lat + split_box.height))]

            split_box.add_data_points(lat_long_loc)

            # if parent box is land, all child boxes are considered land
            if self.land_locked:
                split_box.land_locked = True

            if self._j_grid:
                split_box.grid_uc = self.grid_uc
                split_box.grid_vc = self.grid_vc

                # set gridCoord of split boxes equal to parent.
                split_box.set_grid_coord(self.x_coord, self.y_coord)

                # create focus for split boxes.
                split_box.set_focus(self.getFocus().copy())
                split_box.add_to_focus(split_boxes.index(split_box))


        return split_boxes

    #Misc
    def ice_thickness(self, date):
        """
            Returns mean ice thickness within this cellBox. Data taken from Table 3 in: doi:10.1029/2007JC004254

            TODO - Data is hard coded - should be stored in an external file
        """
        # The table has missing data points for Bellinghausen Autumn and Weddell W Winter, these require further thought
        thicknesses = {'Ross': {'w': 0.72, 'sp': 0.67, 'su': 1.32, 'a': 0.82, 'y': 1.07},
                    'Bellinghausen': {'w': 0.65, 'sp': 0.79, 'su': 2.14, 'a': 0.79, 'y': 0.90},
                    'Weddell E': {'w': 0.54, 'sp': 0.89, 'su': 0.87, 'a': 0.44, 'y': 0.73},
                    'Weddell W': {'w': 1.33, 'sp': 1.33, 'su': 1.20, 'a': 1.38, 'y': 1.33},
                    'Indian': {'w': 0.59, 'sp': 0.78, 'su': 1.05, 'a': 0.45, 'y': 0.68},
                    'West Pacific': {'w': 0.72, 'sp': 0.68, 'su': 1.17, 'a': 0.75, 'y': 0.79}
                    }
        seasons = {1: 'su', 2: 'su', 3: 'a', 4: 'a', 5: 'a', 6: 'w', 7: 'w', 8: 'w', 9: 'sp', 10: 'sp', 11: 'sp',
                12: 'su'}
        month = int(date[5:7])
        season = seasons[month]

        if -130 <= self.long < -60:
            sea = 'Bellinghausen'
        elif -60 <= self.long < -45:
            sea = 'Weddell W'
        elif -45 <= self.long < 20:
            sea = 'Weddell E'
        elif 20 <= self.long < 90:
            sea = 'Indian'
        elif 90 <= self.long < 160:
            sea = 'West Pacific'
        elif (160 <= self.long < 180) or (-180 <= self.long < -130):
            sea = 'Ross'

        return thicknesses[sea][season]

    def ice_density(self, date):
        """
            Returns mean ice density within this cellBox

            TODO - Data is hard coded - should be stored in an external file.
        """
        seasons = {1:'su',2:'su',3:'a',4:'a',5:'a',6:'w',7:'w',8:'w',9:'sp',10:'sp',11:'sp',12:'su'}
        densities = {'su':875.0,'sp':900.0,'a':900.0,'w':920.0}

        month = int(date[5:7])
        season = seasons[month]
        density = densities[season]

        # Seasonal values from: https://doi.org/10.1029/2007JC004254
        return density

    def contains_point(self,lat,long):
        """
            Returns true if a given lat/long coordinate is contained within this cellBox.
        """
        if (lat >= self.lat) & (lat <= self.lat + self.height):
            if (long >= self.long) & (long <= self.long + self.width):
                return True
        return False

    def __str__(self):
        '''
            Converts a cellBox to a String which may be printed to console for debugging purposes
        '''
        cellbox_str = "TODO"
        return cellbox_str

    def to_json(self):
        '''
            convert cellBox to JSON
        '''
        cellbox_json = "{"
        cellbox_json += "}"
        return cellbox_json

    def contains_land(self):
        """
            Returns True if any icepoint within the cell has a 
            depth less than the specified minimum depth.
        """

        if self._j_grid:
            return self.isLandM()

        depth_list = self._data_points.dropna(subset=['depth'])['depth']

        if (depth_list > self.min_depth).any():
            return True
        return False

    def is_land(self):
        """
            Returns True if all icepoints within the cell have a 
            depth less than the specified minimum depth.
        """
        if self._j_grid:
            return self.isLandM()

        depth_list = self._data_points.dropna(subset=['depth'])['depth']
        if (depth_list > self.min_depth).all():
            return True
        return False

    # Fuctions used for j_grid regression testing.
    def set_grid_coord(self, xpos, ypos):
        """
            sets up initial grid-coordinate when creating a j_grid
        """
        self.x_coord = xpos
        self.y_coord = ypos

    def set_focus(self, focus):
        """
            initialize the focus of this cellbox
        """
        self.focus = focus

    def add_to_focus(self, focus):
        """
            append additonal information to the focus of this cellbox
            to be used when splitting.
        """
        self.focus.append(focus)

    def getFocus(self):
        return self.focus

    def gridCoord(self):
        return "(" + str(int(self.x_coord)) + "," + str(int(self.y_coord)) + ")"

    def nodeString(self):
        nodeString = self.gridCoord() + " F:" + str(len(self.getFocus()))

        focusString = "["
        for x in self.getFocus():
            focusString += str(x) + " "
        focusString += "]"
        return nodeString + " " + focusString

    def meshDump(self):
        meshDump = ""
        meshDump += self.nodeString() + "; "  # add node string
        meshDump += "0 "
        meshDump += str(self.getcy()) + ", " + str(self.getcx()) + "; "  # add lat,lon
        meshDump += str(self.iceArea()) + "; "  # add ice area
        if np.isnan(self.grid_uc):
            meshDump += str(0) + ", " + str(0) + ", "
        else:
            meshDump += str(self.grid_uc) + ", " + str(self.grid_vc) + ", "
        meshDump += str(self._data_points.shape[0])
        meshDump += "\n"

        return meshDump

    def addCurrentPoints(self, currentPoints):
        '''
            updates the current points contained within this cellBox to a pandas 
            dataframe provided by parameter currentPoints.

            Required for j_grid creation
        '''
        self._currentPoints = currentPoints.dropna()
        self.grid_uc = self._currentPoints['uC'].mean()
        self.grid_vc = self._currentPoints['vC'].mean()

        self._data_points = pd.concat([self._data_points, currentPoints], axis=0)

    def setLand(self):
        """
            sets attribute 'land_locked' of a cellBox based on the proportion of current vectors contained
            within it that are not empty.

            Only to be used on un-split cells
        """
        if self.split_depth == 0:  # Check if a cell has not been split
            totalCurrents = self._currentPoints.dropna()
            watermin = 112.5

            if totalCurrents.shape[0] < watermin:
                self.land_locked = True

    def isLandM(self):
        return self.land_locked

    def shouldWeSplit(self, splittingPercentage, splitMinProp, splitMaxProp):
        if self._j_grid == False:
            return self.should_be_split()
    
        dataLimit = 1

        icePoints = self._data_points.dropna(subset=['iceArea'])

        if icePoints.shape[0] < dataLimit:
            return False

        propOver = icePoints.loc[icePoints['iceArea'] > splittingPercentage]

        proportionOverXpercent = propOver.shape[0] / icePoints.shape[0]
        return proportionOverXpercent > splitMinProp and proportionOverXpercent < splitMaxProp
