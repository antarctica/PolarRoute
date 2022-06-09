"""
Outlined in this section we will discuss the usage of the CellBox functionallity
of the pyRoutePlanner. In this series of class distributions we house our discrete
representation of input data. In each CellBox we determine the mean and variance of 
the information goverining our nemerical world, this includes and is not limited to:
Ocean Currents, Sea Ice Concentration, Bathemetric depth, whether on land.

Example:
    An example of running this code can be executed by running the following in a ipython/Jupyter Notebook::

        from RoutePlanner import CellBox
        ....

Additional information on constructing document strings using the Google DocString method can be found at
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

    splitDepth = 0

    def __init__(self, lat, long, width, height, splittingConditions = [],j_grid=False):
        # Box information relative to bottom left
        self.lat = lat
        self.long = long
        self.width = width
        self.height = height

        self._j_grid = j_grid

        # Defining the initial centroid information for cell
        # TODO move these out of object attributes at to get methods.
        self.cx = self.long + self.width/2
        self.cy = self.lat + self.height/2

        # TODO move these out of object attributes at to get methods.
        self.dcx = self.width/2
        self.dcy = self.height/2

        # Minimum Depth to be used in the land mask
        self.minDepth = -10

        # For initial implementation of land based from Java codebase.
        self.landLocked = False

        self._dataPoints = pd.DataFrame()

        self._splittingConditions = splittingConditions

    def getcx(self):
        return self.long + self.width/2

    def getcy(self):
        return self.lat + self.height/2

    def setGridCoord(self, x, y):
        self.xCoord = x
        self.yCoord = y

    def setFocus(self, f):
        self.focus = f

    def addToFocus(self, f):
        self.focus.append(f)

    def getFocus(self):
        return self.focus

    def gridCoord(self):
        return "(" + str(int(self.xCoord)) + "," + str(int(self.yCoord)) + ")"

    def nodeString(self):
        nodeString = self.gridCoord() + " F:" + str(len(self.getFocus()))

        focusString = "["
        for x in self.getFocus():
            focusString += str(x) + " "
        focusString += "]"
        return nodeString + " " + focusString

    def iceThickness(self, date):
        """
            Returns mean ice thickness within this cellBox. Data taken from Table 3 in: doi:10.1029/2007JC004254
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

    def iceDensity(self, date):
        """
            Returns mean ice density within this cellBox
        """
        seasons = {1:'su',2:'su',3:'a',4:'a',5:'a',6:'w',7:'w',8:'w',9:'sp',10:'sp',11:'sp',12:'su'}
        densities = {'su':875.0,'sp':900.0,'a':900.0,'w':920.0}

        month = int(date[5:7])
        season = seasons[month]
        d = densities[season]

        # Seasonal values from: https://doi.org/10.1029/2007JC004254
        return d

    def meshDump(self):
        meshDump = ""
        meshDump += self.nodeString() + "; "  # add node string
        meshDump += "0 "
        meshDump += str(self.getcy()) + ", " + str(self.getcx()) + "; "  # add lat,lon
        meshDump += str(self.iceArea()) + "; "  # add ice area
        if np.isnan(self.griduC):
            meshDump += str(0) + ", " + str(0) + ", "
        else:
            meshDump += str(self.griduC) + ", " + str(self.gridvC) + ", "
        meshDump += str(self._dataPoints.shape[0])
        meshDump += "\n"

        return meshDump

    def addIcePoints(self, icePoints):
        '''
            updates the ice points contained within this cellBox to a pandas dataframe provided by parameter icePoints.
        '''
        self._icePoints = icePoints.dropna() # depricated, to be removed

        self._dataPoints = pd.concat([self._dataPoints, icePoints], axis=0)

    def addDepthPoints(self, depthPoints):
        self._dataPoints = pd.concat([self._dataPoints, depthPoints], axis=0)

    def addDataPoints(self, newDataPoints):
        self._dataPoints = pd.concat([self._dataPoints, newDataPoints], axis=0)

    def addCurrentPoints(self, currentPoints):
        '''
            updates the current points contained within this cellBox to a pandas dataframe provided by parameter currentPoints.
        '''
        self._currentPoints = currentPoints.dropna()
        self.griduC = self._currentPoints['uC'].mean()
        self.gridvC = self._currentPoints['vC'].mean()

        self._dataPoints = pd.concat([self._dataPoints, currentPoints], axis=0)

    def _setDataPoints(self, dataPoints):
        self._dataPoints = dataPoints

    def getDataPoints(self, values = []):

        if len(values) == 0:
            return self._dataPoints
        else:
            dataPoints = pd.DataFrame()
            for value in values:
                dataPoints = pd.concat([dataPoints, self.getDataPoints().dropna(subset = [value])], axis = 0)

            columns =  ['lat', 'long', 'time'] + values
            return dataPoints[columns]

    def getValue(self, value):
        dataFrame = self.getDataPoints(values = [value])
        return dataFrame[value].mean()


    def getIcePointLength(self):
        '''
            Returns the number of ice points contained within this cellBox.
        '''
        return self._icePoints.shape[0]

    def getCurrentPointLength(self):
        '''
            Returns the number of current points contained within this cellBox.
        '''
        return len(self._currentPoints)

    def _getLatRange(self):
        '''
            Returns a string details the lat range of this cellBox, used by the _getRange() and toString() methods.
        '''
        return str(self.lat) + " to " + str(self.lat + self.height)

    def _getLongRange(self):
        '''
            Returns a string detailing the long range of this cellBox, used by the _getRange() and toString() methods.
        '''

        return str(self.long) + " to " + str(self.long + self.width)

    def _getRange(self):
        '''
            Returns a string detailing the lat/long range of this cellBox, used by the toString() method.
        '''
        return "Lat Range: " + self._getLatRange() + ", Long Range: " + self._getLongRange()

    def getBounds(self):
        bounds = [[self.long, self.lat],
                    [self.long, self.lat + self.height],
                    [self.long + self.width, self.lat + self.height],
                    [self.long + self.width, self.lat],
                    [self.long, self.lat]]
        return bounds

    def getWidth(self):
        '''
            INCLUDE
        '''
        return self.width * math.cos(self.lat)

    def iceArea(self):
        """
            Returns mean ice area of all icepoints contained within this cellBox
        """
        iceArea = self._dataPoints['iceArea'].mean()
        if np.isnan(iceArea):
            iceArea = 0
        return iceArea

    def depth(self):
        '''
            Returns mean depth of all icepoints contained within this cellBox
        '''
        return self._dataPoints['depth'].mean()

    def getuC(self):
        '''
            INCLUDE
        '''
        if self._j_grid == True:
            return self.griduC

        uC = self._dataPoints['uC'].mean()
        if np.isnan(uC):
            return 0
        return uC

    def getvC(self):
        '''
            INCLU
        '''
        if self._j_grid == True:
            return self.griduC

        vC = self._dataPoints['vC'].mean()
        if np.isnan(vC):
            return 0
        return vC

    def getIcePoints(self):

        '''
            Returns a pandas dataframe of all icepoints contained within this cellBox
        '''
        return self._icePoints

    def getCurrentPoints(self):
        """
            Return a pandas dataframe of all currentpoints contained within this cellBox
        """
        return self._currentPoints

    def containsPoint(self,lat,long):
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
        s = ""
        s += self._getRange() + "\n"
        s += "    No. of IcePoints: " + str(self.getIcePointLength()) + "\n"
        s += "    No. of CurrentPoints: " + str(self.getCurrentPointLength()) + "\n"
        s += "    Max CurrentPoints: " + str(self.maxCurrentVector()) + "\n"
        s += "    Ice Area: " + str(self.iceArea()) + "\n"
        s += "    split Depth: " + str(self.splitDepth) + "\n"
        s += "    uC: " + str(self.getuC()) + "\n"
        s += "    vC: " + str(self.getvC()) + "\n"
        s += "    depth: " + str(self.depth())
        return s

    def toJSON(self):
        '''
            convert cellBox to JSON
        '''
        s = "{"
        s += "\"lat\":" + str(self.lat) + ","
        s += "\"long\":" + str(self.long) + ","
        s += "\"width\":" + str(self.width) + ","
        s += "\"height\":" + str(self.height) + ","
        s += "\"iceArea\":" + str(self.iceArea()) + ","
        s += "\"splitDepth\":" + str(self.splitDepth)
        s += "}"
        return s

    def containsLand(self):
        """
            Returns True if any icepoint within the cell has a depth less than the specified minimum depth.
        """

        if self._j_grid == True:
            return self.isLandM()

        depthList = self._dataPoints.dropna(subset=['depth'])['depth']

        if (depthList > self.minDepth).any():
            return True
        return False

    def isLand(self):
        """
            Returns True if all icepoints within the cell have a depth less than the specified minimum depth.
        """
        if self._j_grid == True:
            return self.isLandM()

        depthList = self._dataPoints.dropna(subset=['depth'])['depth']
        if (depthList > self.minDepth).all():
            return True
        return False


    def maxCurrentVector(self):
        return self.width * 6 * self.height * 6

    def setLand(self):
        """
            sets attribute 'landLocked' of a cellBox based on the proportion of current vectors contained
            within it that are not empty.

            Only to be used on un-split cells
        """
        if self.splitDepth == 0:  # Check if a cell has not been split
            totalCurrents = self._currentPoints.dropna()
            watermin = 112.5

            if totalCurrents.shape[0] < watermin:
                self.landLocked = True

    def isLandM(self):
        return self.landLocked

    def addSplittingCondition(self, splittingCondition):
        self._splittingConditions = self._splittingConditions + [splittingCondition]

    def valueShouldBeSplit(self, value, threshold, lowerBound, upperBound):
        dataLimit = 4

        dataPoints = self.getDataPoints(values = [value])

        if dataPoints.shape[0] < dataLimit:
            return False

        propOver = dataPoints.loc[dataPoints[value] > threshold]

        proportionOverXpercent = propOver.shape[0] / dataPoints.shape[0]
        return (proportionOverXpercent > lowerBound and proportionOverXpercent < upperBound)
    
    def shouldBeSplit(self):
        split = False
        for splittingCondition in self._splittingConditions:
            value = list(splittingCondition.keys())[0]
            threshold = float(splittingCondition[value]['threshold'])
            upperBound = float(splittingCondition[value]['upperBound'])
            lowerBound = float(splittingCondition[value]['lowerBound'])
            split = split or self.valueShouldBeSplit(value, threshold, lowerBound, upperBound)
        return split

    def isHomogenous(self,  splittingPercentage, splitMinProp, splitMaxProp):
        '''
            returns true if a cell is deemed homogenous, used to define a base case for recursive splitting.
        '''

        icePoints = self._dataPoints.dropna(subset=['iceArea'])

        dataLimit = 1
        if icePoints.shape[0] < dataLimit:
            return True

        # if a cell contains only land, it is homogenous and does not require splitting
        if self.isLand():
            return True
        # if a cell contains both land and sea, it not homogenous and requires splitting
        if self.containsLand():
            return False

        """
        threshold = splittingPercentage

        percentIPsAboveThreshold = icePoints.loc[icePoints['iceArea'] > threshold].shape[0] / icePoints.shape[0]

        lowerBound = splitMinProp
        upperBound = splitMaxProp

        if percentIPsAboveThreshold < lowerBound:
            return True
        if percentIPsAboveThreshold > upperBound:
            return True
        """
        propOver = icePoints.loc[icePoints['iceArea'] > splittingPercentage]

        proportionOverXpercent = propOver.shape[0] / icePoints.shape[0]
        return not(proportionOverXpercent > splitMinProp and proportionOverXpercent < splitMaxProp)
        return False

    def shouldWeSplit(self, splittingPercentage, splitMinProp, splitMaxProp):

        if self._j_grid == False:
            return self.shouldBeSplit()
            #return not self.isHomogenous(splittingPercentage, splitMinProp, splitMaxProp)

        dataLimit = 3000

        icePoints = self._dataPoints.dropna(subset=['iceArea'])

        if icePoints.shape[0] < dataLimit:
            return False

        propOver = icePoints.loc[icePoints['iceArea'] > splittingPercentage]

        proportionOverXpercent = propOver.shape[0] / icePoints.shape[0]
        return proportionOverXpercent > splitMinProp and proportionOverXpercent < splitMaxProp


    def split(self):
        '''
            splits the current cellbox into 4 corners, returns as a list of cellbox objects.
        '''

        splitBoxes = [0, 0, 0, 0]

        halfWidth = self.width / 2
        halfHeight = self.height / 2

        # create 4 new cellBoxes
        northWest = CellBox(self.lat + halfHeight, self.long, halfWidth, halfHeight,
                            splittingConditions = self._splittingConditions, j_grid = self._j_grid)
        northEast = CellBox(self.lat + halfHeight, self.long + halfWidth, halfWidth, halfHeight,
                            splittingConditions = self._splittingConditions, j_grid = self._j_grid)
        southWest = CellBox(self.lat, self.long, halfWidth, halfHeight, 
                            splittingConditions = self._splittingConditions, j_grid = self._j_grid)
        southEast = CellBox(self.lat, self.long + halfWidth, halfWidth, halfHeight, 
                            splittingConditions = self._splittingConditions, j_grid = self._j_grid)

        """
        splitBoxes[0] = southWest
        splitBoxes[1] = southEast
        splitBoxes[2] = northWest
        splitBoxes[3] = northEast
        """

        splitBoxes[0] = northWest
        splitBoxes[1] = northEast
        splitBoxes[2] = southWest
        splitBoxes[3] = southEast

        for splitBox in splitBoxes:
            #TODO requires rework for optimization
            splitBox.splitDepth = self.splitDepth + 1

            """
            #Split icePoints per cellBox
            longLoc = self._icePoints.loc[(self._icePoints['long'] > splitBox.long) &
                                          (self._icePoints['long'] <= (splitBox.long + splitBox.width))]
            latLongLoc = longLoc.loc[(longLoc['lat'] > splitBox.lat) &
                                             (longLoc['lat'] <= (splitBox.lat + splitBox.height))]

            splitBox.addIcePoints(latLongLoc)

            #Split currentPoints per box
            longLoc = self._currentPoints.loc[(self._currentPoints['long'] > splitBox.long) &
                                              (self._currentPoints['long'] <= (splitBox.long + splitBox.width))]
            latLongLoc = longLoc.loc[(longLoc['lat'] > splitBox.lat) &
                                                 (longLoc['lat'] <= (splitBox.lat + splitBox.height))]

            splitBox.addCurrentPoints(latLongLoc)
            """

            #Split dataPoints per box
            longLoc = self._dataPoints.loc[(self._dataPoints['long'] > splitBox.long) &
                                              (self._dataPoints['long'] <= (splitBox.long + splitBox.width))]
            latLongLoc = longLoc.loc[(longLoc['lat'] > splitBox.lat) &
                                     (longLoc['lat'] <= (splitBox.lat + splitBox.height))]

            splitBox._setDataPoints(latLongLoc)

            # if parent box is land, all child boxes are considered land
            if self.landLocked:
                splitBox.landLocked = True

            if self._j_grid == True:
                splitBox.griduC = self.griduC
                splitBox.gridvC = self.gridvC

                # set gridCoord of split boxes equal to parent.
                splitBox.setGridCoord(self.xCoord, self.yCoord)

                # create focus for split boxes.
                splitBox.setFocus(self.getFocus().copy())
                splitBox.addToFocus(splitBoxes.index(splitBox))


        return splitBoxes

    def recursiveSplit(self, maxSplits,threshold,lowerBound,upperBound):
        '''
            Recursively splits this cellBox until all split cellBoxes are considered homogenous (defined by the isHomogenous() function)
            or a the cellBox has reached a maximum split depth, given by parameter maxSplits.
        '''
        splitCells = []
        # Base case for recursive splitting.
        # Do not split a cell if it is homogenous or the maximum split depth has been reached
        if self.isHomogenous() or (self.splitDepth >= maxSplits):

            splitCells.append(self)
            return splitCells
        else:
            splitBoxes = self.split()
            for splitBox in splitBoxes:
                splitCells = splitCells + splitBox.recursiveSplit(maxSplits)
            return splitCells