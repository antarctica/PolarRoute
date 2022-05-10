from tracemalloc import start
import numpy as np
from RoutePlanner.CellBox import CellBox
import pandas as pd
#from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MatplotPolygon
import math
import xarray as xr
#import geopandas as gpd

def bearing(st,en): # Should be moved out of the CellBox.py file.
    long1,lat1 = st
    long2,lat2 = en
    dlong = long2-long1
    dlat  = lat2-lat1
    vector_1 = [0, 1]
    vector_2 = [dlong, dlat]

    if np.linalg.norm(vector_2) == 0:
        return np.nan
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)/(np.pi/180)*np.sign(vector_2[0])

    if (angle==0) & (np.sign(dlat)==-1):
        angle=180


    if angle < 0:
        angle = angle +360
    angle
    return angle

def Intersection_BoxLine(Cell_s,Pt,type): # Should be moved out of the CellBox.py file
    X1,Y1 = Cell_s.long+Cell_s.width/2,Cell_s.lat+Cell_s.height/2
    X2,Y2 = Pt
    if type==np.nan:
        Px = np.nan
        Py = np.nan
        return Px,Py
    if type == 2:
        X3,Y3 = Cell_s.long+Cell_s.width,Cell_s.lat
        X4,Y4 = Cell_s.long+Cell_s.width,Cell_s.lat+Cell_s.height
    if type == -4:
        X3,Y3 = Cell_s.long,Cell_s.lat+Cell_s.height
        X4,Y4 = Cell_s.long+Cell_s.width,Cell_s.lat+Cell_s.height
    if type == -2:
        X3,Y3 = Cell_s.long,Cell_s.lat
        X4,Y4 = Cell_s.long,Cell_s.lat+Cell_s.height
    if type == 4:
        X3,Y3 = Cell_s.long,Cell_s.lat
        X4,Y4 = Cell_s.long+Cell_s.width,Cell_s.lat
    if type == 1:
        Px = Cell_s.long + Cell_s.width/2 + Cell_s.width/2
        Py = Cell_s.lat  + Cell_s.height/2 + Cell_s.height/2
        return Px,Py
    if type == -1:
        Px = Cell_s.long + Cell_s.width/2  - Cell_s.width/2
        Py = Cell_s.lat  + Cell_s.height/2 - Cell_s.height/2
        return Px,Py
    if type == 3:
        Px = Cell_s.long + Cell_s.width/2 + Cell_s.width/2
        Py = Cell_s.lat  + Cell_s.height/2 - Cell_s.height/2
        return Px,Py
    if type == -3:
        Px = Cell_s.long + Cell_s.width/2 - Cell_s.width/2
        Py = Cell_s.lat  + Cell_s.height/2 + Cell_s.height/2
        return Px,Py
    D  = (X1-X2)*(Y3-Y4) - (Y1-Y2)*(X3-X4)
    Px = ((X1*Y2 - Y1*X2)*(X3-X4) - (X1-X2)*(X3*Y4-Y3*X4))/D
    Py = ((X1*Y2-Y1*X2)*(Y3-Y4)-(Y1-Y2)*(X3*Y4-Y3*X4))/D
    return Px,Py

class CellGrid:

    def __init__(self, config, j_grid=False):
        self._longMin = config['Region']['longMin']
        self._longMax = config['Region']['longMax']
        self._latMin = config['Region']['latMin']
        self._latMax = config['Region']['latMax']

        self._cellWidth = config['Region']['cellWidth']
        self._cellHeight = config['Region']['cellHeight']

        self._startTime = config['Region']['startTime']
        self._endTime = config['Region']['endTime']

        self._dataSources = config['Data_sources']

        self._j_grid = j_grid

        self.cellBoxes = []

        # Initialise cellBoxes.
        for lat in np.arange(self._latMin, self._latMax, self._cellHeight):
            for long in np.arange(self._longMin, self._longMax, self._cellWidth):
                cellBox = CellBox(lat, long, self._cellWidth, self._cellHeight, 
                                    splittingConditions = [], j_grid = self._j_grid)
                self.cellBoxes.append(cellBox)

        gridWidth = (self._longMax - self._longMin) / self._cellWidth
        gridHeight = (self._latMax - self._latMin) / self._cellHeight

        # Calculate initial neighbours graph.
        self.neighbourGraph = {}
        for cellBox in self.cellBoxes:
            cellBoxIndx = self.cellBoxes.index(cellBox)
            neighbourMap = {1: [], 2: [], 3: [], 4: [], -1: [], -2: [], -3: [], -4: []}

            # add east neighbours to neighbour graph
            if (cellBoxIndx + 1) % gridWidth != 0:
                neighbourMap[2].append(cellBoxIndx + 1)
                # south-east neighbours
                if (cellBoxIndx + gridWidth < len(self.cellBoxes)):
                    neighbourMap[1].append(int((cellBoxIndx + gridWidth) + 1))
                # north-east neighbours
                if (cellBoxIndx - gridWidth >= 0):
                    neighbourMap[3].append(int((cellBoxIndx - gridWidth) + 1))

            # add west neighbours to neighbour graph
            if (cellBoxIndx) % gridWidth != 0:
                neighbourMap[-2].append(cellBoxIndx - 1)
                # add south-west neighbours to neighbour graph
                if (cellBoxIndx + gridWidth < len(self.cellBoxes)):
                    neighbourMap[-3].append(int((cellBoxIndx + gridWidth) - 1))
                # add north-west neighbours to neighbour graph
                if (cellBoxIndx - gridWidth >= 0):
                    neighbourMap[-1].append(int((cellBoxIndx - gridWidth) - 1))

            # add south neighbours to neighbour graph
            if (cellBoxIndx + gridWidth < len(self.cellBoxes)):
                neighbourMap[-4].append(int(cellBoxIndx + gridWidth))

            # add north neighbours to neighbour graph
            if (cellBoxIndx - gridWidth >= 0):
                neighbourMap[4].append(int(cellBoxIndx - gridWidth))

            self.neighbourGraph[cellBoxIndx] = neighbourMap

            # set gridCoord of cellBox
            xCoord = cellBoxIndx % gridWidth
            yCoord = abs(math.floor(cellBoxIndx / gridWidth) - (gridHeight - 1))
            cellBox.setGridCoord(xCoord, yCoord)

            # set focus of cellBox
            cellBox.setFocus([])
        self.splittingConditions = []
        for dataSource in config['Data_sources']:
                self.addDataSource(dataSource)

        self.iterativeSplit(config['Region']["splitDepth"], 0,0,0)
        

    def neighbourTest(self, cellBox):
        """
            Returns a flattened list of all neighbours of a given cellBox.
        """
        cellBoxIndx = self.cellBoxes.index(cellBox)
        neighboursIndx = []

        neighboursIndx.append(self.neighbourGraph[cellBoxIndx][1])  # north-east
        neighboursIndx.append(self.neighbourGraph[cellBoxIndx][2])  # east
        neighboursIndx.append(self.neighbourGraph[cellBoxIndx][3])  # south-east
        neighboursIndx.append(self.neighbourGraph[cellBoxIndx][4])  # south
        neighboursIndx.append(self.neighbourGraph[cellBoxIndx][-1])  # south-west
        neighboursIndx.append(self.neighbourGraph[cellBoxIndx][-2])  # west
        neighboursIndx.append(self.neighbourGraph[cellBoxIndx][-3])  # north-west
        neighboursIndx.append(self.neighbourGraph[cellBoxIndx][-4])  # north

        neighbours = []
        for sublist in neighboursIndx:
            for indx in sublist:
                neighbours.append(self.cellBoxes[indx])

        return neighbours

    def addIcePoints(self, icePoints):
        """
            Takes a dataframe containing ice points and assigns them to cellBoxes within the cellGrid
        """
        for cellBox in self.cellBoxes:

            longLoc    = icePoints.loc[(icePoints['long'] > cellBox.long) & (icePoints['long'] <= (cellBox.long + cellBox.width))]
            latLongLoc = longLoc.loc[(longLoc['lat'] > cellBox.lat) & (longLoc['lat'] <= (cellBox.lat + cellBox.height))]

            cellBox.addIcePoints(latLongLoc)
            """
            latMin = cellBox.lat
            longMin = cellBox.long

            if longMin < 0:
                longMin += 360

            longMax = longMin + cellBox.width
            latMax = latMin + cellBox.height

            icePointCut = icePoints.sel(YC=slice(latMin, latMax), XC=slice(longMin, longMax))
            # icePointCut = icePointCut.sel()
            ipSlice = icePointCut.to_dataframe()

            ipSlice = ipSlice.reset_index()
            ipSlice = ipSlice.rename(columns={'XC': 'long', 'YC': 'lat', 'Depth': 'depth', 'SIarea': 'iceArea'})
            ipSlice['long'] = ipSlice['long'].apply(lambda x: x if x <= 180 else x - 360)

            cellBox.addIcePoints(ipSlice)
            """
    def addDataSource(self, dataSource):

        for value in dataSource['values']:
            if "splittingCondition" in value:
                    splittingCondition = {value['destinationName'] : value['splittingCondition']}
                    self.splittingConditions = self.splittingConditions + [splittingCondition]
    
        for cellBox in self.cellBoxes:
            latMin = cellBox.lat
            longMin = cellBox.long

            longMax = longMin + cellBox.width
            latMax = latMin + cellBox.height
            
            path = dataSource['path']

            dataSet = xr.open_dataset(path)

            if "timeName" in dataSource:
                dataSet = dataSet.rename({dataSource['latName']:'lat',
                                        dataSource['longName']:'long',
                                        dataSource['timeName']:'time'})
            
                dataSlice = dataSet.sel(time = slice(self._startTime, self._endTime),
                                        lat = slice(latMin, latMax),
                                        long = slice(longMin, longMax))
            else:
                dataSet = dataSet.rename({dataSource['latName']:'lat',
                                        dataSource['longName']:'long'})
            
                dataSlice = dataSet.sel(lat = slice(latMin, latMax),
                                        long = slice(longMin, longMax))

            df = dataSlice.to_dataframe()
            df = df.reset_index()

            selected = []
            for value in dataSource['values']:
                df = df.rename(columns = {value['sourceName']:value['destinationName']})
                selected = selected + [value['destinationName']]

                if "splittingCondition" in value:
                    splittingCondition = {value['destinationName'] : value['splittingCondition']}
                    cellBox.addSplittingCondition(splittingCondition)

            df = df.dropna(subset = selected)

            cellBox.addDataPoints(df)

    def addCurrentPoints(self, currentPoints):
        """
            Takes a dataframe containing current points and assigns then to cellBoxes within the cellGrid
        """
        for cellBox in self.cellBoxes:
            longLoc = currentPoints.loc[(currentPoints['long'] > cellBox.long) & (currentPoints['long'] <= (cellBox.long + cellBox.width))]
            latLongLoc = longLoc.loc[(longLoc['lat'] > cellBox.lat) & (longLoc['lat'] <= (cellBox.lat + cellBox.height))]

            cellBox.addCurrentPoints(latLongLoc)
            cellBox.setLand()

    def addDataPoints(self, dataPoints):
        for cellBox in self.cellBoxes:
            longLoc = dataPoints.loc[(dataPoints['long'] > cellBox.long) & (dataPoints['long'] <= (cellBox.long + cellBox.width))]
            latLongLoc = longLoc.loc[(longLoc['lat'] > cellBox.lat) & (longLoc['lat'] <= (cellBox.lat + cellBox.height))]

            cellBox.addDataPoints(latLongLoc)

    def addWindPoints(self, windPoints):
        """
            Takes a dataframe containing current points and assigns then to cellBoxes within the cellGrid
        """
        for cellBox in self.cellBoxes:
            longLoc = windPoints.loc[(windPoints['long'] > cellBox.long) & (windPoints['long'] < (cellBox.long + cellBox.width))]
            latLongLoc = longLoc.loc[(windPoints['lat'] > cellBox.lat) & (windPoints['lat'] < (cellBox.lat + cellBox.height))]

            cellBox.addWindPoints(latLongLoc)
            cellBox.setLand()

    def cellCount(self):
        """
            Returns the number of cellBoxes contained within this cellGrid

            DEPRECIATED
            The list of cellBoxes now contains some empty indexes, so this is no longer accurate.
        """
        return len(self.cellBoxes)

    def toJSON(self):
        """
            Returns this cellGrid converted to JSON format.
        """
        json = "{ \"cellBoxes\":["
        for cellBox in self.cellBoxes:
            json += cellBox.toJSON() + ",\n"

        json = json[:-2] # remove last comma and newline
        json += "]}"
        return json

    def getCellBox(self, long, lat):
        """
            Returns the CellBox which contains a point, given by parameters lat, long
        """
        selectedCell = []
        for cellBox in self.cellBoxes:
            if isinstance(cellBox, CellBox):
                if cellBox.containsPoint(lat, long):
                    selectedCell.append(cellBox)
        return selectedCell

    def recursiveSplit(self, maxSplits):
        """
            Resursively step though all cellBoxes in this cellGrid, splitting them based on a cells 'isHomogenous' function
            and a cellBoxes split depth.

            DEPRECIATED
            This function does not update the neighbours graph and should not be used.
            use function 'iternativeSplit() instead.
        """
        splitCellBoxes = []
        for cellBox in self.cellBoxes:
            splitCellBoxes += cellBox.recursiveSplit(maxSplits,self.threshold,self.lowerBound,self.upperBound)

        self.cellBoxes = splitCellBoxes

    def splitAndReplace(self, cellBox):
        """
            Replaces a cellBox given by parameter 'cellBox' in this grid with 4 smaller cellBoxes representing
            the four corners of the given cellBox. A neighbours map is then created for each of the 4 new cellBoxes
            and the neighbours map for all surrounding cell boxes is updated.

        """
        splitCellBoxes = cellBox.split()

        self.cellBoxes += splitCellBoxes

        cellBoxIndx = self.cellBoxes.index(cellBox)

        northWestIndx = self.cellBoxes.index(splitCellBoxes[0])
        northEastIndx = self.cellBoxes.index(splitCellBoxes[1])
        southWestIndx = self.cellBoxes.index(splitCellBoxes[2])
        southEastIndx = self.cellBoxes.index(splitCellBoxes[3])

        southNeighbourIndx = self.neighbourGraph[cellBoxIndx][4]
        northNeighbourIndx = self.neighbourGraph[cellBoxIndx][-4]
        eastNeighbourIndx = self.neighbourGraph[cellBoxIndx][2]
        westNeighbourIndx = self.neighbourGraph[cellBoxIndx][-2]

        # Create neighbour map for SW split cell.
        SWneighbourMap = {1: [northEastIndx],
                          2: [southEastIndx],
                          3: [],
                          4: [],
                          -1: self.neighbourGraph[cellBoxIndx][-1],
                          -2: [],
                          -3: [],
                          -4: [northWestIndx]}

        for indx in southNeighbourIndx:
            if self.getNeighbourCase(self.cellBoxes[southWestIndx], self.cellBoxes[indx]) == 3:
                SWneighbourMap[3].append(indx)
            if self.getNeighbourCase(self.cellBoxes[southWestIndx], self.cellBoxes[indx]) == 4:
                SWneighbourMap[4].append(indx)
        for indx in westNeighbourIndx:
            if self.getNeighbourCase(self.cellBoxes[southWestIndx], self.cellBoxes[indx]) == -2:
                SWneighbourMap[-2].append(indx)
            if self.getNeighbourCase(self.cellBoxes[southWestIndx], self.cellBoxes[indx]) == -3:
                SWneighbourMap[-3].append(indx)

        self.neighbourGraph[southWestIndx] = SWneighbourMap

        # Create neighbour map for NW split cell
        NWneighbourMap = {1: [],
                          2: [northEastIndx],
                          3: [southEastIndx],
                          4: [southWestIndx],
                          -1: [],
                          -2: [],
                          -3: self.neighbourGraph[cellBoxIndx][-3],
                          -4: []}

        for indx in northNeighbourIndx:
            if self.getNeighbourCase(self.cellBoxes[northWestIndx], self.cellBoxes[indx]) == -4:
                NWneighbourMap[-4].append(indx)
            if self.getNeighbourCase(self.cellBoxes[northWestIndx], self.cellBoxes[indx]) == 1:
                NWneighbourMap[1].append(indx)
        for indx in westNeighbourIndx:
            if self.getNeighbourCase(self.cellBoxes[northWestIndx], self.cellBoxes[indx]) == -2:
                NWneighbourMap[-2].append(indx)
            if self.getNeighbourCase(self.cellBoxes[northWestIndx], self.cellBoxes[indx]) == -1:
                NWneighbourMap[-1].append(indx)

        self.neighbourGraph[northWestIndx] = NWneighbourMap

        # Create neighbour map for NE split cell
        NEneighbourMap = {1: self.neighbourGraph[cellBoxIndx][1],
                          2: [],
                          3: [],
                          4: [southEastIndx],
                          -1: [southWestIndx],
                          -2: [northWestIndx],
                          -3: [],
                          -4: []}

        for indx in northNeighbourIndx:
            if self.getNeighbourCase(self.cellBoxes[northEastIndx], self.cellBoxes[indx]) == -4:
                NEneighbourMap[-4].append(indx)
            if self.getNeighbourCase(self.cellBoxes[northEastIndx], self.cellBoxes[indx]) == -3:
                NEneighbourMap[-3].append(indx)
        for indx in eastNeighbourIndx:
            if self.getNeighbourCase(self.cellBoxes[northEastIndx], self.cellBoxes[indx]) == 2:
                NEneighbourMap[2].append(indx)
            if self.getNeighbourCase(self.cellBoxes[northEastIndx], self.cellBoxes[indx]) == 3:
                NEneighbourMap[3].append(indx)

        self.neighbourGraph[northEastIndx] = NEneighbourMap

        # Create neighbour map for SE split cell
        SEneighbourMap = {1: [],
                          2: [],
                          3: self.neighbourGraph[cellBoxIndx][3],
                          4: [],
                          -1: [],
                          -2: [southWestIndx],
                          -3: [northWestIndx],
                          -4: [northEastIndx]}

        for indx in southNeighbourIndx:
            if self.getNeighbourCase(self.cellBoxes[southEastIndx], self.cellBoxes[indx]) == 4:
                SEneighbourMap[4].append(indx)
            if self.getNeighbourCase(self.cellBoxes[southEastIndx], self.cellBoxes[indx]) == -1:
                SEneighbourMap[-1].append(indx)
        for indx in eastNeighbourIndx:
            if self.getNeighbourCase(self.cellBoxes[southEastIndx], self.cellBoxes[indx]) == 2:
                SEneighbourMap[2].append(indx)
            if self.getNeighbourCase(self.cellBoxes[southEastIndx], self.cellBoxes[indx]) == 1:
                SEneighbourMap[1].append(indx)

        self.neighbourGraph[southEastIndx] = SEneighbourMap

        # Update neighbour map of neighbours of the split box.

        # Update north neighbour map
        for indx in northNeighbourIndx:
            self.neighbourGraph[indx][4].remove(cellBoxIndx)

            crossingCase = self.getNeighbourCase(self.cellBoxes[indx], self.cellBoxes[northWestIndx])
            if crossingCase != 0:
                self.neighbourGraph[indx][crossingCase].append(northWestIndx)

            crossingCase = self.getNeighbourCase(self.cellBoxes[indx], self.cellBoxes[northEastIndx])
            if crossingCase != 0:
                self.neighbourGraph[indx][crossingCase].append(northEastIndx)

        # Update east neighbour map
        for indx in eastNeighbourIndx:
            self.neighbourGraph[indx][-2].remove(cellBoxIndx)

            crossingCase = self.getNeighbourCase(self.cellBoxes[indx], self.cellBoxes[northEastIndx])
            if crossingCase != 0:
                self.neighbourGraph[indx][crossingCase].append(northEastIndx)

            crossingCase = self.getNeighbourCase(self.cellBoxes[indx], self.cellBoxes[southEastIndx])
            if crossingCase != 0:
                self.neighbourGraph[indx][crossingCase].append(southEastIndx)

        # Update south neighbour map
        for indx in southNeighbourIndx:
            self.neighbourGraph[indx][-4].remove(cellBoxIndx)

            crossingCase = self.getNeighbourCase(self.cellBoxes[indx], self.cellBoxes[southEastIndx])
            if crossingCase != 0:
                self.neighbourGraph[indx][crossingCase].append(southEastIndx)

            crossingCase = self.getNeighbourCase(self.cellBoxes[indx], self.cellBoxes[southWestIndx])
            if crossingCase != 0:
                self.neighbourGraph[indx][crossingCase].append(southWestIndx)

        # Update west neighbour map
        for indx in westNeighbourIndx:
            self.neighbourGraph[indx][2].remove(cellBoxIndx)

            crossingCase = self.getNeighbourCase(self.cellBoxes[indx], self.cellBoxes[northWestIndx])
            if crossingCase != 0:
                self.neighbourGraph[indx][crossingCase].append(northWestIndx)

            crossingCase = self.getNeighbourCase(self.cellBoxes[indx], self.cellBoxes[southWestIndx])
            if crossingCase != 0:
                self.neighbourGraph[indx][crossingCase].append(southWestIndx)

        # Update corner neighbour maps
        northEastCornerIndx = self.neighbourGraph[cellBoxIndx][1]
        if len(northEastCornerIndx) > 0:
            self.neighbourGraph[northEastCornerIndx[0]][-1] = [northEastIndx]

        northWestCornerIndx = self.neighbourGraph[cellBoxIndx][-3]
        if len(northWestCornerIndx) > 0:
            self.neighbourGraph[northWestCornerIndx[0]][3] = [northWestIndx]

        southEastCornerIndx = self.neighbourGraph[cellBoxIndx][3]
        if len(southEastCornerIndx) > 0:
            self.neighbourGraph[southEastCornerIndx[0]][-3] = [southEastIndx]

        southWestCornerIndx = self.neighbourGraph[cellBoxIndx][-1]
        if len(southWestCornerIndx) > 0:
            self.neighbourGraph[southWestCornerIndx[0]][1] = [southWestIndx]

        splitContainer = {"northEast": northEastIndx,
                          "northWest": northWestIndx,
                          "southEast": southEastIndx,
                          "southWest": southWestIndx}

        self.cellBoxes[cellBoxIndx] = splitContainer
        self.neighbourGraph.pop(cellBoxIndx)

    def iterativeSplit(self, splitAmount,splittingPercentage, splitMinProp, splitMaxProp):
        """
            Iterates over all cellBoxes in the cellGrid a number of times defined by parameter 'splitAmount',
            splitting and replacing each one if it is not homogenous.
        """
        for i in range(0, splitAmount):
            self.splitGraph(splittingPercentage, splitMinProp, splitMaxProp)

    def splitGraph(self, splittingPercentage, splitMinProp, splitMaxProp):
        """
            Iterates once over all cellBoxes in the cellGrid, splitting and replacing each one if it is not homogenous.
        """
        for indx in range(0, len(self.cellBoxes) - 1):
            cellBox = self.cellBoxes[indx]
            if isinstance(cellBox, CellBox):
                if cellBox.shouldWeSplit(splittingPercentage, splitMinProp, splitMaxProp):
                    self.splitAndReplace(cellBox)

    def recursiveSplitAndReplace(self, cellBox, maxSplits):
        """
            Replaces a cellBox given by parameter 'cellBox' in this grid with 4 smaller cellBoxes representing the four
            corners of the given cellBox. Recursively repeats this process with each of the 4 smaller cellBoxs until
            either the cellBoxes 'isHomogenous' function is met, or the maximum split depth given by parameter 'maxSplits'
            is reached.
        """
        splitCellBoxes = cellBox.recursiveSplit(maxSplits)
        self.cellBoxes.remove(cellBox)
        self.cellBoxes += splitCellBoxes

    def getIndex(self, selectedCellBox):
        """
            Returns the index of the selected cell.

            DEPRECIATED
            Not needed as functionality is identical to self.cellBoxes.index(selectedCellBox) but slower.
        """
        cell_index = []
        for idx,cellBox in enumerate(self.cellBoxes):
            if selectedCellBox==cellBox:
                    cell_index.append(idx)
        return cell_index

    def cellBoxByNodeString(self, nodeString):
        for cellBox in self.cellBoxes:
            if isinstance(cellBox, CellBox):
                if cellBox.nodeString() == nodeString:
                    return cellBox

    def _getLeftNeightbours(self, selectedCellBox):
        """
            Returns a list of all cellBoxes touching the left-hand-side of a cellBox given by parameter 'selectedCellBox'.
            Also returns a list of indexes for the discovered cellBoxes

            DEPRECIATED
            Functionality replaced by self.neighbourGraph
        """
        leftNeightbours = []
        leftNeightbours_indx = []
        for idx,cellBox in enumerate(self.cellBoxes):
            if (cellBox.long + cellBox.width == selectedCellBox.long) and (cellBox.lat <= (selectedCellBox.lat + selectedCellBox.height)) and ((cellBox.lat + cellBox.height) >= selectedCellBox.lat):
                    leftNeightbours_indx.append(idx)
                    leftNeightbours.append(cellBox)
        return leftNeightbours,leftNeightbours_indx

    def _getRightNeightbours(self, selectedCellBox):
        """
            Returns a list of all cellBoxes touching the right-hand-side of a cellBox given by parameter 'selectedCellBox'.
            Also returns a list of indexes for the discovered cellBoxes

            DEPRECIATED
            Functionality replaced by self.neighbourGraph
        """
        rightNeightbours = []
        rightNeightbours_indx = []
        for idx,cellBox in enumerate(self.cellBoxes):
            if (cellBox.long == selectedCellBox.long + selectedCellBox.width) and (cellBox.lat <= (selectedCellBox.lat + selectedCellBox.height)) and ((cellBox.lat + cellBox.height) >= selectedCellBox.lat):
                rightNeightbours_indx.append(idx)
                rightNeightbours.append(cellBox)
        return rightNeightbours,rightNeightbours_indx

    def _getTopNeightbours(self, selectedCellBox):
        """
            Returns a list of all cellBoxes touching the top-side of a cellBox given by parameter 'selectedCellBox'.
            Also returns a list of indexes for the discovered cellBoxes

            DEPRECIATED
            Functionality replaced by self.neighbourGraph
        """
        topNeightbours      = []
        topNeightbours_indx = []
        for idx,cellBox in enumerate(self.cellBoxes):
            if (cellBox.lat == (selectedCellBox.lat + selectedCellBox.height)) and ((cellBox.long + cellBox.width) >= selectedCellBox.long) and (cellBox.long <= (selectedCellBox.long + selectedCellBox.width)):
                topNeightbours.append(cellBox)
                topNeightbours_indx.append(idx)
        return topNeightbours,topNeightbours_indx

    def _getBottomNeightbours(self, selectedCellBox):
        """
            Returns a list of all cellBoxes touching the bottom-side of a cellBox given by parameter 'selectedCellBox'.
            Also returns a list of indexes for the discovered cellBoxes

            DEPRECIATED
            Functionality replaced by self.neighbourGraph
        """
        bottomNeightbours      = []
        bottomNeightbours_indx = []
        for idx,cellBox in enumerate(self.cellBoxes):
            if ((cellBox.lat + cellBox.height) == selectedCellBox.lat) and ((cellBox.long + cellBox.width) >= selectedCellBox.long) and (cellBox.long <= (selectedCellBox.long + selectedCellBox.width)):
                bottomNeightbours_indx.append(idx)
                bottomNeightbours.append(cellBox)
        return bottomNeightbours,bottomNeightbours_indx

    def getNeightbours(self, selectedCellBox):
        """
            Returns a list of call cellBoxes touching a cellBox given by parameter 'selectedCellBox'.
            Also returns a list of indexes for the discovered cellBoxes

            DEPRECIATED
            Functionality replaced by self.neighbourGraph
        """
        leftNeightbours,leftNeightbours_indx     = self._getLeftNeightbours(selectedCellBox)
        rightNeightbours,rightNeightbours_indx   = self._getRightNeightbours(selectedCellBox)
        topNeightbours,topNeightbours_indx       = self._getTopNeightbours(selectedCellBox)
        bottomNeightbours,bottomNeightbours_indx = self._getBottomNeightbours(selectedCellBox)
        neightbours       = leftNeightbours + rightNeightbours + topNeightbours + bottomNeightbours
        neightbours_index = leftNeightbours_indx + rightNeightbours_indx + topNeightbours_indx + bottomNeightbours_indx
        return neightbours,neightbours_index,

    def plot(self, highlightCellBoxes = {}, plotIce = True, plotCurrents = False, plotBorders = True, paths=None, routepoints=False,waypoints=None):
        """
            creates and displays a plot for this cellGrid
        """
        # Create plot figure
        fig, ax = plt.subplots(1, 1, figsize=(25, 11))

        fig.patch.set_facecolor('white')
        ax.set_facecolor('lightblue')

        for cellBox in self.cellBoxes:
            if isinstance(cellBox, CellBox):
                # plot ice
                if plotIce and not np.isnan(cellBox.iceArea()):
                    if self._j_grid == True:
                        if cellBox.iceArea() >= 0.04:
                            ax.add_patch(
                                MatplotPolygon(cellBox.getBounds(), closed=True, fill=True, color='white', alpha=1))
                            if cellBox.iceArea() < 0.8:
                                ax.add_patch(MatplotPolygon(cellBox.getBounds(), closed=True, fill=True, color='grey',
                                                            alpha=(1 - cellBox.iceArea())))
                    else:
                        ax.add_patch(MatplotPolygon(cellBox.getBounds(), closed=True, fill=True, color='white', alpha=cellBox.iceArea()))

                # plot land
                if self._j_grid == True:
                    if cellBox.landLocked:
                        ax.add_patch(MatplotPolygon(cellBox.getBounds(), closed=True, fill=True, facecolor='lime'))
                else:
                    if cellBox.containsLand():
                        ax.add_patch(MatplotPolygon(cellBox.getBounds(), closed=True, fill=True, facecolor='mediumseagreen'))
                #else:
                    #ax.add_patch(MatplotPolygon(cellBox.getBounds(), closed=True, fill=True, facecolor='mediumseagreen'))


                # plot currents
                if plotCurrents:
                    ax.quiver((cellBox.long + cellBox.width / 2), (cellBox.lat + cellBox.height / 2),
                              cellBox.getuC(), cellBox.getvC(), scale=1, width=0.002, color='gray')

                # plot borders
                if plotBorders:
                    ax.add_patch(MatplotPolygon(cellBox.getBounds(), closed=True, fill=False, edgecolor='black'))

                """
                if self._j_grid == True:
                    # plot %iceArea text
                    if not np.isnan(cellBox.iceArea()):
                        ax.text(cellBox.long, cellBox.lat, str(math.floor(cellBox.iceArea() * 100)) + "%", fontsize=8)
                """

        # plot highlighted cells
        for colour in highlightCellBoxes:
            for cellBox in highlightCellBoxes[colour]:
                ax.add_patch(MatplotPolygon(cellBox.getBounds(),
                                            closed=True,
                                            fill=False,
                                            edgecolor=colour,
                                            linewidth = 0.5 + len(list(highlightCellBoxes.keys())) - list(highlightCellBoxes.keys()).index(colour)))

        # plot paths if supplied
        if type(paths) != type(None):
            for Path in paths:
                if Path['Time'] == np.inf:
                    continue
                Points = np.array(Path['Path']['Points'])
                if routepoints:
                    ax.plot(Points[:,0],Points[:,1],linewidth=3.0,color='b')
                    ax.scatter(Points[:,0],Points[:,1],30,zorder=99,color='b')
                else:
                    ax.plot(Points[:,0],Points[:,1],linewidth=3.0,color='b')


        if type(waypoints) != type(None):
            ax.scatter(waypoints['Long'],waypoints['Lat'],150,marker='^',color='r')

        ax.set_xlim(self._longMin, self._longMax)
        ax.set_ylim(self._latMin, self._latMax)

    def getCase(self,cell,ncell):
        """

        """
        SPoly = Polygon(selectedCellBox.getBounds())
        neightbours      = []
        neightbours_index = []
        for idx,cellBox in enumerate(self.cellBoxes):
            if cellBox != selectedCellBox:
                NPoly = Polygon(cellBox.getBounds())
                if SPoly.intersects(NPoly):
                    neightbours_index.append(idx)
                    neightbours.append(cellBox)
        cases = []
        crossing_points = []
        for ncell in neightbours:
            case  = self.getCase(selectedCellBox,(ncell.cx,ncell.cy))
            cp    = self.getCrossingPoint(selectedCellBox,(ncell.cx,ncell.cy))
            cases.append(case)
            crossing_points.append(cp)
        neigh = pd.DataFrame({'Cell':neightbours,'idx':neightbours_index,'Case':cases,'Cp':crossing_points})
        return neigh

    def getCrossingPoint(self,cell,Pt):
        case = self.getCase(cell,Pt)
        crp = Intersection_BoxLine(cell,Pt,case)
        return crp

    def getCase(self,cellBox,Pt):
        corners = []
        for crn in cellBox.getBounds()[:-1]:
            corners.append(bearing((cellBox.cx,cellBox.cy),(crn[0],crn[1])))
        corners = np.sort(corners)
        dbear = bearing((cellBox.cx,cellBox.cy),(Pt[0],Pt[1]))
        if  dbear == corners[0]:
            case = 1
        elif corners[0] < dbear < corners[1]:
            case=2
        elif dbear == corners[1]:
            case = 3
        elif corners[1] < dbear < corners[2]:
            case = 4
        elif dbear == corners[2]:
            case = -1
        elif corners[2] < dbear < corners[3]:
            case = -2
        elif dbear == corners[3]:
            case = -3
        elif (corners[3] > dbear) or (corners[0] < dbear):
            case = -4
        else:
            case = np.nan
        return case

    def toDataFrame(self):
        DF = pd.DataFrame({'idx':np.arange(len(self.cellBoxes))})
        shape   = []
        iceArea = []
        isLand  = []
        currentuC = []
        currentvC = []
        windv10 = []
        windu10 = []

        for cellBox in self.cellBoxes:
            shape.append(Polygon(cellBox.getBounds()))
            iceArea.append(cellBox.iceArea())
            isLand.append(cellBox.containsLand())
            currentuC.append(cellBox.getuC())
            currentvC.append(cellBox.getvC())
            windu10.append(cellBox.getWindu10())
            windv10.append(cellBox.getWindv10())

        DF['Geometry'] = shape
        DF['Ice Area'] = iceArea
        DF['Land']     = isLand
        DF['Current uC'] = currentuC
        DF['Current vC'] = currentvC
        DF['Wind v10'] = windv10
        DF['Wind u10'] = windu10

        DF = gpd.GeoDataFrame(DF, crs={'init': 'epsg:4326'}, geometry='Geometry')
        DF['Land'][np.isnan(DF['Ice Area'])] = True
        return DF

    def getNeightbours(self, selectedCellBox):
        """
            Getting the neighbours and returning idx, case, cp and cell information

            BUG - Currently this is very slow as the Polygon intersection is slower than before.
            Optimising the running of the code should improve this section as its a overarching requirement
            for all routeplanes etc

            DEPRECIATED
            Functionality replaced by self.neighbourGraph
        """
        SPoly = Polygon(selectedCellBox.getBounds())
        neightbours      = []
        neightbours_index = []
        for idx,cellBox in enumerate(self.cellBoxes):
            if cellBox != selectedCellBox:
                NPoly = Polygon(cellBox.getBounds())
                if SPoly.intersects(NPoly):
                    neightbours_index.append(idx)
                    neightbours.append(cellBox)
        cases = []
        crossing_points = []
        for ncell in neightbours:
            case  = self.getCase(selectedCellBox,(ncell.cx,ncell.cy))
            cp    = self.getCrossingPoint(selectedCellBox,(ncell.cx,ncell.cy))
            cases.append(case)
            crossing_points.append(cp)
        neigh = pd.DataFrame({'Cell':neightbours,'idx':neightbours_index,'Case':cases,'Cp':crossing_points})
        return neigh

    def getNeighboursNew(self, selectedCellBox):
        neighbours = {}
        for indx, cellBox in enumerate(self.cellBoxes):
            if self.getNeighbourCase(selectedCellBox, cellBox) != 0:
                neighbours[indx] = cellBox
        return neighbours

    def getNeighbourCase(self, cellBoxA, cellBoxB):
        """
            Given two cellBoxes (cellBoxA, cellBoxB) returns a case number representing where the two cellBoxes are touching.

            case 0 -> cellBoxes are not neighbours

            case 1 -> cellBoxB is the North-East corner of cellBoxA
            case 2 -> cellBoxB is East of cellBoxA
            case 3 -> cellBoxB is the South-East corner of cellBoxA
            case 4 -> cellBoxB is South of cellBoxA
            case -1 -> cellBoxB is the South-West corner of cellBoxA
            case -2 -> cellBoxB is West of cellBoxA
            case -3 -> cellBoxB is the North-West corner of cellBoxA
            case -4 -> cellBoxB is North of cellBoxA
        """

        if (cellBoxA.long + cellBoxA.width) == cellBoxB.long and (cellBoxA.lat + cellBoxA.height) == cellBoxB.lat:
            return 1  # North-East
        if (cellBoxA.long + cellBoxA.width == cellBoxB.long) and (
                cellBoxB.lat < (cellBoxA.lat + cellBoxA.height)) and (
                (cellBoxB.lat + cellBoxB.height) > cellBoxA.lat):
            return 2  # East
        if (cellBoxA.long + cellBoxA.width) == cellBoxB.long and (cellBoxA.lat == cellBoxB.lat + cellBoxB.height):
            return 3  # South-East
        if ((cellBoxB.lat + cellBoxB.height) == cellBoxA.lat) and (
                (cellBoxB.long + cellBoxB.width) > cellBoxA.long) and (
                cellBoxB.long < (cellBoxA.long + cellBoxA.width)):
            return 4  # South
        if cellBoxA.long == (cellBoxB.long + cellBoxB.width) and cellBoxA.lat == (cellBoxB.lat + cellBoxB.height):
            return -1  # South-West
        if (cellBoxB.long + cellBoxB.width == cellBoxA.long) and (
                cellBoxB.lat < (cellBoxA.lat + cellBoxA.height)) and (
                (cellBoxB.lat + cellBoxB.height) > cellBoxA.lat):
            return -2  # West
        if cellBoxA.long == (cellBoxB.long + cellBoxB.width) and (cellBoxA.lat + cellBoxA.height == cellBoxB.lat):
            return -3  # North-West
        if (cellBoxB.lat == (cellBoxA.lat + cellBoxA.height)) and (
                (cellBoxB.long + cellBoxB.width) > cellBoxA.long) and (
                cellBoxB.long < (cellBoxA.long + cellBoxA.width)):
            return -4  # North
        return 0  # Cells are not neighbours.

    def dumpMesh(self, fileLocation):
        meshDump = ""
        for cellBox in self.cellBoxes:
            if isinstance(cellBox, CellBox):
                meshDump += cellBox.meshDump()

        f = open(fileLocation, "w")
        f.write(meshDump)
        f.close()

    def dumpGraph(self, fileLocation):
        graphDump = ""

        maxIceArea = 0.8

        for cellBox in self.cellBoxes:
            if isinstance(cellBox, CellBox):
                if (not cellBox.landLocked) and cellBox.iceArea() < maxIceArea:
                    graphDump += cellBox.nodeString()

                    cellBoxIndx = self.cellBoxes.index(cellBox)

                    # case -3 neighbours
                    nwneighbourIndx = self.neighbourGraph[cellBoxIndx][-3]
                    for neighbour in nwneighbourIndx:
                        if (not self.cellBoxes[neighbour].landLocked) and self.cellBoxes[neighbour].iceArea() < maxIceArea:
                            graphDump += "," + self.cellBoxes[neighbour].nodeString() + ":-3"
                    # case -2 neighbours
                    wneighboursIndx = self.neighbourGraph[cellBoxIndx][-2]
                    for neighbour in wneighboursIndx:
                        if (not self.cellBoxes[neighbour].landLocked) and self.cellBoxes[neighbour].iceArea() < maxIceArea:
                            graphDump += "," + self.cellBoxes[neighbour].nodeString() + ":-2"
                    # case -1 neighbours
                    swneighboursIndx = self.neighbourGraph[cellBoxIndx][-1]
                    for neighbour in swneighboursIndx:
                        if (not self.cellBoxes[neighbour].landLocked) and self.cellBoxes[neighbour].iceArea() < maxIceArea:
                            graphDump += "," + self.cellBoxes[neighbour].nodeString() + ":-1"
                    # case -4 neighbours
                    nneighbourIndx = self.neighbourGraph[cellBoxIndx][-4]
                    for neighbour in nneighbourIndx:
                        if (not self.cellBoxes[neighbour].landLocked) and self.cellBoxes[neighbour].iceArea() < maxIceArea:
                            graphDump += "," + self.cellBoxes[neighbour].nodeString() + ":-4"
                    # case 4 neighbours
                    sneighboursIndx = self.neighbourGraph[cellBoxIndx][4]
                    for neighbour in sneighboursIndx:
                        if (not self.cellBoxes[neighbour].landLocked) and self.cellBoxes[neighbour].iceArea() < maxIceArea:
                            graphDump += "," + self.cellBoxes[neighbour].nodeString() + ":4"
                    # case 1 neighbours
                    neneighbourIndx = self.neighbourGraph[cellBoxIndx][1]
                    for neighbour in neneighbourIndx:
                        if (not self.cellBoxes[neighbour].landLocked) and self.cellBoxes[neighbour].iceArea() < maxIceArea:
                            graphDump += "," + self.cellBoxes[neighbour].nodeString() + ":1"
                    # case 2 neighbours
                    eneighbourIndx = self.neighbourGraph[cellBoxIndx][2]
                    for neighbour in eneighbourIndx:
                        if (not self.cellBoxes[neighbour].landLocked) and self.cellBoxes[neighbour].iceArea() < maxIceArea:
                            graphDump += "," + self.cellBoxes[neighbour].nodeString() + ":2"
                    # case 3 neighbours
                    seneighbourIndx = self.neighbourGraph[cellBoxIndx][3]
                    for neighbour in seneighbourIndx:
                        if (not self.cellBoxes[neighbour].landLocked) and self.cellBoxes[neighbour].iceArea() < maxIceArea:
                            graphDump += "," + self.cellBoxes[neighbour].nodeString() + ":3"

                    graphDump += "\n"



        f = open(fileLocation, "w")
        f.write(graphDump)
        f.close()
