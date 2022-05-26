from tracemalloc import start
import numpy as np
from RoutePlanner.CellBox import CellBox
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MatplotPolygon
import math
import xarray as xr
from shapely.geometry import Polygon
import geopandas as gpd

class CellGrid:

    def __init__(self, config, j_grid=False):
        self.config = config

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
                                    splitting_conditions = [], j_grid = self._j_grid)
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
        self.splitting_conditions = []
        for dataSource in config['Data_sources']:
                self.addDataSource(dataSource)

        self.iterativeSplit(config['Region']["splitDepth"])

    # Functions for adding data to the cellgrid
    def addDataSource(self, dataSource):

        for value in dataSource['values']:
            if "splittingCondition" in value:
                    splittingCondition = {value['destinationName'] : value['splittingCondition']}
                    self.splitting_conditions = self.splitting_conditions + [splittingCondition]
    
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
                    cellBox.add_splitting_condition(splittingCondition)

            df = df.dropna(subset = selected)

            cellBox.add_data_points(df)

    def add_data_points(self, dataPoints):
        for cellBox in self.cellBoxes:
            longLoc = dataPoints.loc[(dataPoints['long'] > cellBox.long) & (dataPoints['long'] <= (cellBox.long + cellBox.width))]
            latLongLoc = longLoc.loc[(longLoc['lat'] > cellBox.lat) & (longLoc['lat'] <= (cellBox.lat + cellBox.height))]

            cellBox.add_data_points(latLongLoc)

    # Functions for outputting the cellgrid
    def output_dataframe(self):
        """
            requires rework as to not used hard-coded data types.
        """
        cellgrid_dataframe = []
        counter=0

        for idx,c in enumerate(self.cellBoxes):
            if isinstance(c, CellBox):
                # # Don't append cell if Ice or above threshold
                # if c.iceArea() >= self.config['Vehicle_Info']['MaxIceExtent']:
                #     continue
                # if self._j_grid:
                #     if c.isLandM():
                #         continue
                # else:
                #     if c.containsLand():
                #         continue


                # Inspecting neighbour graph and outputting in list
                neigh = self.neighbourGraph[idx]
                cases      = []
                neigh_indx = []
                for case in neigh.keys():
                    indxs = neigh[case]
                    if len(indxs) == 0:
                        continue
                    for indx in indxs:
                        if (self.cellBoxes[indx].get_value('iceArea')*100 > self.config['Vehicle_Info']['MaxIceExtent']):
                            continue
                        if self._j_grid:
                            if self.cellBoxes[indx].isLandM():
                                continue
                        else:
                            if self.cellBoxes[indx].contains_land():
                                continue
                        cases.append(case)
                        neigh_indx.append(indx)

                if self._j_grid:
                    IsLand = c.isLandM()
                else:
                    IsLand = c.contains_land()

                index_df = pd.Series({'Index':int(idx),
                        'geometry':Polygon(c.get_bounds()),
                        'cell_info':[c.getcx(),c.getcy(),c.getdcx(),c.getdcy()],
                        'case':cases,
                        'neighbourIndex':neigh_indx,
                        'Land':IsLand,
                        'Ice Area':c.get_value('iceArea')*100,
                        'Ice Thickness':c.ice_thickness(self.config['Region']['startTime']),
                        'Ice Density':c.ice_density(self.config['Region']['startTime']),
                        'Depth': c.get_value('depth'),
                        'Vector':[c.get_value('uC'),c.get_value('vC')]
                        })

                cellgrid_dataframe.append(index_df)

        cellgrid_dataframe = pd.concat(cellgrid_dataframe,axis=1).transpose()

        ## Cell Further South than -78.0 set to land.
        cellgrid_dataframe['Land'][np.array([x[1] for x in cellgrid_dataframe['cell_info']]) < -78.0] = True 

        cellgrid_dataframe = gpd.GeoDataFrame(cellgrid_dataframe,crs={'init': 'epsg:4326'}, geometry='geometry')
        return cellgrid_dataframe

    # Functions for spltting cellboxes within the cellgrid
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

    def iterativeSplit(self, splitAmount):
        """
            Iterates over all cellBoxes in the cellGrid a number of times defined by parameter 'splitAmount',
            splitting and replacing each one if it is not homogenous.
        """
        for i in range(0, splitAmount):
            self.splitGraph()

    def splitGraph(self):
        """
            Iterates once over all cellBoxes in the cellGrid, splitting and replacing each one if it is not homogenous.
        """
        for indx in range(0, len(self.cellBoxes) - 1):
            cellBox = self.cellBoxes[indx]
            if isinstance(cellBox, CellBox):
                if cellBox.should_be_split():
                    self.splitAndReplace(cellBox)

    # Functions for debugging
    def plot(self, highlightCellBoxes = {}, plotIce = True, plotCurrents = False, plotBorders = True, paths=None, routepoints=False,waypoints=None):
        """
            creates and displays a plot for this cellGrid

            To be used for debugging purposes only.
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
                                MatplotPolygon(cellBox.get_bounds(), closed=True, fill=True, color='white', alpha=1))
                            if cellBox.iceArea() < 0.8:
                                ax.add_patch(MatplotPolygon(cellBox.get_bounds(), closed=True, fill=True, color='grey',
                                                            alpha=(1 - cellBox.iceArea())))
                    else:
                        ax.add_patch(MatplotPolygon(cellBox.get_bounds(), closed=True, fill=True, color='white', alpha=cellBox.iceArea()))

                # plot land
                if self._j_grid == True:
                    if cellBox.landLocked:
                        ax.add_patch(MatplotPolygon(cellBox.get_bounds(), closed=True, fill=True, facecolor='lime'))
                else:
                    if cellBox.contains_land():
                        ax.add_patch(MatplotPolygon(cellBox.get_bounds(), closed=True, fill=True, facecolor='mediumseagreen'))
                #else:
                    #ax.add_patch(MatplotPolygon(cellBox.get_bounds(), closed=True, fill=True, facecolor='mediumseagreen'))


                # plot currents
                if plotCurrents:
                    ax.quiver((cellBox.long + cellBox.width / 2), (cellBox.lat + cellBox.height / 2),
                              cellBox.getuC(), cellBox.getvC(), scale=1, width=0.002, color='gray')

                # plot borders
                if plotBorders:
                    ax.add_patch(MatplotPolygon(cellBox.get_bounds(), closed=True, fill=False, edgecolor='black'))

                """
                if self._j_grid == True:
                    # plot %iceArea text
                    if not np.isnan(cellBox.iceArea()):
                        ax.text(cellBox.long, cellBox.lat, str(math.floor(cellBox.iceArea() * 100)) + "%", fontsize=8)
                """

        # plot highlighted cells
        for colour in highlightCellBoxes:
            for cellBox in highlightCellBoxes[colour]:
                ax.add_patch(MatplotPolygon(cellBox.get_bounds(),
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

    def getCellBox(self, long, lat):
        """
            Returns the CellBox which contains a point, given by parameters lat, long
        """
        selectedCell = []
        for cellBox in self.cellBoxes:
            if isinstance(cellBox, CellBox):
                if cellBox.contains_point(lat, long):
                    selectedCell.append(cellBox)
        return selectedCell

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

    # Functions used for j_grid regression testing
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

    def addCurrentPoints(self, currentPoints):
        """
            Takes a dataframe containing current points and assigns then to cellBoxes within the cellGrid
        """
        for cellBox in self.cellBoxes:
            longLoc = currentPoints.loc[(currentPoints['long'] > cellBox.long) & (currentPoints['long'] <= (cellBox.long + cellBox.width))]
            latLongLoc = longLoc.loc[(longLoc['lat'] > cellBox.lat) & (longLoc['lat'] <= (cellBox.lat + cellBox.height))]

            cellBox.addCurrentPoints(latLongLoc)
            cellBox.setLand()

    def cellBoxByNodeString(self, nodeString):
        for cellBox in self.cellBoxes:
            if isinstance(cellBox, CellBox):
                if cellBox.nodeString() == nodeString:
                    return cellBox

