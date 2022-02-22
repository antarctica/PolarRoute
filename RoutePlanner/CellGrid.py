from turtle import fillcolor
import numpy as np
from RoutePlanner.CellBox import CellBox
from RoutePlanner.Function import NewtonianDistance
import matplotlib.pylab as plt
from matplotlib.patches import Polygon

class CellGrid:

    def __init__(self, longMin, longMax, latMin, latMax, cellWidth, cellHeight):
        self._longMin = longMin
        self._longMax = longMax
        self._latMin = latMin
        self._latMax = latMax

        self._cellWidth = cellWidth
        self._cellHeight = cellHeight

        self.cellBoxes = []

        for long in np.arange(longMin, longMax, cellWidth):
            for lat in np.arange(latMin, latMax, cellHeight):
                cellBox = CellBox(lat, long, cellWidth, cellHeight)
                self.cellBoxes.append(cellBox)

    def addIcePoints(self, icePoints):
        """
            Takes a dataframe containing ice points and assigns them to cellBoxes within the cellGrid
        """
        for cellBox in self.cellBoxes:
            longLoc    = icePoints.loc[(icePoints['long'] > cellBox.long) & (icePoints['long'] < (cellBox.long + cellBox.width))]
            latLongLoc = longLoc.loc[(icePoints['lat'] > cellBox.lat) & (icePoints['lat'] < (cellBox.lat + cellBox.height))]
            cellBox.addIcePoints(latLongLoc)

    def addCurrentPoints(self, currentPoints):
        """
            Takes a dataframe containing current points and assigns then to cellBoxes within the cellGrid
        """
        for cellBox in self.cellBoxes:
            longLoc = currentPoints.loc[(currentPoints['long'] > cellBox.long) & (currentPoints['long'] < (cellBox.long + cellBox.width))]
            latLongLoc = longLoc.loc[(currentPoints['lat'] > cellBox.lat) & (currentPoints['lat'] < (cellBox.lat + cellBox.height))]

            cellBox.addCurrentPoints(latLongLoc)

    def cellCount(self):
        """
            Returns the number of cellBoxes contained within this cellGrid
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

    def getCellBox(self, lat, long):
        """
            Returns the CellBox which contains a point, given by parameters lat, long
        """
        selectedCell = []

        for cellBox in self.cellBoxes:
            if cellBox.containsPoint(lat, long):
                selectedCell.append(cellBox)

        # for inital debugging and should be replaced to throw errors correctly
        if len(selectedCell) == 1:
            return selectedCell[0]
        elif len(selectedCell) == 0:
            return "No cellBox was found at lat =" + str(lat) + ", long =" + str(long)
        elif len(selectedCell) > 1:
            return "ERROR: Multiple cellBoxes have been found at lat =" + str(lat) + ", long =" + str(long)

    def recursiveSplit(self, maxSplits):
        """
            Resursively step though all cellBoxes in this cellGrid, splitting them based on a cells 'isHomogenous' function
            and a cellBoxes split depth.
        """
        splitCellBoxes = []
        for cellBox in self.cellBoxes:
            splitCellBoxes += cellBox.recursiveSplit(maxSplits)

        self.cellBoxes = splitCellBoxes

    def splitAndReplace(self, cellBox):
        """
            Replaces a cellBox given by parameter 'cellBox' in this grid with 4 smaller cellBoxes representing
            the four corners of the given cellBox
        """
        splitCellBoxes = cellBox.split()
        self.cellBoxes.remove(cellBox)
        self.cellBoxes += splitCellBoxes

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

    def plot(self,figInfo=None,currents=False,return_ax=False,iceThreshold=None):
        """
            plots this cellGrid for display.

            TODO - requires reworking as part of the plotting work-package
        """
        if type(figInfo) == type(None):
            fig,ax = plt.subplots(1,1,figsize=(15,10))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('lightblue')
        else:
            fig,ax = figInfo

        for cellBox in self.cellBoxes:
            if cellBox.containsLand():
                ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = True, facecolor='mediumseagreen'))
                ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = False, edgecolor='gray'))
                continue


            iceArea = cellBox.iceArea()
            if iceArea >= 0.8:
                ax.add_patch(Polygon(cellBox.getBounds(),closed=True,fill=True,color='White'))
                ax.add_patch(Polygon(cellBox.getBounds(),closed=True,fill=True,color='Pink',alpha=0.4))
                ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = False,edgecolor='gray'))
            elif not np.isnan(iceArea):
                ax.add_patch(Polygon(cellBox.getBounds(),closed=True,fill=True,color='White',alpha=iceArea))
                ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = False,edgecolor='gray'))
            else:
                ax.add_patch(cellBox.getPolygon())

            ax.add_patch(cellBox.getPolygon())
            ax.add_patch(cellBox.getBorder())


            if currents:
                ax.quiver((cellBox.long+cellBox.width/2),(cellBox.lat+cellBox.height/2),cellBox.getuC()*1000,cellBox.getvC()*1000,scale=2,width=0.002,color='gray')

        ax.set_xlim(self._longMin, self._longMax)
        ax.set_ylim(self._latMin, self._latMax)

        if return_ax:
            return ax

    def getIndex(self, selectedCellBox):
        """
            Returns the index of the selected cell
        """
        cell_index = []
        for idx,cellBox in enumerate(self.cellBoxes):
            if selectedCellBox==cellBox:
                    cell_index.append(idx)
        return cell_index

    def _getLeftNeightbours(self, selectedCellBox):
        """
            Returns a list of all cellBoxes touching the left-hand-side of a cellBox given by parameter 'selectedCellBox'.
            Also returns a list of indexes for the discovered cellBoxes
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
        """
        leftNeightbours,leftNeightbours_indx     = self._getLeftNeightbours(selectedCellBox)
        rightNeightbours,rightNeightbours_indx   = self._getRightNeightbours(selectedCellBox)
        topNeightbours,topNeightbours_indx       = self._getTopNeightbours(selectedCellBox)
        bottomNeightbours,bottomNeightbours_indx = self._getBottomNeightbours(selectedCellBox)
        neightbours       = leftNeightbours + rightNeightbours + topNeightbours + bottomNeightbours
        neightbours_index = leftNeightbours_indx + rightNeightbours_indx + topNeightbours_indx + bottomNeightbours_indx
        return neightbours,neightbours_index,

    def highlightCells(self, selectedCellBoxes, figInfo=None):
        """
            Adds a red-border to cellBoxes gien by parameter 'selectedCellBoxes' and plots this cellGrid.
            TODO - requires rework as part of the plotting work-package
        """
        if type(figInfo) == type(None):
            fig,ax = plt.subplots(1,1,figsize=(15,10))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('lightblue')
        else:
            fig,ax = figInfo

        for cellBox in self.cellBoxes:
            ax.add_patch(cellBox.getPolygon())

        for cellBox in selectedCellBoxes:
            ax.add_patch(cellBox.getHighlight())
            ax.quiver((cellBox.long+cellBox.width/2),(cellBox.lat+cellBox.height/2),cellBox.getuC()*0.5,cellBox.getvC()*0.5,scale=2,width=0.002,color='gray')
        bounds = np.array(bounds)

        for cellBox in selectedCellBoxes:
            # Determining Newton crossing points
            TravelTime, CrossPoints, CellPoints = NewtonianDistance(selectedCellBox,cellBox,shipSpeed,shipSpeed,debugging=debugging).value()
            ax.plot([selectedCellBox.long+selectedCellBox.width/2,CrossPoints[0],cellBox.long+cellBox.width/2],\
                        [selectedCellBox.lat+selectedCellBox.height/2,CrossPoints[1],cellBox.lat+cellBox.height/2],marker='o',color='k')
            ax.text(cellBox.long+cellBox.width/2,cellBox.lat+cellBox.height/2,'{:.2f}'.format(TravelTime),color='r',zorder=100)

        if localBounds:
            ax.set_xlim([bounds[:,0].min(),bounds[:,0].max()])
            ax.set_ylim([bounds[:,1].min(),bounds[:,1].max()])
        else:
            ax.set_xlim(self._longMin, self._longMax)
            ax.set_ylim(self._latMin, self._latMax)

    def highlightCell(self, selectedCellBox, figInfo=None):
        """
            Adds a red-border to a cellBox given by parameter 'selectedCellBox' and plots this cellGrid.
            TODO - requires rework as part of the plotting work-package
        """
        if type(figInfo) == type(None):
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
        else:
            fig,ax = figInfo

        for cellBox in self.cellBoxes:
            ax.add_patch(cellBox.getPolygon())

        ax.add_patch(selectedCellBox.getHighlight())

        ax.set_xlim(self._longMin, self._longMax)
        ax.set_ylim(self._latMin, self._latMax)

    def getCase(self,cell,ncell):
        """
        
        """
        def Intersection_BoxLine(Cell_s,Cell_n,type):
            X1,Y1 = Cell_s.long+Cell_s.width/2,Cell_s.lat+Cell_s.height/2
            X2,Y2 = Cell_n.long+Cell_n.width/2,Cell_n.lat+Cell_n.height/2
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

            # Defining the corner cases
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


            try:
                D  = (X1-X2)*(Y3-Y4) - (Y1-Y2)*(X3-X4)
                Px = ((X1*Y2 - Y1*X2)*(X3-X4) - (X1-X2)*(X3*Y4-Y3*X4))/D
                Py = ((X1*Y2-Y1*X2)*(Y3-Y4)-(Y1-Y2)*(X3*Y4-Y3*X4))/D
                return Px,Py
            except:
                Px = np.nan
                Py = np.nan
                return Px,Py 

        def bearing(st,en):
            long1,lat1 = st 
            long2,lat2 = en  
            dlong = long2-long1
            dlat  = lat2-lat1
            vector_1 = [0, 1]
            vector_2 = [dlong, dlat]

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



        corners = []
        for crn in cell.getBounds()[:-1]:
            corners.append(bearing((cell.cx,cell.cy),(crn[0],crn[1])))
        corners = np.sort(corners)

        dbear = bearing((cell.cx,cell.cy),(ncell.cx,ncell.cy))

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
            case=np.nan

        crp = Intersection_BoxLine(cell,ncell,case)
        
        return case,crp

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