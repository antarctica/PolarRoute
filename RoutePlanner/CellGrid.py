from turtle import fillcolor
import numpy as np
from RoutePlanner.CellBox import CellBox
from RoutePlanner.Function import NewtonianDistance
import matplotlib.pylab as plt
from matplotlib.patches import Polygon

class CellGrid:
    
    def __init__(self, OptInfo):
        self.OptInfo = OptInfo

        self._longMin    = self.OptInfo['Bounds Longitude'][0] 
        self._longMax    = self.OptInfo['Bounds Longitude'][1]
        self._latMin     = self.OptInfo['Bounds Latitude'][0]
        self._latMax     = self.OptInfo['Bounds Latitude'][1]
        
        self._cellWidth  = self.OptInfo['Grid Spacing (dx,dy)'][0]
        self._cellHeight = self.OptInfo['Grid Spacing (dx,dy)'][1]
        
        self.cellBoxes = []

        for long in np.arange(self._longMin, self._longMax, self._cellWidth):
            for lat in np.arange(self._latMin, self._latMax, self._cellHeight):
                cellBox = CellBox(lat, long, self._cellWidth, self._cellHeight)
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
            if cellBox.isLand():
                ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = True, facecolor='mediumseagreen'))
                ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = False, edgecolor='gray'))
                continue


            iceArea = cellBox.iceArea()
            if iceArea >= self.OptInfo['MaxIceExtent']:
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
        return neightbours,neightbours_index
        
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