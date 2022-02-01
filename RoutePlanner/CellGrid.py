import numpy as np
from RoutePlanner.CellBox import CellBox
from RoutePlanner.Function import NewtonianDistance
import matplotlib.pylab as plt

class CellGrid:
    
    def __init__(self, longMin, longMax, latMin, latMax, cellWidth, cellHeight):
        self._longMin    = longMin
        self._longMax    = longMax
        self._latMin     = latMin
        self._latMax     = latMax
        
        self._cellWidth  = cellWidth
        self._cellHeight = cellHeight
        
        self.cellBoxes = []

        for long in np.arange(longMin, longMax, cellWidth):
            for lat in np.arange(latMin, latMax, cellHeight):
                cellBox = CellBox(lat, long, cellWidth, cellHeight)
                self.cellBoxes.append(cellBox)
    
    def addIcePoints(self, icePoints):
        for cellBox in self.cellBoxes:
            longLoc    = icePoints.loc[(icePoints['long'] > cellBox.long) & (icePoints['long'] < (cellBox.long + cellBox.width))]
            latLongLoc = longLoc.loc[(icePoints['lat'] > cellBox.lat) & (icePoints['lat'] < (cellBox.lat + cellBox.height))]
            cellBox.addIcePoints(latLongLoc)
        
    def addCurrentPoints(self, currentPoints):
        for cellBox in self.cellBoxes:
            longLoc = currentPoints.loc[(currentPoints['long'] > cellBox.long) & (currentPoints['long'] < (cellBox.long + cellBox.width))]
            latLongLoc = longLoc.loc[(currentPoints['lat'] > cellBox.lat) & (currentPoints['lat'] < (cellBox.lat + cellBox.height))]
    
            cellBox.addCurrentPoints(latLongLoc)
    
    def cellCount(self):
        return len(self.cellBoxes)
    
    def toJSON(self):
        json = "{ \"cellBoxes\":["
        for cellBox in self.cellBoxes:
            json += cellBox.toJSON() + ",\n"
            
        json = json[:-2] # remove last comma and newline
        json += "]}"
        return json
    
    def getCellBox(self, lat, long):
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
        splitCellBoxes = []
        for cellBox in self.cellBoxes:
            splitCellBoxes += cellBox.recursiveSplit(maxSplits)
            
        self.cellBoxes = splitCellBoxes
        
    def splitAndReplace(self, cellBox):
        splitCellBoxes = cellBox.split()
        self.cellBoxes.remove(cellBox)
        self.cellBoxes += splitCellBoxes
        
    def recursiveSplitAndReplace(self, cellBox, maxSplits):
        splitCellBoxes = cellBox.recursiveSplit(maxSplits)
        self.cellBoxes.remove(cellBox)
        self.cellBoxes += splitCellBoxes
        
    def plot(self,figInfo=None,currents=False,return_ax=False,iceThreshold=None):
        if type(figInfo) == type(None):
            fig,ax = plt.subplots(1,1,figsize=(15,10))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('lightblue')
            ax.patch.set_alpha(1.0)
        else:
            fig,ax = figInfo

        for cellBox in self.cellBoxes:
            if type(iceThreshold) != type(None):
                if cellBox.iceArea() >= iceThreshold:
                    qp = ax.add_patch(cellBox.getPolygon())
                    qp.set_hatch('/')
                else:
                    ax.add_patch(cellBox.getPolygon())
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
        leftNeightbours = []
        leftNeightbours_indx = []
        for idx,cellBox in enumerate(self.cellBoxes):
            if (cellBox.long + cellBox.width == selectedCellBox.long) and (cellBox.lat <= (selectedCellBox.lat + selectedCellBox.height)) and ((cellBox.lat + cellBox.height) >= selectedCellBox.lat):
                    leftNeightbours_indx.append(idx)
                    leftNeightbours.append(cellBox)
        return leftNeightbours,leftNeightbours_indx           
    def _getRightNeightbours(self, selectedCellBox):
        rightNeightbours = []
        rightNeightbours_indx = []
        for idx,cellBox in enumerate(self.cellBoxes):
            if (cellBox.long == selectedCellBox.long + selectedCellBox.width) and (cellBox.lat <= (selectedCellBox.lat + selectedCellBox.height)) and ((cellBox.lat + cellBox.height) >= selectedCellBox.lat):
                rightNeightbours_indx.append(idx)
                rightNeightbours.append(cellBox)
        return rightNeightbours,rightNeightbours_indx
    
    def _getTopNeightbours(self, selectedCellBox):
        topNeightbours      = []
        topNeightbours_indx = []
        for idx,cellBox in enumerate(self.cellBoxes):
            if (cellBox.lat == (selectedCellBox.lat + selectedCellBox.height)) and ((cellBox.long + cellBox.width) >= selectedCellBox.long) and (cellBox.long <= (selectedCellBox.long + selectedCellBox.width)):
                topNeightbours.append(cellBox)
                topNeightbours_indx.append(idx)
        return topNeightbours,topNeightbours_indx
    
    def _getBottomNeightbours(self, selectedCellBox):
        bottomNeightbours      = []
        bottomNeightbours_indx = []
        for idx,cellBox in enumerate(self.cellBoxes):
            if ((cellBox.lat + cellBox.height) == selectedCellBox.lat) and ((cellBox.long + cellBox.width) >= selectedCellBox.long) and (cellBox.long <= (selectedCellBox.long + selectedCellBox.width)):
                bottomNeightbours_indx.append(idx)
                bottomNeightbours.append(cellBox)
        return bottomNeightbours,bottomNeightbours_indx
            
    
    def getNeightbours(self, selectedCellBox):
        leftNeightbours,leftNeightbours_indx     = self._getLeftNeightbours(selectedCellBox)
        rightNeightbours,rightNeightbours_indx   = self._getRightNeightbours(selectedCellBox)
        topNeightbours,topNeightbours_indx       = self._getTopNeightbours(selectedCellBox)
        bottomNeightbours,bottomNeightbours_indx = self._getBottomNeightbours(selectedCellBox)
        neightbours       = leftNeightbours + rightNeightbours + topNeightbours + bottomNeightbours
        neightbours_index = leftNeightbours_indx + rightNeightbours_indx + topNeightbours_indx + bottomNeightbours_indx
        return neightbours,neightbours_index
        
    def NewtonNeighbourCells(self,selectedCellBox,figInfo=None,localBounds=False,return_ax=True,shipSpeed=26.3*(1000/(60*60)),debugging=False):
        if type(figInfo) == type(None):
            fig,ax = plt.subplots(1,1,figsize=(15,10))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('lightblue')
            ax.patch.set_alpha(1.0)
        else:
            fig,ax = figInfo

        for cellBox in self.cellBoxes:
            ax.add_patch(cellBox.getPolygon())

        # Plotting the neighbour cell boxes
        selectedCellBoxes,idx = self.getNeightbours(selectedCellBox)
        bounds=[]
        for cellBox in selectedCellBoxes:
            bounds.append([cellBox.long,cellBox.lat])
            bounds.append([cellBox.long+cellBox.width,cellBox.lat+cellBox.height])
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
        
        if return_ax:
            return ax

    def highlightCell(self, selectedCellBox, figInfo=None):
        if type(figInfo) == type(None):
            fig,ax = plt.subplots(1,1,figsize=(15,10))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('lightblue')
        else:
            fig,ax = figInfo

        for cellBox in self.cellBoxes:
            ax.add_patch(cellBox.getPolygon())
        
        ax.add_patch(selectedCellBox.getHighlight())
        
        ax.set_xlim(self._longMin, self._longMax)
        ax.set_ylim(self._latMin, self._latMax)