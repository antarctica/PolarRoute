import pandas as pd
import numpy as np
from refactor.CellBox import CellBox
from matplotlib.patches import Polygon
import matplotlib.pylab as plt

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
        for cellBox in self.cellBoxes:
            longLoc = icePoints.loc[(icePoints['long'] > cellBox.long) & (icePoints['long'] < (cellBox.long + cellBox.width))]
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
    
    """
    def getCellBox(self, lat, long):
        selectedCell = []
        
        for cellBox in self.cellBoxes:
            if(cellBox.lat == lat and cellBox.long == long):
                selectedCell.append(cellBox) 
        
        # for inital debugging and should be replaced to throw errors correctly
        if len(selectedCell) == 1:
            return selectedCell[0]
        elif len(selectedCell) == 0:
            return "No cellBox was found at lat =" + str(lat) + ", long =" + str(long)
        elif len(selectedCell) > 1:
            return "ERROR: Multiple cellBoxes have been found at lat =" + str(lat) + ", long =" + str(long) 
    """
    
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
        
    def plot(self):
        fig, ax = plt.subplots(1,1, figsize=(15,10))
        ax.set_facecolor('xkcd:blue')
        for cellBox in self.cellBoxes:
            ax.add_patch(cellBox.getPolygon())
            ax.add_patch(cellBox.getBorder())
        
        ax.set_xlim(self._longMin, self._longMax)
        ax.set_ylim(self._latMin, self._latMax)
        
    def highlightCell(self, selectedCellBox):
        fig, ax = plt.subplots(1,1, figsize=(15,10))
        ax.set_facecolor('xkcd:blue')
        for cellBox in self.cellBoxes:
            ax.add_patch(cellBox.getPolygon())
        
        ax.add_patch(selectedCellBox.getBorder())
        
        ax.set_xlim(self._longMin, self._longMax)
        ax.set_ylim(self._latMin, self._latMax)