import pandas as pd
import numpy as np
from matplotlib.patches import Polygon

class CellBox:
    # _icePoints = []
    # _currentPoints = []
    splitDepth = 0

    def __init__(self, lat, long, width, height):
        self.lat = lat
        self.long = long
        self.width = width
        self.height = height

    def addIcePoints(self, icePoints):
        self._icePoints = icePoints
        
    def addCurrentPoints(self, currentPoints):
        self._currentPoints = currentPoints

    def getIcePointLength(self):
        return len(self._icePoints)
    
    def getCurrentPointLength(self):
        return len(self._currentPoints)

    def getLatRange(self):
        return str(self.lat) + " to " + str(self.lat + self.height)

    def getLongRange(self):
        return str(self.long) + " to " + str(self.long + self.width)

    def getRange(self):
        return "Lat Range: " + self.getLatRange() + ", Long Range: " + self.getLongRange()
    
    def getPolygon(self):
        bounds = [[self.long, self.lat],
                    [self.long, self.lat + self.height],
                    [self.long + self.width, self.lat + self.height],
                    [self.long + self.width, self.lat],
                    [self.long, self.lat]]

        return Polygon(bounds, closed = True, fill = True, color = 'White', alpha = self.iceArea())
    
    def iceArea(self):
        return self._icePoints['iceArea'].mean()

    def getuC(self):
        return self._currentPoints['uC'].mean()
    
    def getvC(self):
        return self._currentPoints['vC'].mean()
    
    def getIcePoints(self):
        return self._icePoints

    def toString(self):
        s = ""
        s += self.getRange() + "\n"
        s += "    No. of IcePoint: " + str(self.getIcePointLength()) + "\n"
        s += "    Ice Area: " + str(self.iceArea()) + "\n"
        s += "    split Depth: " + str(self.splitDepth) + "\n"
        s += "    uC: " + str(self.getuC()) + "\n"
        s += "    vC: " + str(self.getvC())
        return s

    # convert cellBox to JSON
    def toJSON(self):
        s = "{"
        s += "\"lat\":" + str(self.lat) + ","
        s += "\"long\":" + str(self.long) + ","
        s += "\"width\":" + str(self.width) + ","
        s += "\"height\":" + str(self.height) + ","
        s += "\"iceArea\":" + str(self.iceArea()) + ","
        s += "\"splitDepth\":" + str(self.splitDepth)
        s += "}"
        return s

    # returns true or false if a cell is deemd homogenous, used to define a base case for recursive splitting.
    def isHomogenous(self):
        lowerBound = 0.05
        upperBound = 0.99

        if self.iceArea() < lowerBound:
            return True
        if self.iceArea() > upperBound:
            return True
        return False

    # splits the current cellbox into 4 corners, returns as a list of cellbox objects.
    def split(self):
        splitBoxes = []

        halfWidth = self.width / 2
        halfHeight = self.height / 2

        # create 4 new cellBoxes
        bottomLeft = CellBox(self.lat, self.long, halfWidth, halfHeight)
        bottomRight = CellBox(self.lat, self.long + halfWidth, halfWidth, halfHeight)
        topLeft = CellBox(self.lat + halfHeight, self.long, halfWidth, halfHeight)
        topRight = CellBox(self.lat + halfHeight, self.long + halfWidth, halfWidth, halfHeight)

        splitBoxes.append(bottomLeft)
        splitBoxes.append(bottomRight)
        splitBoxes.append(topLeft)
        splitBoxes.append(topRight)

        for splitBox in splitBoxes:
            splitBox.splitDepth = self.splitDepth + 1
            
            #Split icePoints per box
            longLoc = self._icePoints.loc[(self._icePoints['long'] > splitBox.long) & (
                        self._icePoints['long'] < (splitBox.long + splitBox.width))]
            latLongLoc = self._icePoints.loc[
                (self._icePoints['lat'] > splitBox.lat) & (self._icePoints['lat'] < (splitBox.lat + splitBox.height))]
            
            splitBox.addIcePoints(latLongLoc)
            
            #Split currentPoints per box
            longLoc = self._currentPoints.loc[(self._currentPoints['long'] > splitBox.long) & (
                        self._currentPoints['long'] < (splitBox.long + splitBox.width))]
            latLongLoc = self._currentPoints.loc[
                (self._currentPoints['lat'] > splitBox.lat) & (self._currentPoints['lat'] < (splitBox.lat + splitBox.height))]
            
            splitBox.addCurrentPoints(latLongLoc)

        return splitBoxes

    def recursiveSplit(self, maxSplits):
        splitCells = []
        if self.isHomogenous() or (self.splitDepth >= maxSplits):
            splitCells.append(self)
            return splitCells
        else:
            splitBoxes = self.split()
            for splitBox in splitBoxes:
                splitCells = splitCells + splitBox.recursiveSplit(maxSplits)
            return splitCells
