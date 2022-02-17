from matplotlib.patches import Polygon
import math

class CellBox:
    splitDepth = 0

    def __init__(self, lat, long, width, height):
        # Box information relative to bottom left
        self.lat    = lat
        self.long   = long
        self.width  = width
        self.height = height

        # Defining the initial centroid information for cell
        # TODO move these out of object attributes at to get methods.
        self.cx     = self.long + self.width/2
        self.cy     = self.lat  + self.height/2

    
        # TODO move these out of object attributes at to get methods.
        self.dcx  = self.width/2
        self.dcy  = self.height/2


        # Minimum Depth to be used in the land mask
        self.minDepth = 10


    def addIcePoints(self, icePoints):
        '''
            updates the ice points contained within this cellBox to a pandas dataframe provided by parameter icePoints. 
        '''
        self._icePoints = icePoints
        
    def addCurrentPoints(self, currentPoints):
        '''
            updates the current points contained within this cellBox to a pandas dataframe provided by parameter currentPoints.
        '''
        self._currentPoints = currentPoints

    def getIcePointLength(self):
        '''
            Returns the number of ice points contained within this cellBox. 
        '''
        return len(self._icePoints)
    
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
    
    def getBorder(self):
        '''
            Returns a polygon object representing a grey border around this cellBox, to be used when plotting. 
        '''          
        bounds = [[self.long, self.lat],
                    [self.long, self.lat + self.height],
                    [self.long + self.width, self.lat + self.height],
                    [self.long + self.width, self.lat],
                    [self.long, self.lat]]
        return Polygon(bounds, closed = True, fill = False, color = 'Grey', alpha = 1)
    


    def getBounds(self):
        bounds = [[self.long, self.lat],
                    [self.long, self.lat + self.height],
                    [self.long + self.width, self.lat + self.height],
                    [self.long + self.width, self.lat],
                    [self.long, self.lat]]
        return bounds


    def getHighlight(self):
        '''
            Returns polygon object representing a red border around this cellBox, to be used when plotting. 
        '''  
        bounds = [[self.long, self.lat],
                    [self.long, self.lat + self.height],
                    [self.long + self.width, self.lat + self.height],
                    [self.long + self.width, self.lat],
                    [self.long, self.lat]]
        
        return Polygon(bounds, closed = True, fill = False, color = 'Red', alpha = 1)
    
    def getWidth(self):
        '''
            INCLUDE 
        '''          
        return self.width * math.cos(self.lat)
    
    def iceArea(self):
        """
            Returns mean ice area of all icepoints contained within this cellBox
        """
        return self._icePoints['iceArea'].mean()
    
    def depth(self):
        '''
            Returns mean depth of all icepoints contained within this cellBox
        '''          
        return self._icePoints['depth'].mean()

    def getuC(self):
        '''
            INCLUDE 
        '''          
        return self._currentPoints['uC'].mean()
    
    def getvC(self):
        '''
            INCLUDE 
        '''  
        return self._currentPoints['vC'].mean()
    
    def getIcePoints(self):
        '''
            Returns a pandas dataframe of all icepoints contained within this cellBox 
        '''  
        return self._icePoints

            
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
        s += "    No. of IcePoint: " + str(self.getIcePointLength()) + "\n"
        s += "    Ice Area: " + str(self.iceArea()) + "\n"
        s += "    split Depth: " + str(self.splitDepth) + "\n"
        s += "    uC: " + str(self.getuC()) + "\n"
        s += "    vC: " + str(self.getvC()) + "\n"
        s += "    depth: " + str(self.depth())
        return s

    # convert cellBox to JSON
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
            Returns True if any icepoint within the cell has a depth less than the specifed minimum depth.
        """
        depthList = self._icePoints['depth']
        
        if (depthList < self.minDepth).any():
            return True
        return False
    
    def isLand(self):
        """
            Returns True if all icepoints within the cell have a depth less than the specified minimum depth.
        """
        depthList = self._icePoints['depth']
        if (depthList < self.minDepth).all():
            return True
        return False  
    
    def isHomogenous(self):
        '''
            returns true if a cell is deemed homogenous, used to define a base case for recursive splitting. 
        '''  
        # if a cell contains only land, it is homogenous and does not require splitting
        if self.isLand():
            return True
        # if a cell contains both land and sea, it not homogenous and requires splitting
        if self.containsLand():
            return False
        
        # TODO first interpretation of sea ice homogeneity. Requires refinement
        threshold = 0.04
        
        percentIPsAboveThreshold = self._icePoints.loc[self._icePoints['iceArea'] > threshold].size / self._icePoints.size
        
        lowerBound = 0.05
        upperBound = 0.90
        
        if percentIPsAboveThreshold < lowerBound:
            return True
        if percentIPsAboveThreshold > upperBound:
            return True
        
        return False
        

    def split(self):
        '''
            splits the current cellbox into 4 corners, returns as a list of cellbox objects.
        '''  

        splitBoxes = []

        halfWidth = self.width / 2
        halfHeight = self.height / 2

        # create 4 new cellBoxes
        bottomLeft  = CellBox(self.lat, self.long, halfWidth, halfHeight)
        bottomRight = CellBox(self.lat, self.long + halfWidth, halfWidth, halfHeight)
        topLeft     = CellBox(self.lat + halfHeight, self.long, halfWidth, halfHeight)
        topRight    = CellBox(self.lat + halfHeight, self.long + halfWidth, halfWidth, halfHeight)

        splitBoxes.append(bottomLeft)
        splitBoxes.append(bottomRight)
        splitBoxes.append(topLeft)
        splitBoxes.append(topRight)

        for splitBox in splitBoxes:
            #TODO requires rework for optimization
            splitBox.splitDepth = self.splitDepth + 1
            
            #Split icePoints per cellBox
            longLoc = self._icePoints.loc[(self._icePoints['long'] >= splitBox.long) & 
                                          (self._icePoints['long'] < (splitBox.long + splitBox.width))]
            latLongLoc = longLoc.loc[(self._icePoints['lat'] >= splitBox.lat) & 
                                             (self._icePoints['lat'] < (splitBox.lat + splitBox.height))]
            
            splitBox.addIcePoints(latLongLoc)
            
            #Split currentPoints per box
            longLoc = self._currentPoints.loc[(self._currentPoints['long'] >= splitBox.long) & 
                                              (self._currentPoints['long'] < (splitBox.long + splitBox.width))]
            latLongLoc = longLoc.loc[(self._currentPoints['lat'] >= splitBox.lat) & 
                                                 (self._currentPoints['lat'] < (splitBox.lat + splitBox.height))]
            
            splitBox.addCurrentPoints(latLongLoc)

        return splitBoxes

    def recursiveSplit(self, maxSplits):
        '''
            Recursively splits this cellBox until all split cellBoxes are considered homogenous (defined by the isHomogenous() function) 
            or a the cellBox has reached a maximum split depth, given by parameter maxSplits.
        '''  
        splitCells = []
        #base case for recursive splitting. Do not split a cell if it is homogenous or the maximum split depth has been reached
        if self.isHomogenous() or (self.splitDepth >= maxSplits): 
            splitCells.append(self)
            return splitCells
        else:
            splitBoxes = self.split()
            for splitBox in splitBoxes:
                splitCells = splitCells + splitBox.recursiveSplit(maxSplits)
            return splitCells
