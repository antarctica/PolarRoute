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
        self.cx     = self.long + self.width/2
        self.cy     = self.lat  + self.height/2

        # Defining the Upper-bound (ub) and Lower-bound(lb) width from waypoints
        self.cx_ub  = self.width/2
        self.cx_lb  = self.width/2
        self.cy_ub  = self.height/2
        self.cy_lb  = self.height/2

        # Minimum Depth to be used in the land mask
        self.minDepth = 10

    def _define_waypoints(self,pt):
        self.cx,self.cy = pt
        self.cx_ub = (self.long + self.width) - self.cx
        self.cx_lb = self.cx - self.long
        self.cy_ub = (self.lat + self.height) - self.cy
        self.cy_lb = self.cy - self.lat


    def addIcePoints(self, icePoints):
        '''
            INCLUDE 
        '''
        self._icePoints = icePoints
        
    def addCurrentPoints(self, currentPoints):
        '''
            INCLUDE 
        '''
        self._currentPoints = currentPoints

    def getIcePointLength(self):
        '''
            INCLUDE 
        '''
        return len(self._icePoints)
    
    def getCurrentPointLength(self):
        '''
            INCLUDE 
        '''
        return len(self._currentPoints)

    def getLatRange(self):
        '''
            INCLUDE 
        '''        
        return str(self.lat) + " to " + str(self.lat + self.height)

    def getLongRange(self):
        '''
            INCLUDE 
        '''

        return str(self.long) + " to " + str(self.long + self.width)

    def getRange(self):
        '''
            INCLUDE 
        '''        
        return "Lat Range: " + self.getLatRange() + ", Long Range: " + self.getLongRange()
    
    def getPolygon(self,fill=True):
        '''
            INCLUDE 
        '''        
        bounds = [[self.long, self.lat],
                    [self.long, self.lat + self.height],
                    [self.long + self.width, self.lat + self.height],
                    [self.long + self.width, self.lat],
                    [self.long, self.lat]]
        if self.isLand() == False:
            return Polygon(bounds, closed = True, fill = fill, color = 'White', alpha = self.iceArea())
        return Polygon(bounds, closed = True, fill = True, color = 'mediumseagreen', alpha=1)
        
    def getBorder(self):
        '''
            INCLUDE 
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
            INCLUDE 
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
        return self._icePoints['iceArea'].mean()
    
    def depth(self):
        '''
            INCLUDE 
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
            INCLUDE 
        '''  
        return self._icePoints
    
    def isLand(self):
        if self.depth() <= self.minDepth:
            return True
        return False
            
    def containsPoint(self, lat, long):
        if (lat > self.lat) & (lat < self.lat + self.height):
            if (long > self.long) & (long < self.long + self.width):
                return True
        return False

    def toString(self):
        '''
            INCLUDE 
        '''  
        s = ""
        s += self.getRange() + "\n"
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

    def isHomogenous(self):
        '''
            returns true or false if a cell is deemed homogenous, used to define a base case for recursive splitting. 
        '''  

        lowerBound = 0.15
        upperBound = 0.75
        

        # If a cell contains any point which is considered land, return False
        depthList = self._icePoints['depth']
        # If a cell contains only points condsidered land, return True
        if (depthList < 10).all():
            return True
        if (depthList < 10).any():
            return False
    
        if self.iceArea() < lowerBound:
            return True
        if self.iceArea() > upperBound:
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
            splitBox.splitDepth = self.splitDepth + 1
            
            #Split icePoints per box
            longLoc    = self._icePoints.loc[(self._icePoints['long'] >= splitBox.long) & 
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
            INCLUDE 
        '''  
        splitCells = []
        if self.isHomogenous() or (self.splitDepth >= maxSplits):
            splitCells.append(self)
            return splitCells
        else:
            splitBoxes = self.split()
            for splitBox in splitBoxes:
                splitCells = splitCells + splitBox.recursiveSplit(maxSplits)
            return splitCells
