import numpy as np
from RoutePlanner.CellBox import CellBox
import pandas as pd
from shapely.geometry import Polygon
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

    def getCellBox(self, long, lat):
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

    def getIndex(self, selectedCellBox):
        """
            Returns the index of the selected cell
        """
        cell_index = []
        for idx,cellBox in enumerate(self.cellBoxes):
            if selectedCellBox==cellBox:
                    cell_index.append(idx)
        return cell_index

    def getNeightbours(self, selectedCellBox):
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
            case,cp = self.getCase(selectedCellBox,ncell)
            cases.append(case)
            crossing_points.append(cp)
        neigh = pd.DataFrame({'Cell':neightbours,'idx':neightbours_index,'Case':cases,'Cp':crossing_points})
        return neigh

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
            case = np.nan

        crp = Intersection_BoxLine(cell,ncell,case)
        
        return case,crp