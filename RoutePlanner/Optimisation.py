import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from scipy import optimize
import math
from matplotlib.patches import Polygon

from RoutePlanner.CellBox import CellBox
from RoutePlanner.Mesh import Mesh

def _F(y,x,a,Y,u1,v1,u2,v2,s):
    '''
        Minimisation function of ...

        Variable definitions correspond to those given in the
        paper ??? Figure 4

        Inputs:
          x     - Left cell horizontal distance to right edge from centroid
          y     - Crossing point right of box relative to left cell position
          a     - Right cell horizontal distance to left edge from centroid
          u1,v1 - Current vector in left cell
          u2,v2 - Current vector in right cell
          s     - vesal speed in cells
          Y     - Vertical difference between left and right cell

        Outputs:
    '''
    d1 = x**2 + y**2
    d2 = a**2 + (Y-y)**2
    C1 = s**2 - u1**2 - v1**2
    C2 = s**2 - u2**2 - v2**2
    D1 = x*u1 + y*v1
    D2 = a*u2 + (Y-y)*v2
    X1 = np.sqrt(D1**2 + C1*(d1**2))
    X2 = np.sqrt(D2**2 + C2*(d2**2))
    # Minimisation Function
    F  = X2*(y-((v1*(X1-D1))/C1)) + X1*(y-Y+((v2*(X2-D2))/C2)) 
    return F

def _dF(y,x,a,Y,u1,v1,u2,v2,s):
    '''
        Analytical Differentiation function of ...

        Variable definitions correspond to those given in the
        paper ??? Figure 4

        Inputs:
          x     - Left cell horizontal distance to right edge from centroid
          y     - Crossing point right of box relative to left cell position
          a     - Right cell horizontal distance to left edge from centroid
          u1,v1 - Current vector in left cell
          u2,v2 - Current vector in right cell
          s     - vesal speed in cells
          Y     - Vertical difference between left and right cell

        Outputs:
    '''
    d1 = x**2 + y**2
    d2 = a**2 + (Y-y)**2
    C1 = s**2 - u1**2 - v1**2
    C2 = s**2 - u2**2 - v2**2
    D1 = x*u1 + y*v1
    D2 = a*u2 + (Y-y)*v2
    X1 = np.sqrt(D1**2 + C1*(d1**2))
    X2 = np.sqrt(D2**2 + C2*(d2**2))
    # Analytical Derivatives
    dD1 = v1
    dD2 = -v2
    dX1 = (D1*v1 + C1*y)/X1
    dX2 = (D2*v2 - C1*(Y-y))/X1
    # Derivative Function
    dF = (X1+X2) + y*(dX1 + dX2) - (v1/C1)*(dX2*(X1-D1) + X2*(dX1-dD1)) + (v2/C2)*(dX1*(X2-D2)+X1*(dX2-dD2)) - Y*dX1
    return dF

def _T(y,x,a,Y,u1,v1,u2,v2,s):
    '''
        Indivdual Travel-time between two adjacent Cells given the current field

        Variable definitions correspond to those given in the
        paper ??? Figure 4

        Inputs:
          x     - Left cell horizontal distance to right edge from centroid
          y     - Crossing point right of box relative to left cell position
          a     - Right cell horizontal distance to left edge from centroid
          u1,v1 - Current vector in left cell
          u2,v2 - Current vector in right cell
          s     - vesal speed in cells
          Y     - Vertical difference between left and right cell

        Outputs:
    '''
    d1 = x**2 + y**2
    d2 = a**2 + (Y-y)**2
    C1 = s**2 - u1**2 - v1**2
    C2 = s**2 - u2**2 - v2**2
    D1 = x*u1 + y*v1
    D2 = a*u2 + (Y-y)*v2
    X1 = np.sqrt(D1**2 + C1*(d1**2))
    X2 = np.sqrt(D2**2 + C2*(d2**2))
    t1 = (X1-D1)/C1
    t2 = (X2-D2)/C2
    T  = t1+t2 
    return T


def _Haversine_distance(origin, destination):
    """
    Calculate the Haversine distance between two points 
    Inputs:
      origin      - tuple of floats e.g. (Lat_orig,Long_orig)
      destination - tuple of floats e.g. (Lat_dest,Long_dest)
    Output:
      Distance - Distance between two points in 'km'

    """
    lon1,lat1 = origin
    lon2,lat2 = destination
    radius = 6371  # Radius of earth

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) * math.sin(dlat / 2) +
          math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
          math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

def _Euclidean_distance(origin, destination):
    """
    Replicating original route planner Euclidean distance 
    Inputs:
      origin      - tuple of floats e.g. (Long_orig,Lat_orig)
      destination - tuple of floats e.g. (Long_dest,Lat_dest)
    Output:
      Distance - Distance between two points in 'km'

    """
    lon1,lat1 = origin
    lon2,lat2 = destination

    kmperdeglat          = 111.386
    kmperdeglonAtEquator = 111.321

    d = np.sqrt(((lat2-lat1)*kmperdeglat)**2 + ((lon2-lon1)*kmperdeglonAtEquator)**2)

    return d



class TravelTime:
    def __init__(self,Mesh,OptInfo):
        # Load in the current cell structure & Optimisation Info
        self.Mesh    = Mesh
        self.OptInfo = OptInfo

        # Initialising Waypoint information
        self.OptInfo['WayPoints']['Index'] = np.nan
        for idx,wpt in enumerate(self.OptInfo['WayPoints'].iterrows()):
            Long = wpt[1]['Long']
            Lat  = wpt[1]['Lat']
            for index, cell in enumerate(self.Mesh.cells):
                if (Long>=cell.x) and (Long<=(cell.x+cell.dx)) and (Lat>=cell.y) and (Lat<=(cell.y+cell.dy)):
                    break
            self.OptInfo['WayPoints']['Index'].iloc[idx] = index


        # Initialising the Dijkstra Info Dictionary
        self.DijkstraInfo = {}
            
    def value(self,index,cells):
        '''
        Function for computing the shortest travel-time from a cell to its neighbours by applying the Newtonian method for optimisation
        
        Inputs:
        index - Index of the cell to process
        
        Output:

        Bugs/Alterations:
            - Include travel-time referencing cx and cy instead of dx,dy to allow for off-centre waypoints
            - Correct case definitions for when the centroid not at centre
        
        '''
        # Determining the nearest neighbour index for the cell
        neighbours = self.Mesh.NearestNeighbours(index)

        # Creating Blank travel-time array
        TravelTime = np.zeros((len(neighbours)))

        # Determining if point is a waypoint
        waypoint_list = self.OptInfo['WayPoints']['Index'].astype(int).to_list()   
        Cell_s = self.Mesh.cells[index]
        if index in waypoint_list:
            Cell_s.cx = Cell_s.cx#self.OptInfo['WayPoints']['Long'].iloc[waypoint_list.index(index)]
            Cell_s.cy = Cell_s.cy#self.OptInfo['WayPoints']['Lat'].iloc[waypoint_list.index(index)]
                        
        # Looping over all the nearest neighbours 
        for lp_index, neighbour_index in enumerate(neighbours):
            # Determining the cell for the neighbour
            Cell_n = self.Mesh.cells[neighbour_index]
            if neighbour_index in waypoint_list:
                Cell_n.cx = Cell_n.cx#self.OptInfo['WayPoints']['Long'].iloc[waypoint_list.index(neighbour_index)]
                Cell_n.cy = Cell_n.cy#self.OptInfo['WayPoints']['Lat'].iloc[waypoint_list.index(neighbour_index)]

            # Set travel-time to infinite if neighbour is land or ice-thickness is too large.
            if (Cell_n.value >= self.Mesh.meshinfo['IceExtent']['MaxProportion']) or (Cell_n.isLand):
                TravelTime[lp_index] = np.inf

            # Determine relative degree difference between source and neighbour
            df_x = Cell_n.cx - Cell_s.cx
            df_y = Cell_n.cy - Cell_s.cy
            s    = self.OptInfo['VehicleInfo']['Speed']
            
            # Longitude
            if (abs(df_x) > (Cell_s.dx/2)) and (abs(df_y) < (Cell_s.dy/2)):
                u1 = np.sign(df_x)*Cell_s.vector[0]; v1 = Cell_s.vector[1]
                u2 = np.sign(df_x)*Cell_n.vector[0]; v2 = Cell_n.vector[1]
                x  = _Euclidean_distance( (Cell_s.cx,Cell_s.cy), (Cell_s.cx + np.sign(df_x)*Cell_s.dx,Cell_s.cy))
                a  = _Euclidean_distance( (Cell_n.cx,Cell_n.cy), (Cell_n.cx - np.sign(df_x)*Cell_n.dx,Cell_n.cy))
                Y  = _Euclidean_distance((Cell_s.cx + np.sign(df_x)*(Cell_n.dx + Cell_s.dx), Cell_s.cy), (Cell_s.cx + np.sign(df_x)*(Cell_n.dx + Cell_s.dx),Cell_n.cy))
                y  = optimize.newton(_F,0.0,args=(x,a,Y,u1,v1,u2,v2,s),fprime=_dF)
                TravelTime[lp_index] = _T(y,x,a,Y,u1,v1,u2,v2,s)
            # Latitude
            elif (abs(df_x) < Cell_s.dx/2) and (abs(df_y) > Cell_s.dy/2):
                u1 = np.sign(df_y)*Cell_s.vector[1]; v1 = Cell_s.vector[0]
                u2 = np.sign(df_y)*Cell_n.vector[1]; v2 = Cell_n.vector[0]
                x  = _Euclidean_distance( (Cell_s.cy,Cell_s.cx), (Cell_s.cy + np.sign(df_y)*Cell_s.dy,Cell_s.cx))
                a  = _Euclidean_distance( (Cell_n.cy,Cell_n.cx), (Cell_n.cy - np.sign(df_y)*Cell_n.dy,Cell_n.cx))
                Y  = _Euclidean_distance((Cell_s.cy + np.sign(df_y)*(Cell_n.dy + Cell_s.dy), Cell_s.cx), (Cell_s.cy + np.sign(df_y)*(Cell_n.dy + Cell_s.dy),Cell_n.cx))
                y  = optimize.newton(_F,0.0,args=(x,a,Y,u1,v1,u2,v2,s),fprime=_dF)
                TravelTime[lp_index] = _T(y,x,a,Y,u1,v1,u2,v2,s)
            # Corner
            else:
                u1 = np.sign(df_x)*Cell_s.vector[0]; v1 = Cell_s.vector[1]
                u2 = np.sign(df_x)*Cell_n.vector[0]; v2 = Cell_n.vector[1]
                x  = _Euclidean_distance( (Cell_s.cx,Cell_s.cy), (Cell_s.cx + np.sign(df_x)*Cell_s.dx,Cell_s.cy))
                a  = _Euclidean_distance( (Cell_n.cx,Cell_n.cy), (Cell_n.cx - np.sign(df_x)*Cell_n.dx,Cell_n.cy))
                Y  = _Euclidean_distance((Cell_s.cx + np.sign(df_x)*(Cell_n.dx + Cell_s.dx), Cell_s.cy), (Cell_s.cx + np.sign(df_x)*(Cell_n.dx + Cell_s.dx),Cell_n.cy))
                y  = np.sign((Cell_n.cy - Cell_s.cy))*(Cell_s.dy/2)
                u1 = np.sign(df_y)*Cell_s.vector[1]; v1 = Cell_s.vector[0]
                TravelTime[lp_index] = _T(y,x,a,Y,u1,v1,u2,v2,s)

        return neighbours, TravelTime

    def optimize(self):
        '''
        Determining the shortest path between all waypoints
        '''
        for wpt in self.OptInfo['WayPoints'].iterrows():
            wpt_name  = wpt[1]['Name']
            wpt_index = wpt[1]['Index']
            #Initialising a column array of all the indexs
            self.DijkstraInfo[wpt_name] = {}
            self.DijkstraInfo[wpt_name]['CellIndex']       = np.arange(len(self.Mesh.cells))
            self.DijkstraInfo[wpt_name]['Cost']            = np.full((len(self.Mesh.cells)),np.inf)
            self.DijkstraInfo[wpt_name]['PositionLocked']  = np.zeros((len(self.Mesh.cells)),dtype=bool)
            self.DijkstraInfo[wpt_name]['Paths']           = {}
            for djk in range(len(self.Mesh.cells)):
                self.DijkstraInfo[wpt_name]['Paths'][djk]  = [wpt_index]
            self.DijkstraInfo[wpt_name]['Cost'][wpt_index] = 0.0

            while (self.DijkstraInfo[wpt_name]['PositionLocked'] == False).any():
                # Determining the argument with the next lowest value and hasn't been visited
                idx = self.DijkstraInfo[wpt_name]['CellIndex'][(self.DijkstraInfo[wpt_name]['PositionLocked']==False)][np.argmin(self.DijkstraInfo[wpt_name]['Cost'][(self.DijkstraInfo[wpt_name]['PositionLocked']==False)])]
                # Finding the cost of the nearest neighbours
                Neighbour_index,Points,TT = self.value(idx)
                Neighbour_cost = TT + self.DijkstraInfo[wpt_name]['Cost'][idx]
                # Determining if the visited time is visited    
                for jj_v,jj in enumerate(Neighbour_index):
                    if Neighbour_cost[jj_v] < self.DijkstraInfo[wpt_name]['Cost'][jj]:
                        self.DijkstraInfo[wpt_name]['Cost'][jj]  = Neighbour_cost[jj_v]
                        self.DijkstraInfo[wpt_name]['Paths'][jj] = self.DijkstraInfo[wpt_name]['Paths'][idx] + [jj]
                # Defining the graph point as visited
                self.DijkstraInfo[wpt_name]['PositionLocked'][idx] = True
    
    def smooth(self):
        '''
            Given the current optimum paths smooth the pathways smooth the pathway between based on Great-circle smoothing
        '''
