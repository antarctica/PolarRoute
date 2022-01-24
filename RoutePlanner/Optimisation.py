import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from scipy import optimize
import math
from matplotlib.patches import Polygon
import copy

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


def _Haversine_distance(origin, destination,forward=True):
    """
    Calculate the Haversine distance between two points 
    Inputs:
      origin      - tuple of floats e.g. (Lat_orig,Long_orig)
      destination - tuple of floats e.g. (Lat_dest,Long_dest)
    Output:
      Distance - Distance between two points in 'km'

    """
    R = 6371  # Radius of earth
    def haversine(theta):
        return math.sin(theta/2) ** 2

    def deg2rad(deg,forward=True):
        if forward:
            d = deg * (math.pi/180)
        else:
            d = deg * (180/math.pi)
        return d

    def distance(pa,pb):
        a_long,a_lat = pa
        b_long,b_lat = pb

        lat1  = deg2rad(a_lat)
        lat2  = deg2rad(b_lat)
        dLat  = deg2rad(a_lat - b_lat)
        dLong = deg2rad(a_long - b_long)
        x     =  haversine(dLat) + math.cos(lat1)*math.cos(lat2)*haversine(dLong)
        c     = 2*math.atan2(math.sqrt(x), math.sqrt(1-x))
        return R*c  

    def point(pa,dist):
        # Determining the latituted difference in Long & Lat
        a_long,a_lat = pa
        distX,distY  = dist

        lat1  = deg2rad(a_lat)
        dLat   = deg2rad(distX/R,forward=False)
        dLong  = deg2rad(2*math.asin(math.sqrt(haversine(distY/R)/(math.cos(lat1)**2))),forward=False)
        b_long = a_long + dLong
        b_lat  = a_lat + dLat

        return [b_long,b_lat]

    if forward:
        val = distance(origin,destination)
    else:
        val = point(origin,destination)
    return val

def _Euclidean_distance(origin, dest_dist,forward=True):
    """
    Replicating original route planner Euclidean distance 
    Inputs:
      origin      - tuple of floats e.g. (Long_orig,Lat_orig)
      destination - tuple of floats e.g. (Long_dest,Lat_dest)
      Optional: forward - Boolean True or False
    Output:
      Value - If 'forward' is True then returns Distance between 
              two points in 'km'. If 'False' then return the 
              Lat/Long position of a point.

    """


    kmperdeglat          = 111.386
    kmperdeglonAtEquator = 111.321
    if forward:
        lon1,lat1 = origin
        lon2,lat2 = dest_dist
        val = np.sqrt(((lat2-lat1)*kmperdeglat)**2 + ((lon2-lon1)*kmperdeglonAtEquator)**2)
    else:
        lon1,lat1     = origin
        dist_x,dist_y = dest_dist        
        val = [lon1+(dist_x/kmperdeglonAtEquator),lat1+(dist_y/kmperdeglat)]

    return val



class TravelTime:
    def __init__(self,Mesh,OptInfo,fdist=_Euclidean_distance):
        # Load in the current cell structure & Optimisation Info
        self.Mesh    = copy.copy(Mesh)
        self.OptInfo = copy.copy(OptInfo)


        # Dropping Wapoints not within range
        self.OptInfo['WayPoints'] = self.OptInfo['WayPoints'][(self.OptInfo['WayPoints']['Long'] >= self.Mesh.xmin) & (self.OptInfo['WayPoints']['Long'] <= self.Mesh.xmax) & (self.OptInfo['WayPoints']['Lat'] <= self.Mesh.ymax) & (self.OptInfo['WayPoints']['Lat'] >= self.Mesh.ymin)] 


        # Initialising Waypoint information
        self.OptInfo['WayPoints']['Index'] = np.nan
        for idx,wpt in enumerate(self.OptInfo['WayPoints'].iterrows()):
            Long = wpt[1]['Long']
            Lat  = wpt[1]['Lat']
            for index, cell in enumerate(self.Mesh.cells):
                if (Long>=cell.x) and (Long<=(cell.x+cell.dx)) and (Lat>=cell.y) and (Lat<=(cell.y+cell.dy)):
                    break
            self.OptInfo['WayPoints']['Index'].iloc[idx] = index

        self.fdist = fdist

        # Initialising the Dijkstra Info Dictionary
        self.DijkstraInfo = {}
        self.Paths = {}

        self._paths_smoothed = False

    def _newtonian(self,Cell_n,Cell_s):

            # Determine relative degree difference between source and neighbour
            df_x = Cell_n.cx - Cell_s.cx
            df_y = Cell_n.cy - Cell_s.cy
            s    = self.OptInfo['VehicleInfo']['Speed']
            
            # Longitude
            if ((abs(df_x) > (Cell_s.dx/2)) and (abs(df_y) < (Cell_s.dy/2))):
                try:
                    u1 = np.sign(df_x)*Cell_s.vector[0]; v1 = Cell_s.vector[1]
                    u2 = np.sign(df_x)*Cell_n.vector[0]; v2 = Cell_n.vector[1]
                    if np.sign(df_x) == 1:
                        S_dx = Cell_s.dxp; N_dx = -Cell_n.dxm
                    else:
                        S_dx = -Cell_s.dxm; N_dx = Cell_n.dxp                        
                    x  = self.fdist( (Cell_s.cx,Cell_s.cy), (Cell_s.cx + S_dx,Cell_s.cy))
                    a  = self.fdist( (Cell_n.cx,Cell_n.cy), (Cell_n.cx + N_dx,Cell_n.cy))
                    Y  = self.fdist((Cell_s.cx + np.sign(df_x)*(abs(S_dx) + abs(N_dx)), Cell_s.cy), (Cell_s.cx + np.sign(df_x)*(abs(S_dx) + abs(N_dx)),Cell_n.cy))
                    ang= np.arctan((Cell_n.cy - Cell_s.cy)/(Cell_n.cx - Cell_s.cx))
                    y  = np.tan(ang)*(S_dx)
                    y  = optimize.newton(_F,y,args=(x,a,Y,u1,v1,u2,v2,s),fprime=_dF)
                    TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,s)
                    CrossPoints = self.fdist((Cell_s.cx + S_dx,Cell_s.cy),(0.0,y),forward=False)
                    CellPoints  = [Cell_n.cx,Cell_n.cy]
                except:
                    TravelTime  = np.inf
                    CrossPoints = [np.nan,np.nan]
                    CellPoints  = [Cell_n.cx,Cell_n.cy] 
            # Latitude
            elif (abs(df_x) < Cell_s.dx/2) and (abs(df_y) > Cell_s.dy/2):
                try:
                    u1 = np.sign(df_y)*Cell_s.vector[1]; v1 = Cell_s.vector[0]
                    u2 = np.sign(df_y)*Cell_n.vector[1]; v2 = Cell_n.vector[0]
                    if np.sign(df_y) == 1:
                        S_dy = Cell_s.dyp; N_dy = -Cell_n.dym
                    else:
                        S_dy = -Cell_s.dym; N_dy = Cell_n.dyp    
                    x  = self.fdist((Cell_s.cy,Cell_s.cx), (Cell_s.cy + S_dy,Cell_s.cx))
                    a  = self.fdist((Cell_n.cy,Cell_n.cx), (Cell_n.cy + N_dy,Cell_n.cx))
                    Y  = self.fdist((Cell_s.cy + np.sign(df_y)*(abs(S_dy) + abs(N_dy)), Cell_s.cx), (Cell_s.cy + np.sign(df_y)*(abs(S_dy) + abs(N_dy)),Cell_n.cx))
                    ang= np.arctan((Cell_n.cx - Cell_s.cx)/(Cell_n.cy - Cell_s.cy))
                    y  = np.tan(ang)*(S_dy)
                    y  = optimize.newton(_F,y,args=(x,a,Y,u1,v1,u2,v2,s),fprime=_dF)
                    TravelTime   = _T(y,x,a,Y,u1,v1,u2,v2,s)
                    CrossPoints  = self.fdist((Cell_s.cx,Cell_s.cy + S_dy),(-y,0.0),forward=False)
                    CellPoints   = [Cell_n.cx,Cell_n.cy]
                except:
                    TravelTime  = np.inf
                    CrossPoints = [np.nan,np.nan]
                    CellPoints  = [Cell_n.cx,Cell_n.cy]       
            else:
                try:
                    u1 = np.sign(df_x)*Cell_s.vector[0]; v1 = Cell_s.vector[1]
                    u2 = np.sign(df_x)*Cell_n.vector[0]; v2 = Cell_n.vector[1]
                    if np.sign(df_x) == 1:
                        S_dx = Cell_s.dxp; N_dx = -Cell_n.dxm
                    else:
                        S_dx = -Cell_s.dxm; N_dx = Cell_n.dxp     
                    if np.sign(df_y) == 1:
                        S_dy = Cell_s.dyp; N_dy = -Cell_n.dym
                    else:
                        S_dy = -Cell_s.dym; N_dy = Cell_n.dyp    
                    x  = self.fdist( (Cell_s.cx,Cell_s.cy), (Cell_s.cx + S_dx,Cell_s.cy))
                    a  = self.fdist( (Cell_n.cx,Cell_n.cy), (Cell_n.cx + N_dx,Cell_n.cy))
                    Y  = self.fdist((Cell_s.cx + np.sign(df_x)*(abs(S_dx) + abs(N_dx)), Cell_s.cy), (Cell_s.cx + np.sign(df_x)*(abs(S_dx) + abs(N_dx)),Cell_n.cy))
                    y  = S_dy
                    u1 = np.sign(df_y)*Cell_s.vector[1]; v1 = Cell_s.vector[0]
                    TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,s)
                    CrossPoints = self.fdist((Cell_s.cx,Cell_s.cy),(0.0,y),forward=False)
                    CellPoints  = [Cell_n.cx,Cell_n.cy]
                except:
                    TravelTime  = np.inf
                    CrossPoints = [np.nan,np.nan]
                    CellPoints  = [Cell_n.cx,Cell_n.cy]  

            return TravelTime, CrossPoints, CellPoints



    def value(self,index):
        '''
        Function for computing the shortest travel-time from a cell to its neighbours by applying the Newtonian method for optimisation
        
        Inputs:
        index - Index of the cell to process
        
        Output:

        Bugs/Alterations:
            - Return the crossing point in Lat/Long position
        '''
        # Determining the nearest neighbour index for the cell
        neighbours = self.Mesh.NearestNeighbours(index)

        # Creating Blank travel-time and crossing point array
        TravelTime  = np.zeros((len(neighbours)))
        CrossPoints = np.zeros((len(neighbours),2))
        CellPoints  = np.zeros((len(neighbours),2))

        # Determining if point is a waypoint
        waypoint_list = self.OptInfo['WayPoints']['Index'].astype(int).to_list()   
        Cell_s = self.Mesh.cells[index]
        if index in waypoint_list:
            Cell_s.cx = self.OptInfo['WayPoints']['Long'].iloc[waypoint_list.index(index)]
            Cell_s.cy = self.OptInfo['WayPoints']['Lat'].iloc[waypoint_list.index(index)]
            Cell_s.dxp = ((Cell_s.x + Cell_s.dx) - Cell_s.cx); Cell_s.dxm = (Cell_s.cx - Cell_s.x)
            Cell_s.dyp = ((Cell_s.y + Cell_s.dy) - Cell_s.cy); Cell_s.dym = (Cell_s.cy - Cell_s.y)
        else:
            Cell_s.dxp = Cell_s.dx/2; Cell_s.dxm = Cell_s.dx/2
            Cell_s.dyp = Cell_s.dy/2; Cell_s.dym = Cell_s.dy/2                      


        # Looping over all the nearest neighbours 
        for lp_index, neighbour_index in enumerate(neighbours):
            # Determining the cell for the neighbour
            Cell_n = self.Mesh.cells[neighbour_index]

            # Set travel-time to infinite if neighbour is land or ice-thickness is too large.
            if (Cell_n.value >= self.Mesh.meshinfo['IceExtent']['MaxProportion']) or (Cell_s.value >= self.Mesh.meshinfo['IceExtent']['MaxProportion']) or (Cell_n.isLand) or (Cell_s.isLand):
                TravelTime[lp_index] = np.inf
                continue

            #Determining if the cell includes waypoint, then taking centroid to waypoint location
            if neighbour_index in waypoint_list:
                Cell_n.cx = self.OptInfo['WayPoints']['Long'].iloc[waypoint_list.index(neighbour_index)]
                Cell_n.cy = self.OptInfo['WayPoints']['Lat'].iloc[waypoint_list.index(neighbour_index)]
                Cell_n.dxp = ((Cell_n.x + Cell_n.dx) - Cell_n.cx); Cell_n.dxm = (Cell_n.cx - Cell_n.x)
                Cell_n.dyp = ((Cell_n.y + Cell_n.dy) - Cell_n.cy); Cell_n.dym = (Cell_n.cy - Cell_n.y)
            else:
                Cell_n.dxp = Cell_n.dx/2; Cell_n.dxm = Cell_n.dx/2
                Cell_n.dyp = Cell_n.dy/2; Cell_n.dym = Cell_n.dy/2            


            TravelTime[lp_index], CrossPoints[lp_index,:], CellPoints[lp_index,:] = self._newtonian(Cell_s,Cell_n)
            CrossPoints[lp_index,0] = np.clip(CrossPoints[lp_index,0],Cell_n.x,(Cell_n.x+Cell_n.dx))
            CrossPoints[lp_index,1] = np.clip(CrossPoints[lp_index,1],Cell_n.y,(Cell_n.y+Cell_n.dy))

        return neighbours, TravelTime, CrossPoints, CellPoints

    def optimize(self,verbrose=False):
        '''
        Determining the shortest path between all waypoints
        '''
        for wpt in self.OptInfo['WayPoints'].iterrows():
            wpt_name  = wpt[1]['Name']
            wpt_index = int(wpt[1]['Index'])
            wpt_long  = wpt[1]['Long']
            wpt_lat   = wpt[1]['Lat']
            if verbrose:
                print('=== Processing Waypoint = {} ==='.format(wpt_name))


            # if (self.Mesh.cells[wpt_index].value >= self.Mesh.meshinfo['IceExtent']['MaxProportion']) or (self.Mesh.cells[wpt_index].isLand):
            #     if verbrose:
            #         print('--- Waypoint on land or in deep ice extent')
            #     continue

            #Initialising a column array of all the indexs
            self.DijkstraInfo[wpt_name] = {}
            self.DijkstraInfo[wpt_name]['CellIndex']       = np.arange(len(self.Mesh.cells))
            self.DijkstraInfo[wpt_name]['Cost']            = np.full((len(self.Mesh.cells)),np.inf)
            self.DijkstraInfo[wpt_name]['PositionLocked']  = np.zeros((len(self.Mesh.cells)),dtype=bool)
            self.DijkstraInfo[wpt_name]['Paths']           = {}
            self.DijkstraInfo[wpt_name]['Paths_CellIndex'] = {}
            self.DijkstraInfo[wpt_name]['Paths_Cost']      = {}
            for djk in range(len(self.Mesh.cells)):
                self.DijkstraInfo[wpt_name]['Paths'][djk]            = []
                self.DijkstraInfo[wpt_name]['Paths_CellIndex'][djk]  = [wpt_index]
                self.DijkstraInfo[wpt_name]['Paths_Cost'][djk]       = [np.inf]
            self.DijkstraInfo[wpt_name]['Cost'][wpt_index]       = 0.0
            self.DijkstraInfo[wpt_name]['Paths_Cost'][wpt_index] = [0.0]

            while (self.DijkstraInfo[wpt_name]['PositionLocked'] == False).any():
                # Determining the argument with the next lowest value and hasn't been visited
                idx = self.DijkstraInfo[wpt_name]['CellIndex'][(self.DijkstraInfo[wpt_name]['PositionLocked']==False)][np.argmin(self.DijkstraInfo[wpt_name]['Cost'][(self.DijkstraInfo[wpt_name]['PositionLocked']==False)])]
                # Finding the cost of the nearest neighbours
                Neighbour_index,TT,CrossPoints,CellPoints = self.value(idx)
                Neighbour_cost     = TT + self.DijkstraInfo[wpt_name]['Cost'][idx]
                # Determining if the visited time is visited    
                for jj_v,jj in enumerate(Neighbour_index):
                    if Neighbour_cost[jj_v] < self.DijkstraInfo[wpt_name]['Cost'][jj]:
                        self.DijkstraInfo[wpt_name]['Cost'][jj]            = Neighbour_cost[jj_v]
                        self.DijkstraInfo[wpt_name]['Paths'][jj]           = self.DijkstraInfo[wpt_name]['Paths'][idx] + [[CellPoints[jj_v,0],CellPoints[jj_v,1]]] + [[CrossPoints[jj_v,0],CrossPoints[jj_v,1]]]
                        self.DijkstraInfo[wpt_name]['Paths_CellIndex'][jj] = self.DijkstraInfo[wpt_name]['Paths_CellIndex'][idx] + [jj]
                        self.DijkstraInfo[wpt_name]['Paths_Cost'][jj]      = self.DijkstraInfo[wpt_name]['Paths_Cost'][idx] + [Neighbour_cost[jj_v]]
                # Defining the graph point as visited
                self.DijkstraInfo[wpt_name]['PositionLocked'][idx] = True


        # Using the Dijkstra information, save the paths
        self.Paths ={}
        self.Paths['from']   = []
        self.Paths['to']     = []
        self.Paths['Path']   = [] 
        self.Paths['Path_CellIndex']   = [] 
        self.Paths['Path_Cost']   = [] 
        self.Paths['Cost']   = [] 
        for wpt_a in self.OptInfo['WayPoints'].iterrows():
            wpt_a_name  = wpt_a[1]['Name']; wpt_a_index = int(wpt_a[1]['Index']); wpt_a_loc = [[wpt_a[1]['Long'],wpt_a[1]['Lat']]]
            for wpt_b in self.OptInfo['WayPoints'].iterrows():
                wpt_b_name  = wpt_b[1]['Name']; wpt_b_index = int(wpt_b[1]['Index']); wpt_b_loc   = [[wpt_b[1]['Long'],wpt_b[1]['Lat']]]
                if not wpt_a_name == wpt_b_name:
                    self.Paths['from'].append(wpt_a_name)
                    self.Paths['to'].append(wpt_b_name)
                    try:
                        self.Paths['Path'].append(wpt_a_loc+self.DijkstraInfo[wpt_a_name]['Paths'][wpt_b_index]+wpt_b_loc)
                        self.Paths['Path_CellIndex'].append(self.DijkstraInfo[wpt_a_name]['Paths_CellIndex'][wpt_b_index])
                        self.Paths['Path_Cost'].append(self.DijkstraInfo[wpt_a_name]['Paths_Cost'][wpt_b_index])
                        self.Paths['Cost'].append([self.DijkstraInfo[wpt_a_name]['Cost'][wpt_b_index]])
                    except:
                        self.Paths['Path'].append(np.nan)
                        self.Paths['Path_CellIndex'].append(np.nan)
                        self.Paths['Path_Cost'].append(np.nan)
                        self.Paths['Cost'].append(np.nan)

    def Smmothing(self):
        '''
            Given a series of pathways smooth without centroid locations using great circle smoothing
        '''

        # Make a copy of the smoothed paths.
        if self._paths_smoothed:
            print('Path smmothing has already been run, do you want to smooth again ?\nIf so re-run this function')
        else:
            self._paths_smoothed = True

        self.SmoothedPaths = self.Paths.copy()
        self.fdist = _Haversine_distance

        # Looping over all the optimised paths
        for path_index in range(len(self.SmoothedPaths['Path_CellIndex'])):
            path_edges     = self.SmoothedPaths['Path'][path_index][1:-1:2]
            path_cellIndex = self.SmoothedPaths['Path_CellIndex'][path_index]
            path_cost      = self.SmoothedPaths['Path_Cost'][path_index]

            # Iterate the smoothing several times. This could be based on variance later
            for idxpt in range(len(path_edges)-2):
                pt_sp  = path_edges[idxpt]
                pt_cp  = path_edges[idxpt+1]
                pt_np  = path_edges[idxpt+2]
                ind_sp = path_cellIndex[idxpt]
                ind_np = path_cellIndex[idxpt+1]
                # Determine Starting & Ending cells
                Cell_s    = copy.copy(self.Mesh.cells[ind_sp])
                Cell_s.cx = pt_sp[0]; Cell_s.cy = pt_sp[1]
                Cell_s.dxp = ((Cell_s.x + Cell_s.dx) - Cell_s.cx); Cell_s.dxm = (Cell_s.cx - Cell_s.x)
                Cell_s.dyp = ((Cell_s.y + Cell_s.dy) - Cell_s.cy); Cell_s.dym = (Cell_s.cy - Cell_s.y)
                Cell_n    = copy.copy(self.Mesh.cells[ind_np])
                Cell_n.cx = pt_np[0]; Cell_n.cy = pt_np[1]
                Cell_n.dxp = ((Cell_n.x + Cell_n.dx) - Cell_n.cx); Cell_n.dxm = (Cell_n.cx - Cell_n.x)
                Cell_n.dyp = ((Cell_n.y + Cell_n.dy) - Cell_n.cy); Cell_n.dym = (Cell_n.cy - Cell_n.y)
                TravelTime, CrossingPoint, delA = self._newtonian(Cell_s,Cell_n)

                CrossingPoint[0] = np.clip(CrossingPoint[0],Cell_n.x,(Cell_n.x+Cell_n.dx))
                CrossingPoint[1] = np.clip(CrossingPoint[1],Cell_n.y,(Cell_n.y+Cell_n.dy))

                path_edges[idxpt+1] = CrossingPoint
                path_cost[idxpt+1]  = path_cost[idxpt] + TravelTime

            self.SmoothedPaths['Path'][path_index] = path_edges
    
