import numpy as np
import copy
import pandas as pd

import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning) 
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from RoutePlanner.Function import NewtonianDistance, NewtonianCurve

class TravelTime:
    def __init__(self,CellGrid,OptInfo,CostFunc=NewtonianDistance):
        # Load in the current cell structure & Optimisation Info
        self.Mesh    = copy.copy(CellGrid)
        self.OptInfo = copy.copy(OptInfo)

        self.CostFunc       = CostFunc

        self.unit_shipspeed = self.OptInfo['VehicleInfo']['Unit']
        self.unit_time      = self.OptInfo['Time Unit']
        self.zero_currents  = self.OptInfo['Zero Currents']
        self.variableSpeed  =self. OptInfo['VariableSpeed']


        if type(self.OptInfo['WayPoints']) != pd.core.frame.DataFrame:
            self.OptInfo['WayPoints'] = pd.read_csv(self.OptInfo['WayPoints'])
        

        # ====== Waypoints ======        
        # Dropping waypoints outside domain
        self.OptInfo['WayPoints'] = self.OptInfo['WayPoints'][(self.OptInfo['WayPoints']['Long'] >= self.Mesh._longMin) &\
                                                              (self.OptInfo['WayPoints']['Long'] <= self.Mesh._longMax) &\
                                                              (self.OptInfo['WayPoints']['Lat'] <= self.Mesh._latMax) &\
                                                              (self.OptInfo['WayPoints']['Lat'] >= self.Mesh._latMin)] 

        # Initialising Waypoints positions and cell index
        self.OptInfo['WayPoints']['Index'] = np.nan
        for idx,wpt in self.OptInfo['WayPoints'].iterrows():
            long = wpt['Long']
            lat  = wpt['Lat']
            for index, cell in enumerate(self.Mesh.cellBoxes):
                if cell.containsPoint(lat,long):
                    break
            self.OptInfo['WayPoints']['Index'].loc[idx] = index

    def speedFunction(self,Cell):
        if self.variableSpeed == True:
            S = self.OptInfo['VehicleInfo']['Speed']*(-5.95*Cell.iceArea()**3 + 7.03*Cell.iceArea()**2 - 3.00*Cell.iceArea() + 0.98)
        else:
            S = self.OptInfo['VehicleInfo']['Speed']
        return S

    def Dijkstra2Path(self,StartWaypoints,EndWaypoints):
        Paths = []

        wpts_s = self.OptInfo['WayPoints'][self.OptInfo['WayPoints']['Name'].isin(StartWaypoints)]
        wpts_e = self.OptInfo['WayPoints'][self.OptInfo['WayPoints']['Name'].isin(EndWaypoints)]

        for idx,wpt_a in wpts_s.iterrows():
            wpt_a_name  = wpt_a['Name']; wpt_a_index = int(wpt_a['Index']); wpt_a_loc   = [[wpt_a['Long'],wpt_a['Lat']]]
            for idy,wpt_b in wpts_e.iterrows():
                wpt_b_name  = wpt_b['Name']; wpt_b_index = int(wpt_b['Index']); wpt_b_loc   = [[wpt_b['Long'],wpt_b['Lat']]]
                if not wpt_a_name == wpt_b_name:
                    # ==== Correcting Path for waypoints off cell
                    Path = {}
                    Path['from']                      = wpt_a_name
                    Path['to']                        = wpt_b_name
                    Path['Time']               = float(self.DijkstraInfo[wpt_a_name]['Info']['Time'][self.DijkstraInfo[wpt_a_name]['Info']['CellIndex']==wpt_b_index])
                    if Path['Time'] == np.inf:
                        continue
                    PathTT     = self.DijkstraInfo[wpt_a_name]['Path']['Cost'][wpt_b_index]
                    
                    # CellS      = self.Mesh.cellBoxes[wpt_a_index]
                    # CostF      = self.CostFunc(self.Mesh,Cell_S=CellS,Cell_S_Speed=self.speedFunction(CellS),unit_shipspeed='km/hr',unit_time=self.unit_time,zerocurrents=self.zero_currents)
                    # Ts         = CostF.WaypointCorrection(tuple(wpt_a_loc[0]),\
                    #                                       self.DijkstraInfo[wpt_a_name]['Path']['Points'][wpt_b_index][1],\
                    #                                       self.speedFunction(CellS))
                    # CellE      = self.Mesh.cellBoxes[wpt_b_index]
                    # CostF      = self.CostFunc(self.Mesh,Cell_S=CellE,Cell_S_Speed=self.speedFunction(CellE),unit_shipspeed='km/hr',unit_time=self.unit_time,zerocurrents=self.zero_currents)
                    # Te         = CostF.WaypointCorrection(tuple(wpt_b_loc[0]),\
                    #                                       self.DijkstraInfo[wpt_a_name]['Path']['Points'][wpt_b_index][-2],\
                    #                                       self.speedFunction(CellE))
                    # PathTT[1]  = PathTT[0] + Ts
                    # PathTT[-1] = PathTT[-2]+Te

                    # ===== Appending Path ===== 
                    Path['Path']               = {}
                    Path['Path']['Points']     = np.array(wpt_a_loc+self.DijkstraInfo[wpt_a_name]['Path']['Points'][wpt_b_index][:-1]+wpt_b_loc)
                    Path['Path']['Time']       = PathTT
                    Paths.append(Path)
        return Paths
        

    def NeighbourCost(self,index):
        '''
        Function for computing the shortest travel-time from a cell to its neighbours by applying the Newtonian method for optimisation
        
        Inputs:
        index - Index of the cell to process
        
        Output:

        Bugs/Alterations:
            - If corner of cell is land in adjacent cell then also return 'inf'
        '''
        # Determining the nearest neighbour index for the cell
        neigh          = self.Mesh.getNeightbours(self.Mesh.cellBoxes[index])
        neighbours     = neigh['Cell']
        neighbours_idx = neigh['idx']


        # Creating Blank travel-time and crossing point array
        TravelTime  = np.zeros((len(neighbours_idx),2))
        CrossPoints = np.zeros((len(neighbours_idx),2))
        CellPoints  = np.zeros((len(neighbours_idx),2))

        # Defining the starting cell
        Cell_s = self.Mesh.cellBoxes[index]

        # Looping over all the nearest neighbours 
        for lp_index, Cell_n in enumerate(neighbours):
            # Set travel-time to infinite if neighbour is land or ice-thickness is too large.
            if (Cell_n.iceArea() >= self.OptInfo['MaxIceExtent']) or (Cell_s.iceArea() >= self.OptInfo['MaxIceExtent']) or (Cell_n.containsLand()) or (Cell_s.containsLand()):
                TravelTime[lp_index] = np.inf
                continue

            Cell_s_speed = self.speedFunction(Cell_s)
            Cell_n_speed = self.speedFunction(Cell_n)

            CostF    = self.CostFunc(self.Mesh,Cell_S=Cell_s,Cell_N=Cell_n,Cell_S_Speed=Cell_s_speed,Cell_N_Speed=Cell_n_speed,unit_shipspeed='km/hr',unit_time=self.unit_time,zerocurrents=self.zero_currents)
                        
            TravelTime[lp_index,:],CrossPoints[lp_index,:],CellPoints[lp_index,:] = CostF.value()

        return neighbours_idx, TravelTime, CrossPoints, CellPoints

    def _dijkstra(self,wpt_name,end_waypoints):


        Wpts = self.OptInfo['WayPoints'][self.OptInfo['WayPoints']['Name'].isin(end_waypoints)]

        # Loop over all the points until all waypoints are visited
        while (self.DijkstraInfo[wpt_name]['Info']['PositionLocked'][self.DijkstraInfo[wpt_name]['Info']['CellIndex'].isin(np.array(Wpts['Index'].astype(int)))] == False).any():

            non_locked = self.DijkstraInfo[wpt_name]['Info'][self.DijkstraInfo[wpt_name]['Info']['PositionLocked']==False]
            idx        = non_locked['CellIndex'].loc[non_locked['Time'].idxmin()]

            # Finding the cost of the nearest neighbours
            Neighbour_index,TT,CrossPoints,CellPoints = self.NeighbourCost(idx)
            Neighbour_cost  = np.sum(TT,axis=1) + self.DijkstraInfo[wpt_name]['Info']['Time'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==idx].iloc[0]

            # Determining if the visited time is visited    
            for jj_v,jj in enumerate(Neighbour_index):
                if Neighbour_cost[jj_v] <= self.DijkstraInfo[wpt_name]['Info']['Time'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==jj].iloc[0]:
                    self.DijkstraInfo[wpt_name]['Info']['Time'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==jj] = Neighbour_cost[jj_v]
                    self.DijkstraInfo[wpt_name]['Path']['Points'][jj]       = self.DijkstraInfo[wpt_name]['Path']['Points'][idx]       + [[CrossPoints[jj_v,0],CrossPoints[jj_v,1]]] + [[CellPoints[jj_v,0],CellPoints[jj_v,1]]]
                    self.DijkstraInfo[wpt_name]['Path']['CellIndex'][jj]     = self.DijkstraInfo[wpt_name]['Path']['CellIndex'][idx]   + [jj]
                    self.DijkstraInfo[wpt_name]['Path']['Cost'][jj]          = self.DijkstraInfo[wpt_name]['Path']['Cost'][idx]        + [TT[jj_v,0]+self.DijkstraInfo[wpt_name]['Info']['Time'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==idx].iloc[0]] + [TT[jj_v,0]+ TT[jj_v,1]+self.DijkstraInfo[wpt_name]['Info']['Time'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==idx].iloc[0]]
            
            # Defining the graph point as visited
            self.DijkstraInfo[wpt_name]['Info']['PositionLocked'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==idx] = True



    def Paths(self,source_waypoints=None,end_waypoints=None,verbrose=False,multiprocessing=False):
        '''
        Determining the shortest path between all waypoints
        '''

        # Subsetting the waypoints
        if type(source_waypoints) == type(None):
            source_waypoints = list(self.OptInfo['WayPoints']['Name'])
        if type(end_waypoints) == type(None):
            end_waypoints = list(self.OptInfo['WayPoints']['Name'])

        # Initialising the Dijkstra Info Dictionary
        self.DijkstraInfo = {}
        for wpt in source_waypoints:
            wpt_name  = wpt
            wpt_index = self.OptInfo['WayPoints'][self.OptInfo['WayPoints']['Name'] == wpt_name]['Index'].iloc[0]
            self.DijkstraInfo[wpt_name] = {}
            self.DijkstraInfo 
            self.DijkstraInfo[wpt_name]['Info'] = pd.DataFrame({'CellIndex': np.arange(len(self.Mesh.cellBoxes)), 'Time':np.full((len(self.Mesh.cellBoxes)),np.inf), 'PositionLocked': np.zeros((len(self.Mesh.cellBoxes)),dtype=bool)})
            self.DijkstraInfo[wpt_name]['Path']            = {}
            self.DijkstraInfo[wpt_name]['Path']['Points']           = {}
            self.DijkstraInfo[wpt_name]['Path']['CellIndex']          = {}
            self.DijkstraInfo[wpt_name]['Path']['Cost']               = {}
            for djk in range(len(self.Mesh.cellBoxes)):
                self.DijkstraInfo[wpt_name]['Path']['Points'][djk]  = []
                self.DijkstraInfo[wpt_name]['Path']['CellIndex'][djk] = [wpt_index]
                self.DijkstraInfo[wpt_name]['Path']['Cost'][djk]      = [np.inf]
            self.DijkstraInfo[wpt_name]['Info']['Time'][self.DijkstraInfo[wpt_name]['Info']['CellIndex'] == wpt_index] = 0.0
            self.DijkstraInfo[wpt_name]['Path']['Cost'][wpt_index] = [0.0]

        if multiprocessing:
            import multiprocessing
            pool_obj = multiprocessing.Pool()
            answer = pool_obj.map(self._dijkstra,source_waypoints)

        else:
            for wpt in source_waypoints:
                if verbrose:
                    print('=== Processing Waypoint = {} ==='.format(wpt))
                self._dijkstra(wpt,end_waypoints)

        return self.Dijkstra2Path(source_waypoints,end_waypoints)

         

    def PathSmoothing(self,Paths,maxiter=1000,minimumDiff=1e-4,debugging=0):
        '''
            Given a series of pathways smooth without centroid locations using great circle smoothing
        '''
        import copy

        SmoothedPaths = []
        Pths = copy.deepcopy(Paths)

        # Looping over all the optimised paths
        for indx_Path in range(len(Pths)):
            
            Path = Pths[indx_Path]
            if Path['Time'] == np.inf:
                continue

            startPoint = Path['Path']['Points'][0,:][None,:]
            endPoint   = Path['Path']['Points'][-1,:][None,:]

            print('==================================================')
            print(' PATH: {} -> {} '.format(Path['from'],Path['to']))

            Points = np.concatenate([startPoint,
                                            Path['Path']['Points'][1:-1:2],
                                            endPoint])

            iter = 0
            while iter <= maxiter:
                id = 0

                while id <= (len(Points) - 3):
                    Sp  = tuple(Points[id,:])
                    Cp  = tuple(Points[id+1,:])
                    Np  = tuple(Points[id+2,:])

                    if (np.sqrt((Sp[0]-Cp[0])**2 + (Sp[1]-Cp[1])**2) < 1e-4)  or (np.sqrt((Np[0]-Cp[0])**2 + (Np[1]-Cp[1])**2) < 1e-4):
                        Points = np.delete(Points,id+1,axis=0)
                        continue

                    if abs(Sp[0]-Cp[0]) < 1e-4 or abs(Sp[1]-Cp[1]) < 1e-4  or abs(Np[0]-Cp[0]) < 1e-4 + abs(Np[1]-Cp[1] < 1e-4):
                        Points = np.delete(Points,id+1,axis=0)
                        continue


                    nc = NewtonianCurve(self.Mesh,Sp,Cp,Np,self.OptInfo['VehicleInfo']['Speed'],zerocurrents=self.zero_currents,debugging=debugging)
                    CrossingPoint, Boxes = nc.value()

                    Allowed = True
                    for box in Boxes:
                        if box.containsLand() or box.iceArea() >= self.OptInfo['MaxIceExtent']:
                            Allowed = False
                    if not Allowed:
                        id+=1
                        continue

                    if (np.isnan(CrossingPoint).any()):
                        Points = np.delete(Points,id+1,axis=0)
                    else:
                        Points[id+1,:] = CrossingPoint[0,:]
                        if CrossingPoint.shape[0] > 1:
                            Points = np.insert(Points,id+2,CrossingPoint[1:,:],0)
                    id+=1

                if iter!=0:
                    if Points.shape == oldPoints.shape:
                        if np.max(np.sqrt((Points-oldPoints)**2)) < minimumDiff:
                            break
                
                if iter == maxiter:
                    print('Maximum number of iterations met !')
                
                oldPoints = copy.deepcopy(Points)

                iter+=1

            print('{} iterations to convergence'.format(iter-1))

            Path['Path']['Points']       = Points
            SmoothedPaths.append(Path)
        return SmoothedPaths
