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

        # ====== Dijkstra Formulation ======
        # Initialising the Dijkstra Info Dictionary
        self.DijkstraInfo = {}
        for wpt in self.OptInfo['WayPoints'].iterrows():
            wpt_name  = wpt[1]['Name']
            wpt_index = int(wpt[1]['Index'])
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

    def speedFunction(self,Cell):
        if self.variableSpeed == True:
            S = self.OptInfo['VehicleInfo']['Speed']*(1-np.sqrt(Cell.iceArea()))
        else:
            S = self.OptInfo['VehicleInfo']['Speed']
        return S

    def Dijkstra2Path(self):
        Paths = []
        for wpt_a in self.OptInfo['WayPoints'].iterrows():
            wpt_a_name  = wpt_a[1]['Name']; wpt_a_index = int(wpt_a[1]['Index']); wpt_a_loc   = [[wpt_a[1]['Long'],wpt_a[1]['Lat']]]
            for wpt_b in self.OptInfo['WayPoints'].iterrows():
                wpt_b_name  = wpt_b[1]['Name']; wpt_b_index = int(wpt_b[1]['Index']); wpt_b_loc   = [[wpt_b[1]['Long'],wpt_b[1]['Lat']]]
                if not wpt_a_name == wpt_b_name:
                    # ==== Correcting Path for waypoints off cell
                    Path = {}
                    Path['from']                      = wpt_a_name
                    Path['to']                        = wpt_b_name
                    Path['Time']               = float(self.DijkstraInfo[wpt_a_name]['Info']['Time'][self.DijkstraInfo[wpt_a_name]['Info']['CellIndex']==wpt_b_index])
                    if Path['Time'] == np.inf:
                        continue
                    
                    PathTT    = self.DijkstraInfo[wpt_a_name]['Path']['Cost'][wpt_b_index]

                    # CellS     = self.Mesh.cellBoxes[wpt_a_index]
                    # CostF     = self.CostFunc(self.Mesh,Cell_S=CellS,Cell_S_Speed=self.speedFunction(CellS),unit_shipspeed='km/hr',unit_time=self.unit_time,zerocurrents=self.zero_currents)
                    # Ts        = CostF.WaypointCorrection(tuple(wpt_a_loc[0]),\
                    #                                       self.DijkstraInfo[wpt_a_name]['Path']['Points'][wpt_b_index][1],\
                    #                                       self.speedFunction(CellS))
                    # CellE     = self.Mesh.cellBoxes[wpt_b_index]
                    # CostF     = self.CostFunc(self.Mesh,Cell_S=CellE,Cell_S_Speed=self.speedFunction(CellE),unit_shipspeed='km/hr',unit_time=self.unit_time,zerocurrents=self.zero_currents)
                    # Te        = CostF.WaypointCorrection(tuple(wpt_b_loc[0]),\
                    #                                       self.DijkstraInfo[wpt_a_name]['Path']['Points'][wpt_b_index][-2],\
                    #                                       self.speedFunction(CellE))
                    # print('Start = Original:{},New:{}'.format(PathTT[1],Ts))
                    # print('End   = Original:{},New:{}'.format(PathTT[-1]-PathTT[-2],Te))
                    # PathTT[1] = PathTT[0] + Ts
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

    def _dijkstra(self,wpt_name):
        # Loop over all the points until all waypoints are visited
        while (self.DijkstraInfo[wpt_name]['Info']['PositionLocked'][self.DijkstraInfo[wpt_name]['Info']['CellIndex'].isin(np.array(self.OptInfo['WayPoints']['Index'].astype(int)))] == False).any():


            non_locked = self.DijkstraInfo[wpt_name]['Info'][self.DijkstraInfo[wpt_name]['Info']['PositionLocked']==False]
            idx        = non_locked['CellIndex'].loc[non_locked['Time'].idxmin()]

            # Finding the cost of the nearest neighbours
            Neighbour_index,TT,CrossPoints,CellPoints = self.NeighbourCost(idx)
            Neighbour_cost  = np.sum(TT,axis=1) + self.DijkstraInfo[wpt_name]['Info']['Time'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==idx].iloc[0]

            print(TT.shape)

            # Determining if the visited time is visited    
            for jj_v,jj in enumerate(Neighbour_index):
                if Neighbour_cost[jj_v] <= self.DijkstraInfo[wpt_name]['Info']['Time'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==jj].iloc[0]:
                    self.DijkstraInfo[wpt_name]['Info']['Time'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==jj] = Neighbour_cost[jj_v]
                    self.DijkstraInfo[wpt_name]['Path']['Points'][jj]       = self.DijkstraInfo[wpt_name]['Path']['Points'][idx]       + [[CrossPoints[jj_v,0],CrossPoints[jj_v,1]]] + [[CellPoints[jj_v,0],CellPoints[jj_v,1]]]
                    self.DijkstraInfo[wpt_name]['Path']['CellIndex'][jj]     = self.DijkstraInfo[wpt_name]['Path']['CellIndex'][idx]   + [jj]
                    self.DijkstraInfo[wpt_name]['Path']['Cost'][jj]          = self.DijkstraInfo[wpt_name]['Path']['Cost'][idx]        + list(TT[jj_v,:]+ self.DijkstraInfo[wpt_name]['Info']['Time'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==idx].iloc[0])
            
            # Defining the graph point as visited
            self.DijkstraInfo[wpt_name]['Info']['PositionLocked'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==idx] = True



    def Paths(self,verbrose=False,multiprocessing=False):
        '''
        Determining the shortest path between all waypoints
        '''

        source_waypoints = list(self.OptInfo['WayPoints']['Name'])
        end_waypoints    = list(self.OptInfo['WayPoints']['Name'])

        if multiprocessing:
            import multiprocessing
            pool_obj = multiprocessing.Pool()
            answer = pool_obj.map(self._dijkstra,source_waypoints)

        else:
            for wpt in source_waypoints:
                if verbrose:
                    print('=== Processing Waypoint = {} ==='.format(wpt))
                self._dijkstra(wpt)

        return self.Dijkstra2Path()

         

                    
    def PathSmoothing(self,maxiter=500):
        '''
            Given a series of pathways smooth without centroid locations using great circle smoothing


            Bugs:
                - Currently we loop over the whole path. I need to take the previous point in the loop, otherwise finding global minimum more difficult.

        '''
        import copy
        self.SmoothedPaths = []
        Pths = copy.deepcopy(self.Paths)

        # Looping over all the optimised paths
        for indx_Path in range(len(Pths)):
            
            Path = Pths[indx_Path]
            if Path['Time'] == np.inf:
                continue

            startPoint = Path['Path']['Points'][0,:][None,:]
            endPoint   = Path['Path']['Points'][-1,:][None,:]

            if len(Path['Path']['CrossingPoints']) == 0:
                continue

            print('==================================================')
            print(' PATH: {} -> {} '.format(Path['from'],Path['to']))

            OrgcrossingPoints = np.concatenate([startPoint,
                                            Path['Path']['CrossingPoints'],
                                            endPoint])

            Points = OrgcrossingPoints.copy()

            iter = 0
            while iter <= maxiter:
                id = 0
                while id <= (len(Points) - 3):
                    Sp  = tuple(Points[id,:])
                    Cp  = tuple(Points[id+1,:])
                    Np  = tuple(Points[id+2,:])

                    # # Remove crossing point if are the same location
                    # if (((np.array(Sp)-np.array(Cp))**2).sum() < 1e-4) or\
                    # (((np.array(Cp)-np.array(Np))**2).sum() < 1e-4):
                    #     Points = np.delete(Points,id+1,axis=0)
                    #     continue


                    nc = NewtonianCurve(self.Mesh,Sp,Cp,Np,self.OptInfo['VehicleInfo']['Speed'])
                    
                    #try:
                    TravelTime, CrossingPoint, Box1, Box2 = nc.value()
                    # # Removing Points
                    if (Box1 == Box2) or (type(Box1)==str) or (type(Box2)==str):
                        id+=1
                        continue
                    if (Box1.containsLand()) or (Box2.containsLand()):
                        id+=1
                        continue   
                    else:
                        Points[id+1,:] = CrossingPoint[0,:]
                        if CrossingPoint.shape[0] > 1:
                            Points = np.insert(Points,id+2,CrossingPoint[1:,:],0)
                    id+=1
                    #except:
                    #    id+=1
                iter+=1

            Path['Path']['Points']       = Points
            Path['Path']['CrossingPoints'] = Points


            self.SmoothedPaths.append(Path)