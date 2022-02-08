import numpy as np
import copy
import pandas as pd
import matplotlib.pylab as plt


import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning) 
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from RoutePlanner.Function import NewtonianDistance, NewtonianCurve

class TravelTime:
    def __init__(self,CellGrid,CostFunc=NewtonianDistance):
        # Load in the current cell structure & Optimisation Info
        self.Mesh    = copy.copy(CellGrid)
        self.OptInfo = copy.copy(CellGrid.OptInfo)

        self.CostFunc       = CostFunc

        self.unit_shipspeed = self.OptInfo['VehicleInfo']['Unit']  = 'km/hr'
        self.unit_time      = self.OptInfo['Time Unit']
        self.zero_currents  = self.OptInfo['Zero Currents']
        self.variableSpeed  =self. OptInfo['VariableSpeed']

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
            self.DijkstraInfo[wpt_name]['Info'] = pd.DataFrame({'CellIndex': np.arange(len(self.Mesh.cellBoxes)), 'TotalCost':np.full((len(self.Mesh.cellBoxes)),np.inf), 'PositionLocked': np.zeros((len(self.Mesh.cellBoxes)),dtype=bool)})
            self.DijkstraInfo[wpt_name]['Path']            = {}
            self.DijkstraInfo[wpt_name]['Path']['FullPath']         = {}
            self.DijkstraInfo[wpt_name]['Path']['CrossingPoints']   = {}
            self.DijkstraInfo[wpt_name]['Path']['CentroidPoints']   = {}
            self.DijkstraInfo[wpt_name]['Path']['CellIndex']        = {}
            self.DijkstraInfo[wpt_name]['Path']['Cost']             = {}
            for djk in range(len(self.Mesh.cellBoxes)):
                self.DijkstraInfo[wpt_name]['Path']['FullPath'][djk] = []
                self.DijkstraInfo[wpt_name]['Path']['CrossingPoints'][djk] = []
                self.DijkstraInfo[wpt_name]['Path']['CentroidPoints'][djk] = []
                self.DijkstraInfo[wpt_name]['Path']['CellIndex'][djk]      = [wpt_index]
                self.DijkstraInfo[wpt_name]['Path']['Cost'][djk]           = [np.inf]
            self.DijkstraInfo[wpt_name]['Info']['TotalCost'][self.DijkstraInfo[wpt_name]['Info']['CellIndex'] == wpt_index] = 0.0
            self.DijkstraInfo[wpt_name]['Path']['Cost'][wpt_index] = [0.0]

        # ====== Path Information ======
        # Initialising Path & Path Smoothed defintion
        self.Paths         = {}
        self.SmoothedPaths = {}
        self._paths_smoothed = False


    def speedFunction(self,Cell):
        if self.variableSpeed == True:
            S = self.OptInfo['VehicleInfo']['Speed']*(1-np.sqrt(Cell.iceArea()))
        else:
            S = self.OptInfo['VehicleInfo']['Speed']
        return S

    def Dijkstra2Path(self):
        self.Paths = []
        for wpt_a in self.OptInfo['WayPoints'].iterrows():
            wpt_a_name  = wpt_a[1]['Name']; wpt_a_index = int(wpt_a[1]['Index']); wpt_a_loc   = [[wpt_a[1]['Long'],wpt_a[1]['Lat']]]
            for wpt_b in self.OptInfo['WayPoints'].iterrows():
                wpt_b_name  = wpt_b[1]['Name']; wpt_b_index = int(wpt_b[1]['Index']); wpt_b_loc   = [[wpt_b[1]['Long'],wpt_b[1]['Lat']]]
                if not wpt_a_name == wpt_b_name:
                    # ==== Correcting Path for waypoints off cell

                    # WpS       = tuple(wpt_a_loc[0])
                    # CellS     = self.Mesh.cellBoxes[wpt_a_index]
                    # SpeedS    = self.speedFunction(CellS)
                    # CrossingS = self.DijkstraInfo[wpt_a_name]['Path']['CrossingPoints'][wpt_b_index][0]
                    # CostF     = self.CostFunc(Cell_S=CellS,Cell_S_Speed=SpeedS,unit_shipspeed='km/hr',unit_time=self.unit_time,zerocurrents=self.zero_currents)
                    # T         = CostF.WaypointCorrection(WpS,CrossingS,SpeedS)

                    # ===== Appending Path =====
                    Path = {}
                    Path['from']                   = wpt_a_name
                    Path['to']                     = wpt_b_name
                    Path['TotalCost']              = float(self.DijkstraInfo[wpt_a_name]['Info']['TotalCost'][self.DijkstraInfo[wpt_a_name]['Info']['CellIndex']==wpt_b_index])
                    Path['Path']                   = {}
                    #Path['Path']['FullPath']       = np.array(wpt_a_loc+self.DijkstraInfo[wpt_a_name]['Path']['FullPath'][wpt_b_index]+wpt_b_loc)
                    Path['Path']['FullPath']       = np.array(wpt_a_loc+self.DijkstraInfo[wpt_a_name]['Path']['FullPath'][wpt_b_index][:-1]+wpt_b_loc)
                    Path['Path']['CrossingPoints'] = np.array(self.DijkstraInfo[wpt_a_name]['Path']['CrossingPoints'][wpt_b_index])
                    Path['Path']['Cost']           = np.array(self.DijkstraInfo[wpt_a_name]['Path']['Cost'][wpt_b_index])
                    self.Paths.append(Path)

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
        neighbours,neighbours_idx = self.Mesh.getNeightbours(self.Mesh.cellBoxes[index])

        # Creating Blank travel-time and crossing point array
        TravelTime  = np.zeros((len(neighbours_idx),2))
        CrossPoints = np.zeros((len(neighbours_idx),2))
        CellPoints  = np.zeros((len(neighbours_idx),2))

        # Defining the starting cell
        Cell_s = self.Mesh.cellBoxes[index]

        # Looping over all the nearest neighbours 
        for lp_index, Cell_n in enumerate(neighbours):
            # Set travel-time to infinite if neighbour is land or ice-thickness is too large.
            if (Cell_n.iceArea() >= self.OptInfo['MaxIceExtent']) or (Cell_s.iceArea() >= self.OptInfo['MaxIceExtent']) or (Cell_n.isLand()) or (Cell_s.isLand()):
                TravelTime[lp_index] = np.inf
                continue

            Cell_s_speed = self.speedFunction(Cell_s)
            Cell_n_speed = self.speedFunction(Cell_n)

            CostF    = self.CostFunc(Cell_S=Cell_s,Cell_N=Cell_n,Cell_S_Speed=Cell_s_speed,Cell_N_Speed=Cell_n_speed,unit_shipspeed='km/hr',unit_time=self.unit_time,zerocurrents=self.zero_currents)
                        
            TravelTime[lp_index,:],CrossPoints[lp_index,:],CellPoints[lp_index,:] = CostF.value()

        return neighbours_idx, TravelTime, CrossPoints, CellPoints

    def _dijkstra(self,wpt_name):
        # Loop over all the points until all waypoints are visited
        while (self.DijkstraInfo[wpt_name]['Info']['PositionLocked'][self.DijkstraInfo[wpt_name]['Info']['CellIndex'].isin(np.array(self.OptInfo['WayPoints']['Index'].astype(int)))] == False).any():


            non_locked = self.DijkstraInfo[wpt_name]['Info'][self.DijkstraInfo[wpt_name]['Info']['PositionLocked']==False]
            idx        = non_locked['CellIndex'].loc[non_locked['TotalCost'].idxmin()]

            # Finding the cost of the nearest neighbours
            Neighbour_index,TT,CrossPoints,CellPoints = self.NeighbourCost(idx)
            Neighbour_cost  = np.sum(TT,axis=1) + self.DijkstraInfo[wpt_name]['Info']['TotalCost'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==idx].iloc[0]

            # Determining if the visited time is visited    
            for jj_v,jj in enumerate(Neighbour_index):
                if Neighbour_cost[jj_v] <= self.DijkstraInfo[wpt_name]['Info']['TotalCost'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==jj].iloc[0]:
                    self.DijkstraInfo[wpt_name]['Info']['TotalCost'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==jj] = Neighbour_cost[jj_v]
                    self.DijkstraInfo[wpt_name]['Path']['FullPath'][jj]       = self.DijkstraInfo[wpt_name]['Path']['FullPath'][idx]       + [[CrossPoints[jj_v,0],CrossPoints[jj_v,1]]] + [[CellPoints[jj_v,0],CellPoints[jj_v,1]]]
                    self.DijkstraInfo[wpt_name]['Path']['CrossingPoints'][jj] = self.DijkstraInfo[wpt_name]['Path']['CrossingPoints'][idx] + [[CrossPoints[jj_v,0],CrossPoints[jj_v,1]]]
                    self.DijkstraInfo[wpt_name]['Path']['CentroidPoints'][jj] = self.DijkstraInfo[wpt_name]['Path']['CentroidPoints'][idx] + [[CellPoints[jj_v,0],CellPoints[jj_v,1]]]
                    self.DijkstraInfo[wpt_name]['Path']['CellIndex'][jj]      = self.DijkstraInfo[wpt_name]['Path']['CellIndex'][idx]      + [jj]
                    self.DijkstraInfo[wpt_name]['Path']['Cost'][jj]           = self.DijkstraInfo[wpt_name]['Path']['Cost'][idx]           + list(TT[jj_v]+ self.DijkstraInfo[wpt_name]['Info']['TotalCost'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==idx].iloc[0])
            
            # Defining the graph point as visited
            self.DijkstraInfo[wpt_name]['Info']['PositionLocked'][self.DijkstraInfo[wpt_name]['Info']['CellIndex']==idx] = True



    def Dijkstra(self,verbrose=False,multiprocessing=False):
        '''
        Determining the shortest path between all waypoints
        '''

        # Constructing the cell paths information
        if type(self.OptInfo['Start Waypoints']) == type(None):
            waypointList = list(self.OptInfo['WayPoints']['Name'])
        else:
            waypointList = self.OptInfo['Start Waypoints']



        if multiprocessing:
            import multiprocessing
            pool_obj = multiprocessing.Pool()
            answer = pool_obj.map(self._dijkstra,waypointList)

        else:
            for wpt in waypointList:
                if verbrose:
                    print('=== Processing Waypoint = {} ==='.format(wpt))
                self._dijkstra(wpt)

        # Chaning Dijkstra Information to Paths
        self.Dijkstra2Path()

    def PlotPaths(self,routepoints=False,return_ax=True,currents=False):

        ax = self.Mesh.plot(currents=currents,return_ax=True)

        # Constructing the cell paths information
        if type(self.OptInfo['Start Waypoints']) == type(None):
            waypointList = list(self.OptInfo['WayPoints']['Name'])
        else:
            waypointList = self.OptInfo['Start Waypoints']


        for Path in self.Paths:
            if (Path['from'] in waypointList):
                if Path['TotalCost'] == np.inf:
                    continue
                Points = np.array(Path['Path']['FullPath'])
                if routepoints:
                    ax.plot(Points[:,0],Points[:,1],linewidth=1.0,color='k')
                    ax.scatter(Points[:,0],Points[:,1],30,zorder=99,color='k')
                    #quad = ax.scatter(Points[:,0],Points[:,1],30,Path['Path']['Cost'],zorder=99,cmap='hsv')
                else:
                    ax.plot(Points[:,0],Points[:,1],linewidth=1.0,color='k')

        # Plotting Waypoints
        ax.scatter(self.OptInfo['WayPoints']['Long'],self.OptInfo['WayPoints']['Lat'],100,marker='^',color='k',zorder=100)
        for wpt in self.OptInfo['WayPoints'].iterrows():
            Long = wpt[1]['Long']
            Lat  = wpt[1]['Lat']
            Name = wpt[1]['Name']
            ax.text(Long,Lat,Name,color='k',zorder=100)

        # if routepoints:
        #     plt.colorbar(quad,ax=ax,label='Travel Time ({})'.format(self.unit_time))

        if return_ax:
            return ax

                    
    def PathSmoothing(self,maxiter=50):
        '''
            Given a series of pathways smooth without centroid locations using great circle smoothing


            Bugs:
                - Currently we loop over the whole path. I need to take the previous point in the loop, otherwise finding global minimum more difficult.

        '''
        self.SmoothedPaths = self.Paths.copy()

        # Looping over all the optimised paths
        for Path in self.SmoothedPaths:

            startPoint = Path['Path']['FullPath'][0,:][None,:]
            endPoint   = Path['Path']['FullPath'][-1,:][None,:]

            print('==================================================')
            print(' PATH: {} -> {} '.format(Path['from'],Path['to']))

            OrgcrossingPoints = np.concatenate([startPoint,
                                            Path['Path']['CrossingPoints'],
                                            endPoint])

            Points = OrgcrossingPoints.copy()


            for iter in range(maxiter):
                for id in range(Points.shape[0]-2):
                    Sp  = tuple(Points[id,:])
                    Cp  = tuple(Points[id+1,:])
                    Np  = tuple(Points[id+2,:])
                    nc = NewtonianCurve(self.Mesh,Sp,Cp,Np,self.OptInfo['VehicleInfo']['Speed'])
                    TravelTime, CrossingPoint = nc.value()

                    if (np.isnan(CrossingPoint)).any():
                        CrossingPoint = np.array([[Cp[0],Cp[1]]])

                    if id == 0:
                        newPoints = CrossingPoint
                    else:
                        newPoints = np.concatenate([newPoints,CrossingPoint])

                Points = np.concatenate([startPoint,newPoints,endPoint])


            # iter = 0
            # while iter <= maxiter:
            #     id = 0
            #     while id <= (len(Points) - 3):
            #         Sp  = tuple(Points[id,:])
            #         Cp  = tuple(Points[id+1,:])
            #         Np  = tuple(Points[id+2,:])
            #         nc = NewtonianCurve(self.Mesh,Sp,Cp,Np,self.OptInfo['VehicleInfo']['Speed'])
            #         TravelTime, CrossingPoint = nc.value()    

            #         if (np.isnan(CrossingPoint)).any():
            #             id+=1
            #             continue
            #         else:
            #             Points[id+1,:] = CrossingPoint[0,:]
            #             if CrossingPoint.shape[0] > 1:
            #                 Points = np.insert(Points,id+2,CrossingPoint[1:,:],0)
            #         id+=1
            #     iter+=1

            Path['Path']['FullPath']       = Points
            Path['Path']['CrossingPoints'] = Points