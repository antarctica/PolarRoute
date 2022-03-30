import numpy as np
import copy
import pandas as pd

import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning) 
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from RoutePlanner.Function import NewtonianDistance, NewtonianCurve
from RoutePlanner.CellBox import CellBox

import numpy as np
import copy
import pandas as pd


from RoutePlanner.Function import NewtonianDistance, NewtonianCurve
from RoutePlanner.CellBox import CellBox

class TravelTime:
    def __init__(self,CellGrid,OptInfo,CostFunc=NewtonianDistance):
        # Load in the current cell structure & Optimisation Info
        self.Mesh    = copy.copy(CellGrid)
        self.OptInfo = copy.copy(OptInfo)

        # Constructing Neighbour Graph
        neighbourGraph = {}
        for idx,cell in enumerate(self.Mesh.cellBoxes):
            if not isinstance(cell, CellBox):
                continue
            else:
                neigh     = self.Mesh.neighbourGraph[idx]
                cases     = []
                neighIndx = []
                for case in neigh.keys():
                    indxs = neigh[case]
                    if len(indxs) == 0:
                        continue
                    for indx in indxs:
                        cases.append(case)
                        neighIndx.append(indxs[0])
                neighDict = {}
                neighDict['cX']    = cell.cx
                neighDict['cY']    = cell.cy
                neighDict['case']  = cases
                neighDict['neighbourIndex'] = neighIndx 
            neighbourGraph[idx] = neighDict
        self.neighbourGraph = pd.DataFrame().from_dict(neighbourGraph,orient='index')
        self.neighbourGraph['positionLocked'] = False
        self.neighbourGraph['traveltime']     = np.inf
        self.neighbourGraph['neighbourTravelLegs'] = [list() for x in range(len(self.neighbourGraph.index))]
        self.neighbourGraph['neighbourCrossingPoints'] = [list() for x in range(len(self.neighbourGraph.index))]
        self.neighbourGraph['pathIndex']  = [list() for x in range(len(self.neighbourGraph.index))]
        self.neighbourGraph['pathCost']   = [list() for x in range(len(self.neighbourGraph.index))]
        self.neighbourGraph['pathPoints']   = [list() for x in range(len(self.neighbourGraph.index))]

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
        self.OptInfo['WayPoints']['index'] = np.nan
        for idx,wpt in self.OptInfo['WayPoints'].iterrows():
            long = wpt['Long']
            lat  = wpt['Lat']
            for index, cell in enumerate(self.Mesh.cellBoxes):
                if isinstance(cell, CellBox):
                    if cell.containsPoint(lat,long):
                        break
            self.OptInfo['WayPoints']['index'].loc[idx] = index

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
            wpt_a_name  = wpt_a['Name']; wpt_a_index = int(wpt_a['index']); wpt_a_loc   = [[wpt_a['Long'],wpt_a['Lat']]]
            for idy,wpt_b in wpts_e.iterrows():
                wpt_b_name  = wpt_b['Name']; wpt_b_index = int(wpt_b['index']); wpt_b_loc   = [[wpt_b['Long'],wpt_b['Lat']]]
                if not wpt_a_name == wpt_b_name:
                    # ==== Correcting Path for waypoints off cell
                    Path = {}
                    Path['from']               = wpt_a_name
                    Path['to']                 = wpt_b_name


                    Graph = self.DijkstraInfo[wpt_a_name]
                    Path['Time']               = float(Graph['traveltime'].loc[wpt_b_index])
                    if Path['Time'] == np.inf:
                        continue
                    PathTT     = Graph['pathCost'].loc[wpt_b_index]

                    # ===== Appending Path ===== 
                    Path['Path']               = {}
                    Path['Path']['Points']     = np.array(wpt_a_loc+Graph['pathPoints'].loc[wpt_b_index]+wpt_b_loc)
                    Path['Path']['Time']       = PathTT
                    Paths.append(Path)
        return Paths    

    def NeighbourCost(self,wpt_name,minimumTravelTimeIndex):
        '''
        Function for computing the shortest travel-time from a cell to its neighbours by applying the Newtonian method for optimisation
        
        Inputs:
        index - Index of the cell to process
        
        Output:

        Bugs/Alterations:
            - If corner of cell is land in adjacent cell then also return 'inf'
        '''
        # Determining the nearest neighbour index for the cellƒ
        Sc = self.Mesh.cellBoxes[minimumTravelTimeIndex]
        SourceGraph = self.DijkstraInfo[wpt_name].loc[minimumTravelTimeIndex]

        # Looping over idx
        for idx in range(len(SourceGraph['case'])):
            Nc   = self.Mesh.cellBoxes[SourceGraph['neighbourIndex'][idx]]
            Case = SourceGraph['case'][idx]

            # Set travel-time to infinite if neighbour is land or ice-thickness is too large.
            if (Nc.iceArea() >= self.OptInfo['MaxIceExtent']) or (Sc.iceArea() >= self.OptInfo['MaxIceExtent']) or (Nc.containsLand()) or (Sc.containsLand()):
                SourceGraph['neighbourTravelLegs'].append([np.inf,np.inf])
                continue

            Sc_speed = self.speedFunction(Sc)
            Nc_speed = self.speedFunction(Nc)

            # Applying Newton curve to determine crossing point
            CostF    = self.CostFunc(self.Mesh,Sc=Sc,Nc=Nc,Sc_Speed=Sc_speed,Nc_Speed=Nc_speed,Case=Case,unit_shipspeed='km/hr',unit_time=self.unit_time,zerocurrents=self.zero_currents)
            # Updating the Dijkstra graph with the new information
            TravelTime,CrossPoints,CellPoints = CostF.value()
            

            SourceGraph['neighbourTravelLegs'].append(TravelTime)
            SourceGraph['neighbourCrossingPoints'].append(CrossPoints)

            # Updating the neighbour traveltime if its less the current global optimum
            Neighbour_cost  = SourceGraph['traveltime'] + np.sum(TravelTime)
            NeighbourGraph  = self.DijkstraInfo[wpt_name].loc[SourceGraph['neighbourIndex'][idx]]

            if Neighbour_cost < NeighbourGraph['traveltime']:
                NeighbourGraph['traveltime'] = Neighbour_cost
                NeighbourGraph['pathIndex']  = SourceGraph['pathIndex']  + [idx]
                NeighbourGraph['pathCost']   = SourceGraph['pathCost']   + [Neighbour_cost]
                NeighbourGraph['pathPoints'] = SourceGraph['pathPoints'] + [list(CrossPoints)] + [list(CellPoints)]

            self.DijkstraInfo[wpt_name].loc[SourceGraph['neighbourIndex'][idx]] = NeighbourGraph

        self.DijkstraInfo[wpt_name].loc[minimumTravelTimeIndex] = SourceGraph


    def _dijkstra(self,wpt_name):
        # Including only the End Waypoints defined by the user
        Wpts = self.OptInfo['WayPoints'][self.OptInfo['WayPoints']['Name'].isin(self.end_waypoints)]
        
        # Initalising zero traveltime at the source location
        SourceIndex = int(Wpts[Wpts['Name'] == 'MargueriteBay']['index'])
        self.DijkstraInfo[wpt_name].loc[SourceIndex,'traveltime'] = 0.0
        
        # Updating Dijkstra as long as all the waypoints are not visited.
        while (self.DijkstraInfo[wpt_name].loc[Wpts['index'],'positionLocked'] == False).any():

            # Determining the index of the minimum traveltime that has not been visited
            minimumTravelTimeIndex = self.DijkstraInfo[wpt_name][self.DijkstraInfo[wpt_name]['positionLocked']==False]['traveltime'].idxmin()

            # Finding the cost of the nearest neighbours from the source cell (Sc)
            self.NeighbourCost(wpt_name,minimumTravelTimeIndex)

            # Updating Position to be locked
            self.DijkstraInfo[wpt_name].loc[minimumTravelTimeIndex,'positionLocked'] = True

    def Paths(self,source_waypoints=None,end_waypoints=None,verbrose=False,multiprocessing=False):
        '''
        Determining the shortest path between all waypoints
        '''

        # Subsetting the waypoints
        if type(source_waypoints) == type(None):
            source_waypoints = list(self.OptInfo['WayPoints']['Name'])
        if type(end_waypoints) == type(None):
            self.end_waypoints = list(self.OptInfo['WayPoints']['Name'])

        # Initialising the Dijkstra Info Dictionary
        self.DijkstraInfo = {}
        for wpt in source_waypoints:
            self.DijkstraInfo[wpt] = copy.copy(self.neighbourGraph)


        if multiprocessing:
            import multiprocessing as mp
            pool = mp.Pool(mp.cpu_count())
            [pool.apply(self._dijkstra, args=source) for source in source_waypoints]
            pool.close()   


            pool_obj = multiprocessing.Pool()
            answer = pool_obj.map(self._dijkstra,source_waypoints)
        else:
            for wpt in source_waypoints:
                if verbrose:
                    print('=== Processing Waypoint = {} ==='.format(wpt))
                self._dijkstra(wpt)

        return self.Dijkstra2Path(source_waypoints,self.end_waypoints)

         

    # def PathSmoothing(self,Paths,maxiter=1000,minimumDiff=1e-4,debugging=0):
    #     '''
    #         Given a series of pathways smooth without centroid locations using great circle smoothing
    #     '''
    #     import copy

    #     SmoothedPaths = []
    #     Pths = copy.deepcopy(Paths)

    #     # Looping over all the optimised paths
    #     for indx_Path in range(len(Pths)):
            
    #         Path = Pths[indx_Path]
    #         if Path['Time'] == np.inf:
    #             continue

    #         startPoint = Path['Path']['Points'][0,:][None,:]
    #         endPoint   = Path['Path']['Points'][-1,:][None,:]

    #         print('==================================================')
    #         print(' PATH: {} -> {} '.format(Path['from'],Path['to']))

    #         Points = np.concatenate([startPoint,
    #                                         Path['Path']['Points'][1:-1:2],
    #                                         endPoint])

    #         iter = 0
    #         while iter <= maxiter:
    #             id = 0

    #             while id <= (len(Points) - 3):
    #                 Sp  = tuple(Points[id,:])
    #                 Cp  = tuple(Points[id+1,:])
    #                 Np  = tuple(Points[id+2,:])

    #                 if (np.sqrt((Sp[0]-Cp[0])**2 + (Sp[1]-Cp[1])**2) < 1e-4)  or (np.sqrt((Np[0]-Cp[0])**2 + (Np[1]-Cp[1])**2) < 1e-4):
    #                     Points = np.delete(Points,id+1,axis=0)
    #                     continue

    #                 if abs(Sp[0]-Cp[0]) < 1e-4 or abs(Sp[1]-Cp[1]) < 1e-4  or abs(Np[0]-Cp[0]) < 1e-4 + abs(Np[1]-Cp[1] < 1e-4):
    #                     Points = np.delete(Points,id+1,axis=0)
    #                     continue


    #                 nc = NewtonianCurve(self.Mesh,Sp,Cp,Np,self.OptInfo['VehicleInfo']['Speed'],zerocurrents=self.zero_currents,debugging=debugging)
    #                 CrossingPoint, Boxes = nc.value()

    #                 Allowed = True
    #                 for box in Boxes:
    #                     if box.containsLand() or box.iceArea() >= self.OptInfo['MaxIceExtent']:
    #                         Allowed = False
    #                 if not Allowed:
    #                     id+=1
    #                     continue

    #                 if (np.isnan(CrossingPoint).any()):
    #                     Points = np.delete(Points,id+1,axis=0)
    #                 else:
    #                     Points[id+1,:] = CrossingPoint[0,:]
    #                     if CrossingPoint.shape[0] > 1:
    #                         Points = np.insert(Points,id+2,CrossingPoint[1:,:],0)
    #                 id+=1

    #             if iter!=0:
    #                 if Points.shape == oldPoints.shape:
    #                     if np.max(np.sqrt((Points-oldPoints)**2)) < minimumDiff:
    #                         break
                
    #             if iter == maxiter:
    #                 print('Maximum number of iterations met !')
                
    #             oldPoints = copy.deepcopy(Points)

    #             iter+=1

    #         print('{} iterations to convergence'.format(iter-1))

    #         Path['Path']['Points']       = Points
    #         SmoothedPaths.append(Path)
    #     return SmoothedPaths