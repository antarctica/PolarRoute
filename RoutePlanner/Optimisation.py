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

        # Creating a blank path construct
        self.paths         = None
        self.smoothedPaths = None

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
                        neighIndx.append(indx)
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

    def iceResistance(self, Cell):
        """
                Function to find the ice resistance force at a given speed in a given cell.

                Inputs:
                Cell - Cell box object

                Outputs:
                r - Resistance force
        """
        hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}

        hull = self.OptInfo['VehicleInfo']['HullType']
        beam = self.OptInfo['VehicleInfo']['Beam']
        k, b, n = hull_params[hull]
        g = 9.81  # m/s-2

        V = self.OptInfo['VehicleInfo']['Speed']*(5./18.)  # assume km/h and convert to m/s

        Fr = V/np.sqrt(g*Cell.iceArea()*Cell.iceThickness())

        r = 0.5*k*(Fr**b)*Cell.iceDensity()*beam*Cell.iceThickness()*(V**2)*(Cell.iceArea()**n)

        return r

    def inverseResistance(self, Fl, Cell):
        """
        Function to find the fastest speed that keeps the ice resistance force below a given threshold.

        Inputs:
        Fl - Force limit
        Cell - Cell box object

        Outputs:
        v - speed
        """
        hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}

        hull = self.OptInfo['VehicleInfo']['HullType']
        beam = self.OptInfo['VehicleInfo']['Beam']
        k, b, n = hull_params[hull]
        g = 9.81  # m/s-2

        exp = 2.0 + b

        vexp = 2*Fl/(k*Cell.iceDensity()*beam*Cell.iceThickness()*(Cell.iceArea()**n)*(g*Cell.iceThickness()*Cell.iceArea())**-(b/2))

        vms = vexp**(1/exp)
        v = vms*(18./5.)  # convert from m/s to km/h

        return v

    def speedFunction(self, Cell):
        if self.variableSpeed:
            if Cell.iceArea() == 0.0:
                s = self.OptInfo['VehicleInfo']['Speed']
            elif self.iceResistance(Cell) < self.OptInfo['VehicleInfo']['ForceLimit']:
                s = self.OptInfo['VehicleInfo']['Speed']
            else:
                s = self.inverseResistance(self.OptInfo['VehicleInfo']['ForceLimit'], Cell)
        else:
            s = self.OptInfo['VehicleInfo']['Speed']
        return s


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
                    Path['Path']                = {}
                    Path['Path']['Points']      = np.array(wpt_a_loc+list(np.array(Graph['pathPoints'].loc[wpt_b_index])[:-1,:])+wpt_b_loc)
                    Path['Path']['CellIndices'] = np.array(Graph['pathIndex'].loc[wpt_b_index])
                    Path['Path']['CaseTypes']   = np.array([wpt_a_index] + Graph['pathPoints'].loc[wpt_b_index] + [wpt_b_index])
                    Path['Path']['Time']        = PathTT
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
        # Determining the nearest neighbour index for the cell
        Sc = self.Mesh.cellBoxes[minimumTravelTimeIndex]
        SourceGraph = self.DijkstraInfo[wpt_name].loc[minimumTravelTimeIndex]

        # Looping over idx
        for idx in range(len(SourceGraph['case'])):
            indx = SourceGraph['neighbourIndex'][idx]
            Nc   = self.Mesh.cellBoxes[indx]
            Case = SourceGraph['case'][idx]
            

            # Set travel-time to infinite if neighbour is land or ice-thickness is too large.
            if (Nc.iceArea() >= self.OptInfo['MaxIceExtent']) or (Sc.iceArea() >= self.OptInfo['MaxIceExtent']) or (Nc.containsLand()) or (Sc.containsLand()):
                SourceGraph['neighbourTravelLegs'].append([np.inf,np.inf])
                SourceGraph['neighbourCrossingPoints'].append([np.nan,np.nan])
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
            Neighbour_cost  = [SourceGraph['traveltime'] + TravelTime[0],SourceGraph['traveltime'] + np.sum(TravelTime)]
            NeighbourGraph  = self.DijkstraInfo[wpt_name].loc[indx]

            if Neighbour_cost[1] < NeighbourGraph['traveltime']:
                NeighbourGraph['traveltime'] = Neighbour_cost[1]
                NeighbourGraph['pathIndex']  = SourceGraph['pathIndex']  + [indx]
                NeighbourGraph['pathCost']   = SourceGraph['pathCost']   + Neighbour_cost
                NeighbourGraph['pathPoints'] = SourceGraph['pathPoints'] + [list(CrossPoints)] + [list(CellPoints)]

            self.DijkstraInfo[wpt_name].loc[SourceGraph['neighbourIndex'][idx]] = NeighbourGraph

        self.DijkstraInfo[wpt_name].loc[minimumTravelTimeIndex] = SourceGraph


    def _dijkstra(self,wpt_name):
        # Including only the End Waypoints defined by the user
        Wpts = self.OptInfo['WayPoints'][self.OptInfo['WayPoints']['Name'].isin(self.end_waypoints)]
        
        # Initalising zero traveltime at the source location
        SourceIndex = int(self.OptInfo['WayPoints'][self.OptInfo['WayPoints']['Name'] == wpt_name]['index'])
        self.DijkstraInfo[wpt_name].loc[SourceIndex,'traveltime'] = 0.0
        self.DijkstraInfo[wpt_name].loc[SourceIndex,'pathIndex'].append(SourceIndex)
        
        # Updating Dijkstra as long as all the waypoints are not visited.
        while (self.DijkstraInfo[wpt_name].loc[Wpts['index'],'positionLocked'] == False).any():

            # Determining the index of the minimum traveltime that has not been visited
            minimumTravelTimeIndex = self.DijkstraInfo[wpt_name][self.DijkstraInfo[wpt_name]['positionLocked']==False]['traveltime'].idxmin()

            # Finding the cost of the nearest neighbours from the source cell (Sc)
            self.NeighbourCost(wpt_name,minimumTravelTimeIndex)

            # Updating Position to be locked
            self.DijkstraInfo[wpt_name].loc[minimumTravelTimeIndex,'positionLocked'] = True

        # Correct travel-time off grid for start and end indices
        # ----> TO-DO 

    def Paths(self,source_waypoints=None,end_waypoints=None,verbose=False,multiprocessing=False,return_paths=True):
        '''
        Determining the shortest path between all waypoints
        '''

        # Subsetting the waypoints
        if type(source_waypoints) == type(None):
            source_waypoints = list(self.OptInfo['WayPoints']['Name'])
        if type(end_waypoints) == type(None):
            self.end_waypoints = list(self.OptInfo['WayPoints']['Name'])
        else:
            self.end_waypoints = end_waypoints

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
                if verbose:
                    print('=== Processing Waypoint = {} ==='.format(wpt))
                self._dijkstra(wpt)


        self.paths = self.Dijkstra2Path(source_waypoints,self.end_waypoints)
        if return_paths:
            return self.paths

         

    def PathSmoothing(self,maxiter=1000,minimumDiff=1e-4,debugging=0,return_paths=True,verbose=False):
        '''
            Given a series of pathways smooth without centroid locations using great circle smoothing
        '''
        import copy

        SmoothedPaths = []

        if type(self.paths) == type(None):
            raise Exception('Paths not constructed, please re-run path construction')
        Pths = copy.deepcopy(self.paths)
        # Looping over all the optimised paths
        for indx_Path in range(len(Pths)):
            Path = Pths[indx_Path]
            if Path['Time'] == np.inf:
                continue

            startPoint = Path['Path']['Points'][0,:][None,:]
            endPoint   = Path['Path']['Points'][-1,:][None,:]

            if verbose:
                print('==================================================')
                print(' PATH: {} -> {} '.format(Path['from'],Path['to']))

            Points      = np.concatenate([startPoint,Path['Path']['Points'][1:-1:2],endPoint])
            cellIndices = np.concatenate((Path['Path']['CellIndices'],Path['Path']['CellIndices'][-1][None]))

            nc = NewtonianCurve(self.Mesh,self.DijkstraInfo[Path['from']],self.OptInfo,zerocurrents=self.zero_currents,debugging=debugging)
            nc.PathSmoothing(Points,cellIndices)

            Path['Path']['Points']       = nc.path
            SmoothedPaths.append(Path)

        self.smoothedPaths = SmoothedPaths
        if return_paths:
            return self.smoothedPaths