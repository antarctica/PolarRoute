import numpy as np
import copy
import pandas as pd

from RoutePlanner.Function import NewtonianDistance, NewtonianCurve
from RoutePlanner.CellBox import CellBox

import numpy as np
import copy
import pandas as pd


from RoutePlanner.Function import NewtonianDistance, NewtonianCurve
from RoutePlanner.CellBox import CellBox

class TravelTime:
    def __init__(self,CellGrid,config,neighbourGraph=None,CostFunc=NewtonianDistance):
        # Load in the current cell structure & Optimisation Info
        self.Mesh    = copy.copy(CellGrid)
        self.config  = config

        # Creating a blank path construct
        self.paths         = None
        self.smoothedPaths = None

        # # Constructing Neighbour Graph
        if type(neighbourGraph) == type(None):
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
                            if (self.Mesh.cellBoxes[indx].iceArea() >= self.config['Vehicle_Info']['MaxIceExtent']):
                                continue
                            if self.Mesh._j_grid:
                                if self.Mesh.cellBoxes[indx].isLandM():
                                    continue
                            else:
                                if self.Mesh.cellBoxes[indx].containsLand():
                                    continue
                            cases.append(case)
                            neighIndx.append(indx)
                    neighDict = {}
                    neighDict['cX']    = cell.cx
                    neighDict['cY']    = cell.cy
                    neighDict['case']  = cases
                    neighDict['neighbourIndex'] = neighIndx 
                    neighbourGraph[idx] = neighDict
            self.neighbourGraph = pd.DataFrame().from_dict(neighbourGraph,orient='index')
        else:
            self.neighbourGraph = neighbourGraph
        self.neighbourGraph['positionLocked'] = False
        self.neighbourGraph['traveltime']     = np.inf
        self.neighbourGraph['neighbourTravelLegs'] = [list() for x in range(len(self.neighbourGraph.index))]
        self.neighbourGraph['neighbourCrossingPoints'] = [list() for x in range(len(self.neighbourGraph.index))]
        self.neighbourGraph['pathIndex']  = [list() for x in range(len(self.neighbourGraph.index))]
        self.neighbourGraph['pathCost']   = [list() for x in range(len(self.neighbourGraph.index))]
        self.neighbourGraph['pathPoints']   = [list() for x in range(len(self.neighbourGraph.index))]

        self.CostFunc       = CostFunc

        self.unit_shipspeed = self.config['Vehicle_Info']['Unit']
        self.unit_time      = self.config['Route_Info']['Time_Unit']
        self.zero_currents  = self.config['Route_Info']['Zero_Currents']
        self.variableSpeed  =self. config['Route_Info']['Variable_Speed']

        print('Zero Currets {}'.format(self.zero_currents))


        if type(self.config['Route_Info']['WayPoints']) != pd.core.frame.DataFrame:
            self.config['Route_Info']['WayPoints'] = pd.read_csv(self.config['Route_Info']['WayPoints'])
        

        # ====== Waypoints ======        
        # Dropping waypoints outside domain
        self.config['Route_Info']['WayPoints'] = self.config['Route_Info']['WayPoints'][(self.config['Route_Info']['WayPoints']['Long'] >= self.Mesh._longMin) &\
                                                              (self.config['Route_Info']['WayPoints']['Long'] <= self.Mesh._longMax) &\
                                                              (self.config['Route_Info']['WayPoints']['Lat'] <= self.Mesh._latMax) &\
                                                              (self.config['Route_Info']['WayPoints']['Lat'] >= self.Mesh._latMin)] 

        # Initialising Waypoints positions and cell index
        self.config['Route_Info']['WayPoints']['index'] = np.nan
        for idx,wpt in self.config['Route_Info']['WayPoints'].iterrows():
            long = wpt['Long']
            lat  = wpt['Lat']
            for index, cell in enumerate(self.Mesh.cellBoxes):
                if isinstance(cell, CellBox):
                    if cell.containsPoint(lat,long):
                        break
            self.config['Route_Info']['WayPoints']['index'].loc[idx] = index

    def ice_resistance(self, cell):
        """
                Function to find the ice resistance force at a given speed in a given cell.

                Inputs:
                cell - Cell box object

                Outputs:
                resistance - Resistance force
        """
        hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}

        hull = self.config['Vehicle_Info']['HullType']
        beam = self.config['Vehicle_Info']['Beam']
        kparam, bparam, nparam = hull_params[hull]
        gravity = 9.81  # m/s-2
        speed = self.config['Vehicle_Info']['Speed']*(5./18.)  # assume km/h and convert to m/s
        force_limit = speed/np.sqrt(gravity*cell.iceArea()*cell.iceThickness(self.config['Region']['startTime']))
        resistance = 0.5*kparam*(force_limit**bparam)*cell.iceDensity(self.config['Region']['startTime'])*beam*cell.iceThickness(self.config['Region']['startTime'])*(speed**2)*(cell.iceArea()**nparam)
        return resistance

    def inverse_resistance(self, force_limit, cell):
        """
        Function to find the speed that keeps the ice resistance force below a given threshold.

        Inputs:
        force_limit - Force limit
        cell        - Cell box object

        Outputs:
        speed - Vehicle Speed
        """
        hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}

        hull = self.config['Vehicle_Info']['HullType']
        beam = self.config['Vehicle_Info']['Beam']
        kparam, bparam, nparam = hull_params[hull]
        gravity = 9.81  # m/s-2

        vexp = 2*force_limit/(kparam*cell.iceDensity(self.config['Region']['startTime'])*beam*cell.iceThickness(self.config['Region']['startTime'])*(cell.iceArea()**nparam)*(gravity*cell.iceThickness(self.config['Region']['startTime'])*cell.iceArea())**-(bparam/2))

        vms = vexp**(1/(2.0 + bparam))
        speed = vms*(18./5.)  # convert from m/s to km/h

        return speed

    def speedFunction(self, cell):
        '''
            FILL
        '''
        if self.variableSpeed:
            if cell.iceArea() == 0.0:
                speed = self.config['Vehicle_Info']['Speed']
            elif self.ice_resistance(cell) < self.config['Vehicle_Info']['ForceLimit']:
                speed = self.config['Vehicle_Info']['Speed']
            else:
                speed = self.inverse_resistance(self.config['Vehicle_Info']['ForceLimit'], cell)
        else:
            speed = self.config['Vehicle_Info']['Speed']
        return speed




    def Dijkstra2Path(self,StartWaypoints,EndWaypoints):
        Paths = []

        wpts_s = self.config['Route_Info']['WayPoints'][self.config['Route_Info']['WayPoints']['Name'].isin(StartWaypoints)]
        wpts_e = self.config['Route_Info']['WayPoints'][self.config['Route_Info']['WayPoints']['Name'].isin(EndWaypoints)]

        for idx,wpt_a in wpts_s.iterrows():
            wpt_a_name  = wpt_a['Name']; wpt_a_index = int(wpt_a['index']); wpt_a_loc   = [[wpt_a['Long'],wpt_a['Lat']]]
            for idy,wpt_b in wpts_e.iterrows():
                wpt_b_name  = wpt_b['Name']; wpt_b_index = int(wpt_b['index']); wpt_b_loc   = [[wpt_b['Long'],wpt_b['Lat']]]
                if not wpt_a_name == wpt_b_name:
                    try:
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
                    except:
                        print('Failure')
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
        Sc          = self.Mesh.cellBoxes[minimumTravelTimeIndex]
        SourceGraph = self.DijkstraInfo[wpt_name].loc[minimumTravelTimeIndex]

        # Looping over idx
        for idx in range(len(SourceGraph['case'])):
            indx = SourceGraph['neighbourIndex'][idx]
            
            # Don't inspect neighbour if position is already locked
            if self.DijkstraInfo[wpt_name].loc[indx,'positionLocked']:
                continue
            
            Nc   = self.Mesh.cellBoxes[indx]
            Case = SourceGraph['case'][idx]
            

            # Set travel-time to infinite if neighbour is land or ice-thickness is too large.
            if (Nc.iceArea() >= self.config['Vehicle_Info']['MaxIceExtent']):
                SourceGraph['neighbourTravelLegs'].append([np.inf,np.inf])
                SourceGraph['neighbourCrossingPoints'].append([np.nan,np.nan])
                continue

            if self.Mesh._j_grid:
                if Nc.isLandM():
                    SourceGraph['neighbourTravelLegs'].append([np.inf,np.inf])
                    SourceGraph['neighbourCrossingPoints'].append([np.nan,np.nan])
                    continue
            else:
                if Nc.containsLand():
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
                NeighbourGraph = pd.Series(
                    {
                    'cX':NeighbourGraph['cX'],
                    'cY':NeighbourGraph['cY'],
                    'case':NeighbourGraph['case'],
                    'neighbourIndex':NeighbourGraph['neighbourIndex'],
                    'positionLocked':NeighbourGraph['positionLocked'],
                    'traveltime': Neighbour_cost[1],
                    'neighbourTravelLegs':NeighbourGraph['neighbourTravelLegs'],
                    'neighbourCrossingPoints':NeighbourGraph['neighbourCrossingPoints'],
                    'pathIndex': SourceGraph['pathIndex']  + [indx],
                    'pathCost':SourceGraph['pathCost']   + Neighbour_cost,
                    'pathPoints':SourceGraph['pathPoints'] +[list(CrossPoints)] +[list(CellPoints)]}
                    )



                self.DijkstraInfo[wpt_name].loc[indx] = NeighbourGraph

        self.DijkstraInfo[wpt_name].loc[minimumTravelTimeIndex] = SourceGraph


    def _dijkstra(self,wpt_name):
        # Including only the End Waypoints defined by the user
        Wpts = self.config['Route_Info']['WayPoints'][self.config['Route_Info']['WayPoints']['Name'].isin(self.end_waypoints)]
        
        # Initalising zero traveltime at the source location
        SourceIndex = int(self.config['Route_Info']['WayPoints'][self.config['Route_Info']['WayPoints']['Name'] == wpt_name]['index'])
        self.DijkstraInfo[wpt_name].loc[SourceIndex,'traveltime'] = 0.0
        self.DijkstraInfo[wpt_name].loc[SourceIndex,'pathIndex'].append(SourceIndex)
        
        # Updating Dijkstra as long as all the waypoints are not visited.
        while (self.DijkstraInfo[wpt_name].loc[Wpts['index'],'positionLocked'] == False).any():
        #while (self.DijkstraInfo[wpt_name]['positionLocked'] == False).any():    

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

        self.source_waypoints = source_waypoints
        self.end_waypoints    = end_waypoints

        # Subsetting the waypoints
        if type(source_waypoints) == type(None):
            source_waypoints = list(self.config['Route_Info']['WayPoints']['Name'])
        if type(end_waypoints) == type(None):
            self.end_waypoints = list(self.config['Route_Info']['WayPoints']['Name'])
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

         

    def PathSmoothing(self,maxiter=10000,minimumDiff=1e-4,debugging=0,return_paths=True,verbose=False):
        '''
            Given a series of pathways smooth without centroid locations using great circle smoothing
        '''
        import copy

        SmoothedPaths = []

        if type(self.paths) == type(None):
            raise Exception('Paths not constructed, please re-run path construction')
        Pths = copy.deepcopy(self.paths)

        for ii in range(len(Pths)):
            Path = Pths[ii]
            print('===Smoothing {} to {} ======'.format(Path['from'],Path['to']))

            nc = NewtonianCurve(self.Mesh,self.DijkstraInfo[Path['from']],self.config,maxiter=1,zerocurrents=True)
            nc.pathIter = maxiter

            # -- Generating a dataframe of the case information -- 
            Points      = np.concatenate([Path['Path']['Points'][0,:][None,:],Path['Path']['Points'][1:-1:2],Path['Path']['Points'][-1,:][None,:]])
            cellIndices = np.concatenate([Path['Path']['CellIndices'][0][None],Path['Path']['CellIndices'],Path['Path']['CellIndices'][-1][None]])
            cellDijk    = [nc.DijkstraInfo.loc[ii] for ii in cellIndices]
            nc.CrossingDF  = pd.DataFrame({'cX':Points[:,0],'cY':Points[:,1],'cellStart':cellDijk[:-1],'cellEnd':cellDijk[1:]})

            # -- Determining the cases from the cell information. If within cell then case 0 -- 
            Cases = []
            for idx,row in nc.CrossingDF.iterrows():
                try:
                    Cases.append(row.cellStart['case'][np.where(np.array(row.cellStart['neighbourIndex'])==row.cellEnd.name)[0][0]])
                except:
                    Cases.append(0)
            nc.CrossingDF['case'] = Cases
            nc.CrossingDF.index = np.arange(int(nc.CrossingDF.index.min()),int(nc.CrossingDF.index.max()*1e3 + 1e3),int(1e3))

            nc.orgDF = copy.deepcopy(nc.CrossingDF)
            iter=0
            #try:
            while iter < nc.pathIter:

                nc.previousDF = copy.deepcopy(nc.CrossingDF)
                id = 0
                while id <= (len(nc.CrossingDF) - 3):
                    nc.triplet = nc.CrossingDF.iloc[id:id+3]


                    nc._updateCrossingPoint()
                    self.nc = nc

                    # -- Horseshoe Case Detection -- 
                    nc._horseshoe()
                    # -- Removing reseverse cases
                    nc._reverseCase()

                    id+=1+nc.id

                self.nc = nc
                nc._mergePoint()
                self.nc = nc
                iter+=1

                # Stop optimisation if the points are within some minimum difference
                if len(nc.previousDF) == len(nc.CrossingDF):
                    Dist = np.max(np.sqrt((nc.previousDF['cX'] - nc.CrossingDF['cX'])**2 + (nc.previousDF['cY'] - nc.CrossingDF['cY'])**2))
                    if Dist < 1e-3:
                        break
            print('{} iterations'.format(iter))
            # except:
            #     print('Failed {}->{}'.format(Path['from'],Path['to']))

            SmoothedPath ={}
            SmoothedPath['from'] = Path['from']
            SmoothedPath['to']   = Path['to']
            SmoothedPath['Path'] = {}
            SmoothedPath['Path']['Points'] = nc.CrossingDF[['cX','cY']].to_numpy()    
            SmoothedPaths.append(SmoothedPath)

        

        self.smoothedPaths = SmoothedPaths
        if return_paths:
            return self.smoothedPaths


        # SmoothedPaths = []

        # if type(self.paths) == type(None):
        #     raise Exception('Paths not constructed, please re-run path construction')
        # Pths = copy.deepcopy(self.paths)
        # # Looping over all the optimised paths
        # for indx_Path in range(len(Pths)):
        #     Path = Pths[indx_Path]
        #     if Path['Time'] == np.inf:
        #         continue

        #     startPoint = Path['Path']['Points'][0,:][None,:]
        #     endPoint   = Path['Path']['Points'][-1,:][None,:]

        #     if verbose:
        #         print('==================================================')
        #         print(' PATH: {} -> {} '.format(Path['from'],Path['to']))

        #     Points      = np.concatenate([startPoint,Path['Path']['Points'][1:-1:2],endPoint])
        #     cellIndices = np.concatenate((Path['Path']['CellIndices'],Path['Path']['CellIndices'][-1][None]))

        #     nc = NewtonianCurve(self.Mesh,self.DijkstraInfo[Path['from']],self.OptInfo,zerocurrents=self.zero_currents,debugging=debugging)
        #     nc.PathSmoothing(Points,cellIndices)

        #     Path['Path']['Points']       = nc.path
        #     SmoothedPaths.append(Path)

        # self.smoothedPaths = SmoothedPaths
        # if return_paths:
        #     return self.smoothedPaths


# '''
#     FILL IN
# '''

# # == Packages ==
# import multiprocessing as mp
# import copy
# import time

# import numpy as np
# import pandas as pd

# from RoutePlanner.CrossingPoint import NewtonianCrossingPoint, NewtonianCrossingPointSmooth
# from RoutePlanner.CellBox import CellBox

# class TravelTime:
#     '''
#     FILL IN
#     '''
#     def __init__(self, mesh, config,neighbour_graph=None,
#                  cost_func=NewtonianCrossingPoint,
#                  smooth_cost_func=NewtonianCrossingPointSmooth):
#         start_time = time.time()
#         # Load in the current cell structure & Optimisation Info
#         self.mesh    = copy.copy(mesh)
#         self.config  = config

#         self.cost_func        = cost_func
#         self.smooth_cost_func = smooth_cost_func

#         self.unit_shipspeed  = self.config['Vehicle_Info']['Unit']
#         self.unit_time       = self.config['Route_Info']['Time_Unit']
#         self.zero_currents   = self.config['Route_Info']['Zero_Currents']
#         self.variable_speed  = self.config['Route_Info']['Variable_Speed']

#         # Creating a blank path construct
#         self.paths         = None
#         self.smoothed_paths = None

#         self.test = 0

#         # # Constructing Neighbour Graph
#         if isinstance(neighbour_graph, type(None)):
#             neighbour_graph = {}
#             for idx,cell in enumerate(self.mesh.cellboxes):
#                 if not isinstance(cell, CellBox):
#                     continue
#                 else:
#                     neigh     = self.mesh.neighbour_graph[idx]
#                     cases     = []
#                     neigh_indx = []
#                     for case in neigh.keys():
#                         indxs = neigh[case]
#                         if len(indxs) == 0:
#                             continue
#                         for indx in indxs:
#                             if self.mesh.cellboxes[indx].iceArea() >= \
#                                  self.config['Vehicle_Info']['MaxIceExtent']:
#                                 continue
#                             if self.mesh._j_grid:
#                                 if self.mesh.cellboxes[indx].isLandM():
#                                     continue
#                             else:
#                                 if self.mesh.cellboxes[indx].containsLand():
#                                     continue
#                             cases.append(case)
#                             neigh_indx.append(indx)
#                     neigh_dict = {}
#                     neigh_dict['cX']    = cell.cx
#                     neigh_dict['cY']    = cell.cy
#                     neigh_dict['case']  = cases
#                     neigh_dict['neighbourIndex'] = neigh_indx
#                     neighbour_graph[idx] = neigh_dict
#             self.neighbour_graph = pd.DataFrame().from_dict(neighbour_graph,orient='index')
#         else:
#             self.neighbour_graph = neighbour_graph
#         self.neighbour_graph['positionLocked'] = False
#         self.neighbour_graph['traveltime']     = np.inf
#         self.neighbour_graph['neighbourTravelLegs'] = \
#             [list() for x in range(len(self.neighbour_graph.index))]
#         self.neighbour_graph['neighbourCrossingPoints'] = \
#             [list() for x in range(len(self.neighbour_graph.index))]
#         self.neighbour_graph['pathIndex'] = \
#              [list() for x in range(len(self.neighbour_graph.index))]
#         self.neighbour_graph['pathCost'] = \
#              [list() for x in range(len(self.neighbour_graph.index))]
#         self.neighbour_graph['pathPoints'] = \
#              [list() for x in range(len(self.neighbour_graph.index))]


#         print('Zero Currets', self.zero_currents)


#         if not isinstance(self.config['Route_Info']['WayPoints'],pd.core.frame.DataFrame):
#             self.config['Route_Info']['WayPoints'] = \
#                  pd.read_csv(self.config['Route_Info']['WayPoints'])

#         # ====== Waypoints ======
#         # Dropping waypoints outside domain
#         self.config['Route_Info']['WayPoints'] =\
#              self.config['Route_Info']['WayPoints'][(self.config['Route_Info']['WayPoints']['Long']\
#              >= self.mesh._longMin) & (self.config['Route_Info']['WayPoints']['Long'] <= \
#              self.mesh._longMax) & (self.config['Route_Info']['WayPoints']['Lat'] <= \
#              self.mesh._latMax) & (self.config['Route_Info']['WayPoints']['Lat'] >= \
#              self.mesh._latMin)]

#         # Initialising Waypoints positions and cell index
#         self.config['Route_Info']['WayPoints']['index'] = np.nan
#         for idx,wpt in self.config['Route_Info']['WayPoints'].iterrows():
#             long = wpt['Long']
#             lat  = wpt['Lat']
#             for index, cell in enumerate(self.mesh.cellboxes):
#                 if isinstance(cell, CellBox):
#                     if cell.containsPoint(lat,long):
#                         break
#             self.config['Route_Info']['WayPoints']['index'].loc[idx] = index

#         print("--- Initial Setup -",(time.time() - start_time),"seconds ---")


#     def ice_resistance(self, cell):
#         """
#                 Function to find the ice resistance force at a given speed in a given cell.

#                 Inputs:
#                 cell - Cell box object

#                 Outputs:
#                 resistance - Resistance force
#         """
#         hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}

#         hull = self.config['Vehicle_Info']['HullType']
#         beam = self.config['Vehicle_Info']['Beam']
#         kparam, bparam, nparam = hull_params[hull]
#         gravity = 9.81  # m/s-2
#         speed = self.config['Vehicle_Info']['Speed']*(5./18.)  # assume km/h and convert to m/s
#         force_limit = speed/np.sqrt(gravity*cell.iceArea()*\
#             cell.iceThickness(self.config['Region']['startTime']))
#         resistance = 0.5*kparam,*(force_limit**bparam,)*\
#             cell.iceDensity(self.config['Region']['startTime'])*\
#             beam*cell.iceThickness(self.config['Region']['startTime'])\
#             *(speed**2)*(cell.iceArea()**nparam)
#         return resistance

#     def inverse_resistance(self, force_limit, cell):
#         """
#         Function to find the speed that keeps the ice resistance force below a given threshold.

#         Inputs:
#         force_limit - Force limit
#         cell        - Cell box object

#         Outputs:
#         speed - Vehicle Speed
#         """
#         hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}

#         hull = self.config['Vehicle_Info']['HullType']
#         beam = self.config['Vehicle_Info']['Beam']
#         kparam, bparam, nparam = hull_params[hull]
#         gravity = 9.81  # m/s-2

#         vexp = 2*force_limit/(kparam*cell.iceDensity(self.config['Region']['startTime'])*\
#                beam*cell.iceThickness(self.config['Region']['startTime'])*\
#                (cell.iceArea()**nparam)*\
#                (gravity*cell.iceThickness(self.config['Region']['startTime'])*\
#                cell.iceArea())**-(bparam/2))

#         vms = vexp**(1/(2.0 + bparam))
#         speed = vms*(18./5.)  # convert from m/s to km/h

#         return speed

#     def speed_function(self, cell):
#         '''
#             FILL
#         '''
#         if self.variable_speed:
#             #s = (1-np.sqrt(Cell.iceArea()))*self.config['Vehicle_Info']['Speed']
#             if cell.iceArea() == 0.0:
#                 speed = self.config['Vehicle_Info']['Speed']
#             elif self.ice_resistance(cell) < self.config['Vehicle_Info']['ForceLimit']:
#                 speed = self.config['Vehicle_Info']['Speed']
#             else:
#                 speed = self.inverse_resistance(self.config['Vehicle_Info']['ForceLimit'], cell)
#         else:
#             speed = self.config['Vehicle_Info']['Speed']
#         return speed


#     def dijkstra_path(self,start_waypoints,end_waypoints):
#         '''
#             FILL
#         '''
#         paths = []

#         wpts_s = self.config['Route_Info']['WayPoints']\
#             [self.config['Route_Info']['WayPoints']['Name'].isin(start_waypoints)]
#         wpts_e = self.config['Route_Info']['WayPoints']\
#             [self.config['Route_Info']['WayPoints']['Name'].isin(end_waypoints)]

#         for _,wpt_a in wpts_s.iterrows():
#             wpt_a_name  = wpt_a['Name']
#             wpt_a_index = int(wpt_a['index'])
#             wpt_a_loc   = [[wpt_a['Long'],wpt_a['Lat']]]
#             for _,wpt_b in wpts_e.iterrows():
#                 wpt_b_name  = wpt_b['Name']
#                 wpt_b_index = int(wpt_b['index'])
#                 wpt_b_loc   = [[wpt_b['Long'],wpt_b['Lat']]]
#                 if not wpt_a_name == wpt_b_name:
#                     try:
#                         # ==== Correcting Path for waypoints off cell
#                         path = {}
#                         path['from']               = wpt_a_name
#                         path['to']                 = wpt_b_name

#                         graph = self.dijkstra_info[wpt_a_name]
#                         path['Time']               = float(graph['traveltime'].loc[wpt_b_index])
#                         if path['Time'] == np.inf:
#                             continue
#                         path_traveltime     = graph['pathCost'].loc[wpt_b_index]

#                         # ===== Appending Path =====
#                         path['Path']                = {}
#                         path['Path']['Points']      = np.array(wpt_a_loc +\
#                             list(np.array(graph['pathPoints'].loc[wpt_b_index])[:-1,:])+ wpt_b_loc)
#                         path['Path']['CellIndices'] = np.array(graph['pathIndex'].loc[wpt_b_index])
#                         path['Path']['CaseTypes']   = np.array([wpt_a_index] +\
#                             graph['pathPoints'].loc[wpt_b_index] + [wpt_b_index])
#                         path['Path']['Time']        = path_traveltime
#                         paths.append(path)
#                     except:
#                         print('Failure')
#         return paths

#     def neighbour_cost(self,wpt_name,minimum_traveltime_index):
#         '''
#         Function for computing the shortest travel-time from a cell to its neighbours
#         by applying the Newtonian method for optimisation
#         Inputs:
#         index - Index of the cell to process
#         Output:
#         Bugs/Alterations:
#             - If corner of cell is land in adjacent cell then also return 'inf'
#         '''
#         # Determining the nearest neighbour index for the cell
#         source_cell  = self.mesh.cellboxes[minimum_traveltime_index]
#         source_graph = self.dijkstra_info[wpt_name].loc[minimum_traveltime_index]

#         # Looping over idx
#         for idx in range(len(source_graph['case'])):
#             indx = source_graph['neighbourIndex'][idx]

#             # Don't inspect neighbour if position is already locked
#             if self.dijkstra_info[wpt_name].loc[indx,'positionLocked']:
#                 continue

#             neigh_cell = self.mesh.cellboxes[indx]
#             case       = source_graph['case'][idx]

#             # Set travel-time to infinite if neighbour is land or ice-thickness is too large.
#             if neigh_cell.iceArea() >= self.config['Vehicle_Info']['MaxIceExtent']:
#                 source_graph['neighbourTravelLegs'].append([np.inf,np.inf])
#                 source_graph['neighbourCrossingPoints'].append([np.nan,np.nan])
#                 continue

#             if self.mesh._j_grid:
#                 if neigh_cell.isLandM():
#                     source_graph['neighbourTravelLegs'].append([np.inf,np.inf])
#                     source_graph['neighbourCrossingPoints'].append([np.nan,np.nan])
#                     continue
#             else:
#                 if neigh_cell.containsLand():
#                     source_graph['neighbourTravelLegs'].append([np.inf,np.inf])
#                     source_graph['neighbourCrossingPoints'].append([np.nan,np.nan])
#                     continue

#             source_speed = self.speed_function(source_cell)
#             neigh_speed = self.speed_function(neigh_cell)

#             # Applying Newton curve to determine crossing point
#             if self.test ==0:
#                 start_time = time.time()
#             cost_func  = self.cost_func(self.mesh,source_cell=source_cell,neigh_cell=neigh_cell,
#                             source_Speed=source_speed,neigh_Speed=neigh_speed,case=case,
#                             unit_shipspeed='km/hr',unit_time=self.unit_time,
#                             zerocurrents=self.zero_currents)
#             # Updating the Dijkstra graph with the new information
#             traveltime,cross_points,cell_points = cost_func.value()
#             if self.test ==0:
#                 print("--- neighbour_cost - cost_func -", (time.time() - start_time), "seconds ---")

#             if self.test ==0:
#                 start_time = time.time()

#             source_graph['neighbourTravelLegs'].append(traveltime)
#             source_graph['neighbourCrossingPoints'].append(cross_points)

#             # Updating the neighbour traveltime if its less the current global optimum
#             neighbour_cost  = [source_graph['traveltime'] +\
#                 traveltime[0],source_graph['traveltime'] + np.sum(traveltime)]
#             neighbour_graph  = self.dijkstra_info[wpt_name].loc[indx]

#             if self.test ==0:
#                 print("--- neighbour_cost - Updating Graph -indx=" +\
#                 indx + "- Part1 -" + (time.time() - start_time) + "seconds ---")

#             if self.test ==0:
#                 start_time = time.time()
#             if neighbour_cost[1] < neighbour_graph['traveltime']:
#                 neighbour_graph = pd.Series(
#                     {
#                     'cX':neighbour_graph['cX'],
#                     'cY':neighbour_graph['cY'],
#                     'case':neighbour_graph['case'],
#                     'neighbourIndex':neighbour_graph['neighbourIndex'],
#                     'positionLocked':neighbour_graph['positionLocked'],
#                     'traveltime': neighbour_cost[1],
#                     'neighbourTravelLegs':neighbour_graph['neighbourTravelLegs'],
#                     'neighbourCrossingPoints':neighbour_graph['neighbourCrossingPoints'],
#                     'pathIndex': source_graph['pathIndex']  + [indx],
#                     'pathCost':source_graph['pathCost']   + neighbour_cost,
#                     'pathPoints':source_graph['pathPoints'] +\
#                                 [list(cross_points)] +\
#                                 [list(cell_points)]}
#                     )


#             self.dijkstra_info[wpt_name].loc[indx] = neighbour_graph
#             if self.test ==0:
#                 print("--- neighbour_cost - Updating Graph - Part2 -" +
#                     (time.time() - start_time) +
#                      "seconds ---")



#         self.dijkstra_info[wpt_name].loc[minimum_traveltime_index] = source_graph


#     def _dijkstra(self,wpt_name):
#         start_time =time.time()
#         # Including only the End Waypoints defined by the user
#         wpts = self.config['Route_Info']['WayPoints']\
#             [self.config['Route_Info']['WayPoints']['Name'].isin(self.end_waypoints)]

#         # Initalising zero traveltime at the source location
#         source_index = int(self.config['Route_Info']['WayPoints']
#             [self.config['Route_Info']['WayPoints']['Name'] == wpt_name]['index'])
#         self.dijkstra_info[wpt_name].loc[source_index,'traveltime'] = 0.0
#         self.dijkstra_info[wpt_name].loc[source_index,'pathIndex'].append(source_index)

#         print("--- Dijkstr Run Preface -",(time.time() - start_time),"seconds ---")

#         while (self.dijkstra_info[wpt_name].loc[wpts['index'],'positionLocked'] is False).any():
#         #while (self.dijkstra_info[wpt_name]['positionLocked'] == False).any():

#             if self.test == 0:
#                 start_time = time.time()
#             # Determining the index of the minimum traveltime that has not been visited
#             minimum_traveltime_index = self.dijkstra_info[wpt_name]\
#                 [self.dijkstra_info[wpt_name]['positionLocked'] is False]['traveltime'].idxmin()
#             if self.test == 0:
#                 print("--- Dijkstr Index of Minimum traveltime -" +\
#                      (time.time() - start_time) + "seconds ---")


#             if self.test == 0:
#                 start_time = time.time()
#             # Finding the cost of the nearest neighbours from the source cell (Sc)
#             self.neighbour_cost(wpt_name,minimum_traveltime_index)
#             if self.test == 0:
#                 print("--- Dijkstr Index of neighbour_cost -" +
#                      (time.time() - start_time) + "seconds ---")
#                 self.test+=1


#             # Updating Position to be locked
#             self.dijkstra_info[wpt_name].loc[minimum_traveltime_index,'positionLocked'] = True

#         # Correct travel-time off grid for start and end indices
#         # ----> TO-DO



#     def compute_paths(self,source_waypoints=None,end_waypoints=None,
#                      verbose=False,return_paths=True):
#         '''
#         Determining the shortest path between all waypoints
#         '''

#         self.dijkstra_info    = {}
#         self.source_waypoints = source_waypoints
#         self.end_waypoints    = end_waypoints

#         # Subsetting the waypoints
#         if isinstance(source_waypoints,type(None)):
#             source_waypoints = list(self.config['Route_Info']['WayPoints']['Name'])

#         if isinstance(end_waypoints,type(None)):
#             self.end_waypoints = list(self.config['Route_Info']['WayPoints']['Name'])
#         else:
#             self.end_waypoints = end_waypoints

#         # Initialising the Dijkstra Info Dictionary
#         for wpt in source_waypoints:
#             self.dijkstra_info[wpt] = copy.copy(self.neighbour_graph)

#         for wpt in source_waypoints:
#             if verbose:
#                 print('=== Processing Waypoint =' + wpt + '===')
#             self._dijkstra(wpt)


#         start_time = time.time()
#         self.paths = self.dijkstra_path(source_waypoints,self.end_waypoints)
#         print("--- dijkstra_pathJSON -" + (time.time() - start_time) + "seconds ---")
#         if return_paths:
#             return self.paths

#     def compute_smoothed_paths(self,maxiter=10000,minimum_diff=1e-4,
#                                debugging=0,return_paths=True,verbose=False):
#         '''
#             Given a series of pathways smooth without centroid locations
#             using great circle smoothing
#         '''

#         smoothed_paths = []

#         if isinstance(self.paths,type(None)):
#             raise Exception('Paths not constructed, please re-run path construction')
#         paths = copy.deepcopy(self.paths)

#         for _, pth in enumerate(paths):
#             print('===Smoothing' + pth['from'] + 'to' + pth['to'] + '======')

#             smoothing_costing = self.smooth_cost_func(self.mesh,\
#                 self.dijkstra_info[pth['from']],self.config,maxiter=1,zerocurrents=True)
#             smoothing_costing.pathIter = maxiter

#             # -- Generating a dataframe of the case information -- 
#             points       = np.concatenate([pth['Path']['Points'][0,:][None,:],\
#                 pth['Path']['Points'][1:-1:2],pth['Path']['Points'][-1,:][None,:]])
#             cell_indices = np.concatenate([pth['Path']['CellIndices'][0][None],\
#                 pth['Path']['CellIndices'],pth['Path']['CellIndices'][-1][None]])
#             cell_dijk    = [smoothing_costing.DijkstraInfo.loc[ii] for ii in cell_indices]
#             smoothing_costing.CrossingDF  = pd.DataFrame({'cX':points[:,0],\
#                 'cY':points[:,1],'cellStart':cell_dijk[:-1],'cellEnd':cell_dijk[1:]})

#             # -- Determining the cases from the cell information. If within cell then case 0 --
#             cases = []
#             for _,row in smoothing_costing.CrossingDF.iterrows():
#                 try:
#                     cases.append(row.cellStart['case']\
#                         [np.where(np.array(row.cellStart['neighbourIndex'])==\
#                             row.cellEnd.name)[0][0]])
#                 except:
#                     cases.append(0)


#             smoothing_costing.CrossingDF['case'] = cases
#             smoothing_costing.CrossingDF.index =\ 
#                 np.arange(int(smoothing_costing.CrossingDF.index.min()),
#                 int(smoothing_costing.CrossingDF.index.max()*1e3 + 1e3),int(1e3))

#             smoothing_costing.orgDF = copy.deepcopy(smoothing_costing.CrossingDF)
#             iteration=0
#             #try:
#             while iteration < smoothing_costing.pathIter:

#                 smoothing_costing.previousDF = copy.deepcopy(smoothing_costing.CrossingDF)
#                 ids = 0
#                 while (id <= (len(smoothing_costing.CrossingDF) - 3)):
#                     smoothing_costing.triplet = smoothing_costing.CrossingDF.iloc[id:id+3]


#                     smoothing_costing._updateCrossingPoint()
#                     self.smoothing_costing = smoothing_costing

#                     # -- Horseshoe Case Detection --
#                     smoothing_costing._horseshoe()
#                     # -- Removing reseverse cases
#                     smoothing_costing._reverseCase()

#                     ids+=1+smoothing_costing.id

#                 self.smoothing_costing = smoothing_costing
#                 nc._mergePoint()
#                 self.smoothing_costing = smoothing_costing
#                 iteration+=1

#                 # Stop optimisation if the points are within some minimum difference
#                 if len(smoothing_costing.previousDF) == len(smoothing_costing.CrossingDF):
#                     Dist = np.max(np.sqrt((smoothing_costing.previousDF['cX'] -\
#                          smoothing_costing.CrossingDF['cX'])**2 +\
#                         (smoothing_costing.previousDF['cY'] -\
#                          smoothing_costing.CrossingDF['cY'])**2))
#                     if Dist < 1e-3:
#                         break
#             print('{} iterations'.format(iter))
#             # except:
#             #     print('Failed {}->{}'.format(Path['from'],Path['to']))

#             smoothed_path ={}
#             smoothed_path['from'] = pth['from']
#             smoothed_path['to']   = pth['to']
#             smoothed_path['Path'] = {}
#             smoothed_path['Path']['Points'] = nc.CrossingDF[['cX','cY']].to_numpy()    
#             smoothed_paths.append(smoothed_path)

#         self.smoothed_paths = smoothed_paths
#         if return_paths:
#             return self.smoothed_paths