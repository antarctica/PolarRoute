'''
    FILL IN
'''

# == Packages ==
from RoutePlanner.CrossingPoint import NewtonianCrossingPoint, NewtonianCrossingPointSmooth
from RoutePlanner.CellBox import CellBox

import numpy as np
import copy
import pandas as pd
import time

class TravelTime:
    '''
    FILL IN
    '''
    def __init__(self, mesh, config,neighbour_graph=None,
                 cost_func=NewtonianCrossingPoint, smooth_cost_func=NewtonianCrossingPointSmooth):
        start_time = time.time()
        # Load in the current cell structure & Optimisation Info
        self.mesh    = copy.copy(mesh)
        self.config  = config

        self.cost_func        = cost_func
        self.smooth_cost_func = smooth_cost_func

        self.unit_shipspeed  = self.config['Vehicle_Info']['Unit']
        self.unit_time       = self.config['Route_Info']['Time_Unit']
        self.zero_currents   = self.config['Route_Info']['Zero_Currents']
        self.variable_speed  = self.config['Route_Info']['Variable_Speed']

        # Creating a blank path construct
        self.paths         = None
        self.smoothed_paths = None

        self.test = 0

        # # Constructing Neighbour Graph
        if isinstance(neighbour_graph, type(None)):
            neighbour_graph = {}
            for idx,cell in enumerate(self.mesh.cellBoxes):
                if not isinstance(cell, CellBox):
                    continue
                else:
                    neigh     = self.mesh.neighbour_graph[idx]
                    cases     = []
                    neigh_indx = []
                    for case in neigh.keys():
                        indxs = neigh[case]
                        if len(indxs) == 0:
                            continue
                        for indx in indxs:
                            if (self.mesh.cellBoxes[indx].iceArea() >= self.config['Vehicle_Info']['MaxIceExtent']):
                                continue
                            if self.mesh._j_grid:
                                if self.mesh.cellBoxes[indx].isLandM():
                                    continue
                            else:
                                if self.mesh.cellBoxes[indx].containsLand():
                                    continue
                            cases.append(case)
                            neigh_indx.append(indx)
                    neigh_dict = {}
                    neigh_dict['cX']    = cell.cx
                    neigh_dict['cY']    = cell.cy
                    neigh_dict['case']  = cases
                    neigh_dict['neighbourIndex'] = neigh_indx 
                    neighbour_graph[idx] = neigh_dict
            self.neighbour_graph = pd.DataFrame().from_dict(neighbour_graph,orient='index')
        else:
            self.neighbour_graph = neighbour_graph
        self.neighbour_graph['positionLocked'] = False
        self.neighbour_graph['traveltime']     = np.inf
        self.neighbour_graph['neighbourTravelLegs'] = [list() for x in range(len(self.neighbour_graph.index))]
        self.neighbour_graph['neighbourCrossingPoints'] = [list() for x in range(len(self.neighbour_graph.index))]
        self.neighbour_graph['pathIndex'] = [list() for x in range(len(self.neighbour_graph.index))]
        self.neighbour_graph['pathCost'] = [list() for x in range(len(self.neighbour_graph.index))]
        self.neighbour_graph['pathPoints'] = [list() for x in range(len(self.neighbour_graph.index))]

   

        print('Zero Currets {}'.format(self.zero_currents))


        if not isinstance(self.config['Route_Info']['WayPoints'],pd.core.frame.DataFrame):
            self.config['Route_Info']['WayPoints'] = pd.read_csv(self.config['Route_Info']['WayPoints'])     

        # ====== Waypoints ======        
        # Dropping waypoints outside domain
        self.config['Route_Info']['WayPoints'] = self.config['Route_Info']['WayPoints'][(self.config['Route_Info']['WayPoints']['Long'] >= self.mesh._longMin) &\
                                                              (self.config['Route_Info']['WayPoints']['Long'] <= self.mesh._longMax) &\
                                                              (self.config['Route_Info']['WayPoints']['Lat'] <= self.mesh._latMax) &\
                                                              (self.config['Route_Info']['WayPoints']['Lat'] >= self.mesh._latMin)] 

        # Initialising Waypoints positions and cell index
        self.config['Route_Info']['WayPoints']['index'] = np.nan
        for idx,wpt in self.config['Route_Info']['WayPoints'].iterrows():
            long = wpt['Long']
            lat  = wpt['Lat']
            for index, cell in enumerate(self.mesh.cellBoxes):
                if isinstance(cell, CellBox):
                    if cell.containsPoint(lat,long):
                        break
            self.config['Route_Info']['WayPoints']['index'].loc[idx] = index

        print("--- Initial Setup - %s seconds ---" % (time.time() - start_time))


    def ice_resistance(self, Cell):
        """
                Function to find the ice resistance force at a given speed in a given cell.

                Inputs:
                Cell - Cell box object

                Outputs:
                r - Resistance force
        """
        hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}

        hull = self.config['Vehicle_Info']['HullType']
        beam = self.config['Vehicle_Info']['Beam']
        k, b, n = hull_params[hull]
        g = 9.81  # m/s-2

        V = self.config['Vehicle_Info']['Speed']*(5./18.)  # assume km/h and convert to m/s

        Fr = V/np.sqrt(g*Cell.iceArea()*Cell.iceThickness(self.config['Region']['startTime']))

        r = 0.5*k*(Fr**b)*Cell.iceDensity(self.config['Region']['startTime'])*beam*Cell.iceThickness(self.config['Region']['startTime'])\
            *(V**2)*(Cell.iceArea()**n)

        return r

    def inverse_resistance(self, force_limit, cell):
        """
        Function to find the fastest speed that keeps the ice resistance force below a given threshold.

        Inputs:
        Fl - Force limit
        Cell - Cell box object

        Outputs:
        v - speed
        """
        hull_params = {'slender': [4.4, -0.8267, 2.0], 'blunt': [16.1, -1.7937, 3]}

        hull = self.config['Vehicle_Info']['HullType']
        beam = self.config['Vehicle_Info']['Beam']
        k, b, n = hull_params[hull]
        g = 9.81  # m/s-2

        exp = 2.0 + b

        vexp = 2*force_limit/(k*cell.iceDensity(self.config['Region']['startTime'])*beam*cell.iceThickness(self.config['Region']['startTime'])*
                    (cell.iceArea()**n)*(g*cell.iceThickness(self.config['Region']['startTime'])*cell.iceArea())**-(b/2))

        vms = vexp**(1/exp)
        v = vms*(18./5.)  # convert from m/s to km/h

        return v       

    def speedFunction(self, Cell):
        if self.variable_speed:
            #s = (1-np.sqrt(Cell.iceArea()))*self.config['Vehicle_Info']['Speed']
            if Cell.iceArea() == 0.0:
                s = self.config['Vehicle_Info']['Speed']
            elif self.ice_resistance(Cell) < self.config['Vehicle_Info']['ForceLimit']:
                s = self.config['Vehicle_Info']['Speed']
            else:
                s = self.inverse_resistance(self.config['Vehicle_Info']['ForceLimit'], Cell)
        else:
            s = self.config['Vehicle_Info']['Speed']
        return s


    def Dijkstra2Path(self,StartWaypoints,EndWaypoints):
        Paths = []

        wpts_s = self.config['Route_Info']['WayPoints'][self.config['Route_Info']['WayPoints']['Name'].isin(StartWaypoints)]
        wpts_e = self.config['Route_Info']['WayPoints'][self.config['Route_Info']['WayPoints']['Name'].isin(EndWaypoints)]

        for idx,wpt_a in wpts_s.iterrows():
            wpt_a_name  = wpt_a['Name']
            wpt_a_index = int(wpt_a['index'])
            wpt_a_loc   = [[wpt_a['Long'],wpt_a['Lat']]]
            for idy,wpt_b in wpts_e.iterrows():
                wpt_b_name  = wpt_b['Name']
                wpt_b_index = int(wpt_b['index'])
                wpt_b_loc   = [[wpt_b['Long'],wpt_b['Lat']]]
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
        Sc          = self.mesh.cellBoxes[minimumTravelTimeIndex]
        SourceGraph = self.DijkstraInfo[wpt_name].loc[minimumTravelTimeIndex]

        # Looping over idx
        for idx in range(len(SourceGraph['case'])):
            indx = SourceGraph['neighbourIndex'][idx]
            
            # Don't inspect neighbour if position is already locked
            if self.DijkstraInfo[wpt_name].loc[indx,'positionLocked']:
                continue
            
            Nc   = self.mesh.cellBoxes[indx]
            Case = SourceGraph['case'][idx]
            

            # Set travel-time to infinite if neighbour is land or ice-thickness is too large.
            if (Nc.iceArea() >= self.config['Vehicle_Info']['MaxIceExtent']):
                SourceGraph['neighbourTravelLegs'].append([np.inf,np.inf])
                SourceGraph['neighbourCrossingPoints'].append([np.nan,np.nan])
                continue

            if self.mesh._j_grid:
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
            if self.test ==0:
                start_time = time.time()
            CostF    = self.cost_func(self.mesh,Sc=Sc,Nc=Nc,Sc_Speed=Sc_speed,Nc_Speed=Nc_speed,Case=Case,unit_shipspeed='km/hr',unit_time=self.unit_time,zerocurrents=self.zero_currents)
            # Updating the Dijkstra graph with the new information
            TravelTime,CrossPoints,CellPoints = CostF.value()
            if self.test ==0:
                print("--- NeighbourCost - cost_func - %s seconds ---" % (time.time() - start_time))

            if self.test ==0:
                start_time = time.time()

            SourceGraph['neighbourTravelLegs'].append(TravelTime)
            SourceGraph['neighbourCrossingPoints'].append(CrossPoints)

            # Updating the neighbour traveltime if its less the current global optimum
            Neighbour_cost  = [SourceGraph['traveltime'] + TravelTime[0],SourceGraph['traveltime'] + np.sum(TravelTime)]
            neighbour_graph  = self.DijkstraInfo[wpt_name].loc[indx]

            if self.test ==0:
                print("--- NeighbourCost - Updating Graph -indx=%s - Part1 - %s seconds ---" % (indx,time.time() - start_time))

            if self.test ==0:
                start_time = time.time()
            if Neighbour_cost[1] < neighbour_graph['traveltime']:
                # neighbour_graph['traveltime'] = Neighbour_cost[1]
                # neighbour_graph['pathIndex']  = SourceGraph['pathIndex']  + [indx]
                # neighbour_graph['pathCost']   = SourceGraph['pathCost']   + Neighbour_cost
                # neighbour_graph['pathPoints'] = SourceGraph['pathPoints'] + [list(CrossPoints)] + [list(CellPoints)]

                neighbour_graph = pd.Series(
                                {
                                'cX':neighbour_graph['cX'],
                                'cY':neighbour_graph['cY'],  
                                'case':neighbour_graph['case'],
                                'neighbourIndex':neighbour_graph['neighbourIndex'],
                                'positionLocked':neighbour_graph['positionLocked'],
                                'traveltime': Neighbour_cost[1],
                                'neighbourTravelLegs':neighbour_graph['neighbourTravelLegs'],
                                'neighbourCrossingPoints':neighbour_graph['neighbourCrossingPoints'],
                                'pathIndex': SourceGraph['pathIndex']  + [indx],
                                'pathCost':SourceGraph['pathCost']   + Neighbour_cost,
                                'pathPoints':SourceGraph['pathPoints'] + [list(CrossPoints)] + [list(CellPoints)]}
                                )




            self.DijkstraInfo[wpt_name].loc[indx] = neighbour_graph
            if self.test ==0:
                print("--- NeighbourCost - Updating Graph - Part2 - %s seconds ---" % (time.time() - start_time))



        self.DijkstraInfo[wpt_name].loc[minimumTravelTimeIndex] = SourceGraph


    def _dijkstra(self,wpt_name):
        start_time =time.time()
        # Including only the End Waypoints defined by the user
        Wpts = self.config['Route_Info']['WayPoints'][self.config['Route_Info']['WayPoints']['Name'].isin(self.end_waypoints)]
        
        # Initalising zero traveltime at the source location
        SourceIndex = int(self.config['Route_Info']['WayPoints'][self.config['Route_Info']['WayPoints']['Name'] == wpt_name]['index'])
        self.DijkstraInfo[wpt_name].loc[SourceIndex,'traveltime'] = 0.0
        self.DijkstraInfo[wpt_name].loc[SourceIndex,'pathIndex'].append(SourceIndex)
        
        print("--- Dijkstr Run Preface - %s seconds ---" % (time.time() - start_time))

        while (self.DijkstraInfo[wpt_name].loc[Wpts['index'],'positionLocked'] == False).any():
        #while (self.DijkstraInfo[wpt_name]['positionLocked'] == False).any():   

            #
            if self.test == 0:
                start_time = time.time() 
            # Determining the index of the minimum traveltime that has not been visited
            minimumTravelTimeIndex = self.DijkstraInfo[wpt_name][self.DijkstraInfo[wpt_name]['positionLocked']==False]['traveltime'].idxmin()
            if self.test == 0:
                print("--- Dijkstr Index of Minimum traveltime - %s seconds ---" % (time.time() - start_time))


            if self.test == 0:
                start_time = time.time() 
            # Finding the cost of the nearest neighbours from the source cell (Sc)
            self.NeighbourCost(wpt_name,minimumTravelTimeIndex)
            if self.test == 0:
                print("--- Dijkstr Index of NeighbourCost - %s seconds ---" % (time.time() - start_time))
                self.test+=1


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
            self.DijkstraInfo[wpt] = copy.copy(self.neighbour_graph)


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


        start_time = time.time()
        self.paths = self.Dijkstra2Path(source_waypoints,self.end_waypoints)
        print("--- Dijkstra2PathJSON - %s seconds ---" % (time.time() - start_time))
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

            nc = self.smooth_cost_func(self.mesh,self.DijkstraInfo[Path['from']],self.config,maxiter=1,zerocurrents=True)
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

        

        self.smoothed_paths = SmoothedPaths
        if return_paths:
            return self.smoothed_paths
