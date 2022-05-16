'''
    FILL
'''
import numpy as np
import copy
import pandas as pd

import time
import multiprocessing as mp

from RoutePlanner.crossing import NewtonianDistance, NewtonianCurve
from RoutePlanner.CellBox import CellBox


class TravelTime:
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self,mesh,config,neighbour_graph=None,cost_func=NewtonianDistance,verbose=False):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.

        """

        # Load in the current cell structure & Optimisation Info̦
        self.mesh    = copy.copy(mesh)
        self.config  = config


        self.source_waypoints = self.config['Route_Info']['Source_Waypoints']
        self.end_waypoints    = self.config['Route_Info']['End_Waypoints']

        # Creating a blank path construct
        self.paths          = None
        self.smoothed_paths = None
        self.dijkstra_info = {}
        # # Constructing Neighbour Graph
        if isinstance(neighbour_graph,type(None)):
            neighbour_graph = {}
            for idx,cell in enumerate(self.mesh.cellBoxes):
                if not isinstance(cell, CellBox):
                    continue
                else:
                    neigh      = self.mesh.neighbourGraph[idx]
                    cases      = []
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
        self.neighbour_graph['pathIndex']  = [list() for x in range(len(self.neighbour_graph.index))]
        self.neighbour_graph['pathCost']   = [list() for x in range(len(self.neighbour_graph.index))]
        self.neighbour_graph['pathPoints']   = [list() for x in range(len(self.neighbour_graph.index))]

        self.cost_func       = cost_func

        self.unit_shipspeed = self.config['Vehicle_Info']['Unit']
        self.unit_time      = self.config['Route_Info']['Time_Unit']
        self.zero_currents  = self.config['Route_Info']['Zero_Currents']
        self.variable_speed  =self. config['Route_Info']['Variable_Speed']

        # ====== Waypoints ======
        # Reading in Waypoints and casting into pandas if not already
        if not isinstance(self.config['Route_Info']['WayPoints'],pd.core.frame.DataFrame):
            self.config['Route_Info']['WayPoints'] = pd.read_csv(self.config['Route_Info']['WayPoints'])

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


            if self.mesh.cellBoxes[index].iceArea() >= self.config['Vehicle_Info']['MaxIceExtent']:
                continue
            if self.mesh._j_grid:
                if self.mesh.cellBoxes[index].isLandM():
                    continue
            else:
                if self.mesh.cellBoxes[index].containsLand():
                    continue
            self.config['Route_Info']['WayPoints']['index'].loc[idx] = index

        self.config['Route_Info']['WayPoints'] = self.config['Route_Info']['WayPoints'][~ self.config['Route_Info']['WayPoints']['index'].isnull()]


        # ==== Printing Configuration and Information
        self.verbose = verbose
        if self.verbose:
            # JDS - Add in configuration print, read and write functions
            print(self.config)


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

    def speed(self, cell):
        '''
            FILL
        '''
        if self.variable_speed:
            if cell.iceArea() == 0.0:
                speed = self.config['Vehicle_Info']['Speed']
            elif self.ice_resistance(cell) < self.config['Vehicle_Info']['ForceLimit']:
                speed = self.config['Vehicle_Info']['Speed']
            else:
                speed = self.inverse_resistance(self.config['Vehicle_Info']['ForceLimit'], cell)
        else:
            speed = self.config['Vehicle_Info']['Speed']
        return speed


    def dijkstra_paths(self,start_waypoints,end_waypoints):
        '''
            FILL
        '''
        paths = []

        wpts_s = self.config['Route_Info']['WayPoints'][self.config['Route_Info']['WayPoints']['Name'].isin(start_waypoints)]
        wpts_e = self.config['Route_Info']['WayPoints'][self.config['Route_Info']['WayPoints']['Name'].isin(end_waypoints)]

        for _,wpt_a in wpts_s.iterrows():
            wpt_a_name  = wpt_a['Name']
            wpt_a_index = int(wpt_a['index'])
            wpt_a_loc   = [[wpt_a['Long'],wpt_a['Lat']]]
            for _,wpt_b in wpts_e.iterrows():
                wpt_b_name  = wpt_b['Name']
                wpt_b_index = int(wpt_b['index'])
                wpt_b_loc   = [[wpt_b['Long'],wpt_b['Lat']]]
                if not wpt_a_name == wpt_b_name:
                    try:
                        # ==== Correcting Path for waypoints off cell
                        path = {}
                        path['from']               = wpt_a_name
                        path['to']                 = wpt_b_name

                        graph = self.dijkstra_info[wpt_a_name]
                        path['Time']               = float(graph['traveltime'].loc[wpt_b_index])
                        if path['Time'] == np.inf:
                            continue
                        path_traveltime = graph['pathCost'].loc[wpt_b_index]

                        # ===== Appending Path =====
                        path['Path']                = {}
                        path['Path']['Points']      = (np.array(wpt_a_loc+list(np.array(graph['pathPoints'].loc[wpt_b_index])[:-1,:])+wpt_b_loc)).tolist()
                        path['Path']['CellIndices'] = (np.array(graph['pathIndex'].loc[wpt_b_index])).tolist()
                        #path['Path']['CaseTypes']   = np.array([wpt_a_index] + graph['pathPoints'].loc[wpt_b_index] + [wpt_b_index],dtype=object)
                        path['Path']['Time']        = path_traveltime
                        paths.append(path)
                    except IOError:
                        print('Failure to construct path from Dijkstra information')
        return paths

    def neighbour_cost(self,wpt_name,minimum_traveltime_index):
        '''
        Function for computing the shortest travel-time from a cell to its neighbours by applying the Newtonian method for optimisation
        
        Inputs:
        index - Index of the cell to process
        
        Output:

        Bugs/Alterations:
            - If corner of cell is land in adjacent cell then also return 'inf'
        '''
        # Determining the nearest neighbour index for the cell
        source_cellbox = self.mesh.cellBoxes[minimum_traveltime_index]
        source_graph   = self.dijkstra_info[wpt_name].loc[minimum_traveltime_index]

        # Looping over idx
        for idx in range(len(source_graph['case'])):
            indx = source_graph['neighbourIndex'][idx]

            # Don't inspect neighbour if position is already locked
            if self.dijkstra_info[wpt_name].loc[indx,'positionLocked']:
                continue

            neighbour_cellbox = self.mesh.cellBoxes[indx]
            case = source_graph['case'][idx]


            source_speed    = self.speed(source_cellbox)
            neighbour_speed = self.speed(neighbour_cellbox)

            # Applying Newton curve to determine crossing point
            cost_func    = self.cost_func(self.mesh,source_cell=source_cellbox,neighbour_cell=neighbour_cellbox,
                                      source_speed=source_speed,neighbour_speed=neighbour_speed,case=case,
                                      unit_shipspeed='km/hr',unit_time=self.unit_time,zerocurrents=self.zero_currents)
            # Updating the Dijkstra graph with the new information
            traveltime,crossing_points,cell_points = cost_func.value()

            
            source_graph['neighbourTravelLegs'].append(traveltime)
            source_graph['neighbourCrossingPoints'].append(crossing_points)


            # Updating the neighbour traveltime if its less the current global optimum
            
            neighbour_cost   = [source_graph['traveltime'] + traveltime[0],source_graph['traveltime'] + np.sum(traveltime)]
            neighbour_graph  = self.dijkstra_info[wpt_name].loc[indx]

            if neighbour_cost[1] < neighbour_graph['traveltime']:
                neighbour_graph = pd.Series(
                    {
                    'cX':neighbour_graph['cX'],
                    'cY':neighbour_graph['cY'],
                    'case':neighbour_graph['case'],
                    'neighbourIndex':neighbour_graph['neighbourIndex'],
                    'positionLocked':neighbour_graph['positionLocked'],
                    'traveltime': neighbour_cost[1],
                    'neighbourTravelLegs':neighbour_graph['neighbourTravelLegs'],
                    'neighbourCrossingPoints':neighbour_graph['neighbourCrossingPoints'],
                    'pathIndex': source_graph['pathIndex']  + [indx],
                    'pathCost':source_graph['pathCost']   + neighbour_cost,
                    'pathPoints':source_graph['pathPoints'] +[list(crossing_points)] +[list(cell_points)]}
                    )

                self.dijkstra_info[wpt_name].loc[indx] = neighbour_graph

        self.dijkstra_info[wpt_name].loc[minimum_traveltime_index] = source_graph




    def _dijkstra(self,wpt_name):


        # Including only the End Waypoints defined by the user
        wpts = self.config['Route_Info']['WayPoints'][self.config['Route_Info']['WayPoints']['Name'].isin(self.end_waypoints)]
        
        # Initalising zero traveltime at the source location
        source_index = int(self.config['Route_Info']['WayPoints'][self.config['Route_Info']['WayPoints']['Name'] == wpt_name]['index'])
        self.dijkstra_info[wpt_name].loc[source_index,'traveltime'] = 0.0
        self.dijkstra_info[wpt_name].loc[source_index,'pathIndex'].append(source_index)
        
        # # Updating Dijkstra as long as all the waypoints are not visited or for full graph
        if self.config['Route_Info']['Early_Stopping_Criterion']:
            stopping_criterion_indices = wpts['index']
        else:
            stopping_criterion_indices = self.dijkstra_info[wpt_name].index


        while (self.dijkstra_info[wpt_name].loc[stopping_criterion_indices,'positionLocked'] == False).any():

            # Determining the index of the minimum traveltime that has not been visited
            minimum_traveltime_index = self.dijkstra_info[wpt_name][self.dijkstra_info[wpt_name]['positionLocked']==False]['traveltime'].idxmin()
  
            # Finding the cost of the nearest neighbours from the source cell (Sc)
            self.neighbour_cost(wpt_name,minimum_traveltime_index)

            # Updating Position to be locked
            self.dijkstra_info[wpt_name].loc[minimum_traveltime_index,'positionLocked'] = True
    
          

        # Correct travel-time off grid for start and end indices
        # ----> TO-DO

    def compute_routes(self):
        '''
            FILL
        '''

        # Subsetting the waypoints
        if len(self.source_waypoints) == 0:
            self.source_waypoints = list(self.config['Route_Info']['WayPoints']['Name'])
        if len(self.end_waypoints) == 0:
            self.end_waypoints = list(self.config['Route_Info']['WayPoints']['Name'])

        # Removing end and source


        # Initialising the Dijkstra Info Dictionary
        for wpt in self.source_waypoints:
            self.dijkstra_info[wpt] = copy.copy(self.neighbour_graph)


        # if multiprocessing:

        #     pool = mp.Pool(mp.cpu_count())
        #     [pool.apply(self._dijkstra, args=source) for source in source_waypoints]
        #     pool.close()
        #     pool_obj = multiprocessing.Pool()
        #     answer = pool_obj.map(self._dijkstra,source_waypoints)

        for wpt in self.source_waypoints:
            if self.verbose:
                print('=== Processing Waypoint = {} ==='.format(wpt))
            self._dijkstra(wpt)

        # Using Dijkstra Graph compute path and meta information to all end_waypoints
        self.paths = self.dijkstra_paths(self.source_waypoints,self.end_waypoints)

        # if self.config['Route_Info']['Save_Dijkstra_Graphs']:
        #     print('Currently not operational - Come back soon :) ')
        #     #JDS - Add in saving option for the full dijkstra graphs.

    def compute_smoothed_routes(self,maxiter=10000,minimumDiff=1e-4,debugging=0,return_paths=True,verbose=False):
        '''
            Given a series of pathways smooth without centroid locations using great circle smoothing
        '''

        SmoothedPaths = []

        if type(self.paths) == type(None):
            raise Exception('Paths not constructed, please re-run path construction')
        Pths = copy.deepcopy(self.paths)

        for ii in range(len(Pths)):
            Path = Pths[ii]
            print('===Smoothing {} to {} ======'.format(Path['from'],Path['to']))

            nc = NewtonianCurve(self.mesh,self.dijkstra_info[Path['from']],self.config,maxiter=1,zerocurrents=True)
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

            nc = NewtonianCurve(self.mesh,self.dijkstra_info[Path['from']],self.OptInfo,zerocurrents=self.zero_currents,debugging=debugging)
            nc.PathSmoothing(Points,cellIndices)

            Path['Path']['Points']       = nc.path
            SmoothedPaths.append(Path)

        self.smoothed_paths = SmoothedPaths