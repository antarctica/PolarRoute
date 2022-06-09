'''
    FILL
'''
import numpy as np
import copy
import pandas as pd

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import time
import multiprocessing as mp

from RoutePlanner.crossing import NewtonianDistance, NewtonianCurve
from RoutePlanner.CellBox import CellBox

from shapely.geometry import Polygon, Point
from shapely import wkt
import geopandas as gpd
import ast
import json


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
    def __init__(self,config,cost_func=NewtonianDistance,verbose=False):
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
        #self.mesh    = copy.copy(mesh)
        self.config  = config


        self.source_waypoints = self.config['Route_Info']['Source_Waypoints']
        self.end_waypoints    = self.config['Route_Info']['End_Waypoints']

        # Creating a blank path construct
        self.paths          = None
        self.smoothed_paths = None
        self.dijkstra_info = {}
        self.neighbour_graph = pd.read_csv(self.config['Route_Info']['Mesh_Filename']).set_index('Index')
        self.neighbour_graph['geometry'] = self.neighbour_graph['geometry'].apply(wkt.loads)
        self.neighbour_graph = gpd.GeoDataFrame(self.neighbour_graph,crs='EPSG:4326', geometry='geometry')

        # Reformating the columns into corret type
        self.neighbour_graph['case'] = self.neighbour_graph['case'].apply(lambda x: ast.literal_eval(x))
        #self.neighbour_graph['cell_info'] = self.neighbour_graph['cell_info'].apply(lambda x: ast.literal_eval(x))
        self.neighbour_graph['neighbourIndex'] = self.neighbour_graph['neighbourIndex'].apply(lambda x: ast.literal_eval(x))
        #self.neighbour_graph['Vector'] = self.neighbour_graph['Vector'].apply(lambda x: ast.literal_eval(x))

        # ====== Speed Function Checking
        # Checking if Speed defined in file
        if 'Speed' not in self.neighbour_graph:
            self.neighbour_graph['Speed'] = self.config["Vehicle_Info"]["Speed"]

        # ===== Objective Function Information =====
        #  Checking if objective function is in the cellgrid
        print(self.config['Route_Info']['Objective_Function'])
        if (self.config['Route_Info']['Objective_Function'] != 'traveltime'):
            if (self.config['Route_Info']['Objective_Function'] not in self.neighbour_graph):
                    raise Exception("Objective Function require '{}' column in mesh dataframe".format(self.config['Route_Info']['Objective_Function']))

    
        # ===== Setting Up Dijkstra Graph =====
        self.neighbour_graph['positionLocked']          = False
        for vrbl in self.config['Route_Info']['Path_Variables']:
            self.neighbour_graph['shortest_{}'.format(vrbl)]    = np.inf
        self.neighbour_graph['neighbourTravelLegs']     = [list() for x in range(len(self.neighbour_graph.index))]
        self.neighbour_graph['neighbourCrossingPoints'] = [list() for x in range(len(self.neighbour_graph.index))]
        self.neighbour_graph['pathIndex']               = [list() for x in range(len(self.neighbour_graph.index))]
        for vrbl in self.config['Route_Info']['Path_Variables']:
            self.neighbour_graph['path_{}'.format(vrbl)] = [list() for x in range(len(self.neighbour_graph.index))]
        self.neighbour_graph['pathPoints']               = [list() for x in range(len(self.neighbour_graph.index))]

        # ====== Defining the cost function ======
        self.cost_func       = cost_func

        # ====== Outlining some constant values ======
        self.unit_shipspeed = self.config['Vehicle_Info']['Unit']
        self.unit_time      = self.config['Route_Info']['Time_Unit']
        self.zero_currents  = self.config['Route_Info']['Zero_Currents']
        self.variable_speed  =self. config['Route_Info']['Variable_Speed']


        # ====== Waypoints ======
        # Reading in Waypoints and casting into pandas if not already
        if not isinstance(self.config['Route_Info']['WayPoints'],pd.core.frame.DataFrame):
            self.config['Route_Info']['WayPoints'] = pd.read_csv(self.config['Route_Info']['WayPoints'])

        # Dropping waypoints outside domain
        self.config['Route_Info']['WayPoints'] = self.config['Route_Info']['WayPoints'][\
                                                              (self.config['Route_Info']['WayPoints']['Long'] >= self.config['Region']['longMin']) &\
                                                              (self.config['Route_Info']['WayPoints']['Long'] <=  self.config['Region']['longMax']) &\
                                                              (self.config['Route_Info']['WayPoints']['Lat'] <=  self.config['Region']['latMax']) &\
                                                              (self.config['Route_Info']['WayPoints']['Lat'] >=  self.config['Region']['latMin'])] 

        # # Initialising Waypoints positions and cell index
        wpts = self.config['Route_Info']['WayPoints']
        wpts['index'] = np.nan
        for idx,wpt in wpts.iterrows():
            indices = self.neighbour_graph[self.neighbour_graph['geometry'].contains(Point(wpt[['Long','Lat']]))].index
            if len(indices) > 1:
                raise Exception('Wapoint lies in multiple cell boxes. Please check mesh ! ')
            elif len(indices) == 0:
                continue
            else:
                if (self.neighbour_graph['Land'].loc[indices[0]] == True) or (self.neighbour_graph['Ice Area'].loc[indices[0]] > self.config['Vehicle_Info']['MaxIceExtent']):
                    continue
                wpts['index'].loc[idx] = int(indices[0])
        self.config['Route_Info']['WayPoints'] = wpts[~wpts['index'].isnull()]
        self.config['Route_Info']['WayPoints']['index'] = self.config['Route_Info']['WayPoints']['index'].astype(int)


        # ==== Printing Configuration and Information
        self.verbose = verbose
        if self.verbose:
            # JDS - Add in configuration print, read and write functions
            print(self.config)

    def dijkstra_paths(self,start_waypoints,end_waypoints):
        '''
            FILL
        '''

        geojson = {}
        geojson['type'] = "FeatureCollection"

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
                        graph = self.dijkstra_info[wpt_a_name]
                        path = {}
                        path['type'] = "Feature"
                        path['geometry'] = {}
                        path['geometry']['type'] = "LineString"
                        path_points = (np.array(wpt_a_loc+list(np.array(graph['pathPoints'].loc[wpt_b_index])[:-1,:])+wpt_b_loc))
                        path['geometry']['coordinates'] = path_points.tolist()

                        path['properties'] = {}
                        path['properties']['name'] = 'Route Path - {} to {}'.format(wpt_a_name,wpt_b_name)
                        path['properties']['from'] = '{}'.format(wpt_a_name)
                        path['properties']['to'] = '{}'.format(wpt_b_name)

                        cellIndices  = np.array(graph['pathIndex'].loc[wpt_b_index])
                        path_indices = np.array([cellIndices[0]] + list(np.repeat(cellIndices[1:-1],2)) + [cellIndices[-1]])
                        path['properties']['CellIndices'] = path_indices.tolist()

                        # Applying in-cell correction for travel-time
                        cost_func    = self.cost_func(source_graph=self.dijkstra_info[wpt_a_name].loc[path_indices[0]],
                                                    neighbour_graph=self.dijkstra_info[wpt_a_name].loc[path_indices[0]],
                                                    unit_shipspeed='km/hr',unit_time=self.unit_time,zerocurrents=self.zero_currents)
                        tt_start = cost_func.waypoint_correction(path_points[0,:],path_points[1,:])
                        cost_func    = self.cost_func(source_graph=self.dijkstra_info[wpt_a_name].loc[path_indices[-1]],
                                                    neighbour_graph=self.dijkstra_info[wpt_a_name].loc[path_indices[0]],
                                                    unit_shipspeed='km/hr',unit_time=self.unit_time,zerocurrents=self.zero_currents)
                        tt_end = cost_func.waypoint_correction(path_points[-1,:],path_points[-2,:])
                        path['properties']['traveltime'] = np.array(graph['path_traveltime'].loc[wpt_b_index])
                        path['properties']['traveltime'] = (path['properties']['traveltime'] - path['properties']['traveltime'][0]) + tt_start
                        path['properties']['traveltime'][-1] = (path['properties']['traveltime'][-2] + tt_end)

                        for vrbl in self.config['Route_Info']['Path_Variables']:
                            if vrbl == 'traveltime':
                                continue
                            path['properties'][vrbl] = np.cumsum(np.r_[path['properties']['traveltime'][0], np.diff(path['properties']['traveltime'])]*self.dijkstra_info[wpt_a_name].loc[path_indices,'{}'.format(vrbl)].to_numpy()).tolist()
                        path['properties']['traveltime'] = path['properties']['traveltime'].tolist()



                        paths.append(path)

                    except:
                        print('Failure to construct path from Dijkstra information')


        geojson['features'] = paths
        return geojson


    def save_paths(self):
        '''
        '''
        with open(self.config['Route_Info']['Paths_Filename'], 'w') as fp:
            json.dump(self.paths, fp)


    def objective_value(self,variable,source_graph,neighbour_graph,traveltime):
        if variable == 'traveltime':
            return np.array([source_graph['shortest_traveltime'] + traveltime[0],source_graph['shortest_traveltime'] + np.sum(traveltime)])
        else:
            return np.array([source_graph['shortest_{}'.format(variable)] +\
                    traveltime[0]*source_graph['{}'.format(variable)],
                    source_graph['shortest_{}'.format(variable)] +\
                    traveltime[0]*source_graph['{}'.format(variable)] +\
                    traveltime[1]*neighbour_graph['{}'.format(variable)]])


    def neighbour_cost(self,wpt_name,minimum_objective_index):
        '''
        Function for computing the shortest travel-time from a cell to its neighbours by applying the Newtonian method for optimisation
        
        Inputs:
        index - Index of the cell to process
        
        Output:

        Bugs/Alterations:
            - If corner of cell is land in adjacent cell then also return 'inf'
        '''
        # Determining the nearest neighbour index for the cell
        source_graph   = self.dijkstra_info[wpt_name].loc[minimum_objective_index]

        # Looping over idx
        for idx in range(len(source_graph['case'])):
            indx = source_graph['neighbourIndex'][idx]

            neighbour_graph = self.dijkstra_info[wpt_name].loc[indx]
            case = source_graph['case'][idx]

            # Applying Newton curve to determine crossing point
            cost_func    = self.cost_func(source_graph=source_graph,neighbour_graph=neighbour_graph,case=case,
                                      unit_shipspeed='km/hr',unit_time=self.unit_time,zerocurrents=self.zero_currents)
            # Updating the Dijkstra graph with the new information
            traveltime,crossing_points,cell_points = cost_func.value()


            source_graph['neighbourTravelLegs'].append(traveltime)
            source_graph['neighbourCrossingPoints'].append(np.array(crossing_points))

            # Using neighbourhood cost determine objective function value
            value = self.objective_value(self.config['Route_Info']['Objective_Function'],source_graph,neighbour_graph,traveltime)
            if value[1] < neighbour_graph['shortest_{}'.format(self.config['Route_Info']['Objective_Function'])]:
                for vrbl in self.config['Route_Info']['Path_Variables']:
                    value = self.objective_value(vrbl,source_graph,neighbour_graph,traveltime)
                    neighbour_graph['shortest_{}'.format(vrbl)] = value[1]
                    neighbour_graph['path_{}'.format(vrbl)]   = source_graph['path_{}'.format(vrbl)] + list(value)
                neighbour_graph['pathIndex']  = source_graph['pathIndex']  + [indx]
                neighbour_graph['pathPoints'] = source_graph['pathPoints'] +[list(crossing_points)] +[list(cell_points)]

                # neighbour_graph = pd.Series(
                #     {
                #     'cX':neighbour_graph['cX'],
                #     'cY':neighbour_graph['cY'],
                #     'case':neighbour_graph['case'],
                #     'neighbourIndex':neighbour_graph['neighbourIndex'],
                #     'positionLocked':neighbour_graph['positionLocked'],
                #     'traveltime': neighbour_cost[1],
                #     'neighbourTravelLegs':neighbour_graph['neighbourTravelLegs'],
                #     'neighbourCrossingPoints':neighbour_graph['neighbourCrossingPoints'],
                #     'pathIndex': source_graph['pathIndex']  + [indx],
                #     'path_traveltime':source_graph['path_traveltime']   + neighbour_cost,
                #     'pathPoints':source_graph['pathPoints'] +[list(crossing_points)] +[list(cell_points)]}
                #     )

                self.dijkstra_info[wpt_name].loc[indx] = neighbour_graph

        self.dijkstra_info[wpt_name].loc[minimum_objective_index] = source_graph


    def _dijkstra(self,wpt_name):


        # Including only the End Waypoints defined by the user
        wpts = self.config['Route_Info']['WayPoints'][self.config['Route_Info']['WayPoints']['Name'].isin(self.end_waypoints)]
        
        # Initalising zero traveltime at the source location
        source_index = int(self.config['Route_Info']['WayPoints'][self.config['Route_Info']['WayPoints']['Name'] == wpt_name]['index'])
        

        for vrbl in self.config['Route_Info']['Path_Variables']:
            self.dijkstra_info[wpt_name].loc[source_index,'shortest_{}'.format(vrbl)] = 0.0
        self.dijkstra_info[wpt_name].loc[source_index,'pathIndex'].append(source_index)
        
        # # Updating Dijkstra as long as all the waypoints are not visited or for full graph
        if self.config['Route_Info']['Early_Stopping_Criterion']:
            stopping_criterion_indices = wpts['index']
        else:
            stopping_criterion_indices = self.dijkstra_info[wpt_name].index

        while (self.dijkstra_info[wpt_name].loc[stopping_criterion_indices,'positionLocked'] == False).any():

            # Determining the index of the minimum objective function that has not been visited
            minimum_objective_index = self.dijkstra_info[wpt_name][self.dijkstra_info[wpt_name]['positionLocked']==False]['shortest_{}'.format(self.config['Route_Info']['Objective_Function'])].idxmin()
  
            # Finding the cost of the nearest neighbours from the source cell (Sc)
            self.neighbour_cost(wpt_name,minimum_objective_index)

            # Updating Position to be locked
            self.dijkstra_info[wpt_name].loc[minimum_objective_index,'positionLocked'] = True


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

        self.save_paths()
        # save paths
        

    def compute_smoothed_routes(self,maxiter=10000,minimumDiff=1e-4,debugging=0,return_paths=False,verbose=False):
        '''
            Given a series of pathways smooth without centroid locations using great circle smoothing
        '''

        SmoothedPaths = []
        geojson = {}
        geojson['type'] = "FeatureCollection"

        if type(self.paths) == type(None):
            raise Exception('Paths not constructed, please re-run path construction')
        Pths = copy.deepcopy(self.paths)['features']  

        for ii in range(len(Pths)):
            Path = Pths[ii]
            print('===Smoothing {}'.format(Path['properties']['name']))

            nc          = NewtonianCurve(self.dijkstra_info[Path['properties']['from']],self.config,maxiter=1,zerocurrents=True)
            nc.pathIter = maxiter

            org_path_points = np.array(Path['geometry']['coordinates'])
            org_cellindices = np.array(Path['properties']['CellIndices'])

            # -- Generating a dataframe of the case information -- 
            Points      = np.concatenate([org_path_points[0,:][None,:],org_path_points[1:-1:2],org_path_points[-1,:][None,:]])
            cellIndices = np.concatenate([[org_cellindices[0]],[org_cellindices[0]],org_cellindices[1:-1:2],[org_cellindices[-1]],[org_cellindices[-1]]])
            cellDijk    = [self.neighbour_graph.loc[ii] for ii in cellIndices]
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

                try:
                    nc._mergePoint()
                except:
                    self.nc = nc
                    iter+=1
                    continue
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


            TravelTime = nc.objective_function()

            SmoothedPath ={}
            SmoothedPath['type'] = 'Feature'
            SmoothedPath['geometry'] = {}
            SmoothedPath['geometry']['type'] = "LineString"
            SmoothedPath['geometry']['coordinates'] = nc.CrossingDF[['cX','cY']].to_numpy().tolist()            
            SmoothedPath['properties'] = {}
            SmoothedPath['properties']['from'] = Path['properties']['from']
            SmoothedPath['properties']['to']   = Path['properties']['to']
            SmoothedPath['properties']['traveltime'] = TravelTime
            SmoothedPaths.append(SmoothedPath)

        geojson['features'] = SmoothedPaths
        self.smoothed_paths = geojson
        if return_paths:
            return self.smoothed_paths


        with open(self.config['Route_Info']['Smoothpaths_Filename'], 'w') as fp:
            json.dump(self.smoothed_paths, fp)


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

        #     nc = NewtonianCurve(self.mesh,self.dijkstra_info[Path['from']],self.OptInfo,zerocurrents=self.zero_currents,debugging=debugging)
        #     nc.PathSmoothing(Points,cellIndices)

        #     Path['Path']['Points']       = nc.path
        #     SmoothedPaths.append(Path)

        # self.smoothed_paths = SmoothedPaths