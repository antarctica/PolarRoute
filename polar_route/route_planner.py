'''
    FILL
'''

import copy, json, ast, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely import wkt
from shapely.geometry.polygon import Point
import geopandas as gpd

from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from polar_route.crossing import NewtonianDistance, NewtonianCurve

def _flattenCases(id,mesh):
    neighbour_case = []
    neighbour_indx = []
    neighbours = mesh['neighbour_graph'][id]
    for case in neighbours.keys():
        for neighbour in neighbours[case]:
            neighbour_case.append(int(case))
            neighbour_indx.append(int(neighbour))
    return neighbour_case, neighbour_indx

class RoutePlanner:
    """The summary line for a class docstring should fit on one line.   
        ....
    """
    def __init__(self,mesh,cost_func=NewtonianDistance,verbose=False):
        """
        ...
        """

        # Load in the current cell structure & Optimisation Info̦
        #self.mesh    = copy.copy(mesh)
        self.mesh = json.loads(mesh) 
        self.config  = self.mesh['config']
        self.source_waypoints = self.config['Route_Info']['source_waypoints']
        self.end_waypoints    = self.config['Route_Info']['end_waypoints']

        # Creating a blank path construct
        self.paths          = None
        self.smoothed_paths = None
        self.dijkstra_info = {}

        self.verbose = verbose


        # ====== Loading Mesh & Neighbour Graph ======
        # Formating the Mesh and Neigbour Graph to the right form
        self.neighbour_graph = pd.DataFrame(self.mesh['cellboxes']).set_index('id')
        self.neighbour_graph['geometry'] = self.neighbour_graph['geometry'].apply(wkt.loads)
        self.neighbour_graph = gpd.GeoDataFrame(self.neighbour_graph,crs='EPSG:4326', geometry='geometry')
        self.neighbour_graph = self.neighbour_graph.loc[list(self.mesh['neighbour_graph'].keys())]

        # Using neighbour graph to determine neighbour indicies and cases
        self.neighbour_graph['case'], self.neighbour_graph['neighbourIndex'] = zip(*self.neighbour_graph.apply(lambda row : _flattenCases(row.name,self.mesh), axis = 1))

        self.neighbour_graph.index = self.neighbour_graph.index.astype(int)

        # Renaming the vector columns
        self.neighbour_graph = self.neighbour_graph.rename(columns={self.config['Route_Info']['vector_names'][0]: "Vector_x", 
                                                                    self.config['Route_Info']['vector_names'][1]: "Vector_y"})

        # ====== Speed Function Checking ======
        # Checking if Speed defined in file
        if 'Speed' not in self.neighbour_graph:
            self.neighbour_graph['Speed'] = self.config["Vessel"]["Speed"]


        # ====== Objective Function Information ======
        #  Checking if objective function is in the cellgrid            
        if (self.config['Route_Info']['objective_function'] != 'traveltime'):
            if (self.config['Route_Info']['objective_function'] not in self.neighbour_graph):
                    raise Exception("Objective Function require '{}' column in mesh dataframe".format(self.config['Route_Info']['objective_function']))

    
        # ===== Dijkstra Graph =====
        # Adding the required columns needed for the dijkstra graph
        self.neighbour_graph['positionLocked']          = False
        for vrbl in self.config['Route_Info']['path_variables']:
            self.neighbour_graph['shortest_{}'.format(vrbl)]    = np.inf
        self.neighbour_graph['neighbourTravelLegs']     = [list() for x in range(len(self.neighbour_graph.index))]
        self.neighbour_graph['neighbourCrossingPoints'] = [list() for x in range(len(self.neighbour_graph.index))]
        self.neighbour_graph['pathIndex']               = [list() for x in range(len(self.neighbour_graph.index))]
        for vrbl in self.config['Route_Info']['path_variables']:
            self.neighbour_graph['path_{}'.format(vrbl)] = [list() for x in range(len(self.neighbour_graph.index))]
        self.neighbour_graph['pathPoints']               = [list() for x in range(len(self.neighbour_graph.index))]

        # ====== Defining the cost function ======
        self.cost_func       = cost_func

        # ====== Outlining some constant values ======
        self.unit_shipspeed = self.config['Vessel']['Unit']
        self.unit_time      = self.config['Route_Info']['time_unit']
        self.zero_currents  = self.config['Route_Info']['zero_currents']
        self.variable_speed  =self.config['Route_Info']['variable_speed']

        if self.variable_speed == False:
            self.neighbour_graph['Speed'] = self.config["Vessel"]["Speed"]
        


        # ====== Waypoints ======
        self.mesh['waypoints'] = pd.read_csv(self.config['Route_Info']['waypoints_path'])

        # Dropping waypoints outside domain
        self.mesh['waypoints'] = self.mesh['waypoints'][\
                                                              (self.mesh['waypoints']['Long'] >= self.config['Mesh_info']['Region']['longMin']) &\
                                                              (self.mesh['waypoints']['Long'] <=  self.config['Mesh_info']['Region']['longMax']) &\
                                                              (self.mesh['waypoints']['Lat'] <=  self.config['Mesh_info']['Region']['latMax']) &\
                                                              (self.mesh['waypoints']['Lat'] >=  self.config['Mesh_info']['Region']['latMin'])] 

        # # Initialising Waypoints positions and cell index
        wpts = self.mesh['waypoints']
        wpts['index'] = np.nan
        for idx,wpt in wpts.iterrows():
            indices = self.neighbour_graph[self.neighbour_graph['geometry'].contains(Point(wpt[['Long','Lat']]))].index
            if len(indices) > 1:
                raise Exception('Wapoint lies in multiple cell boxes. Please check mesh ! ')
            elif len(indices) == 0:
                continue
            else:
                wpts['index'].loc[idx] = int(indices[0])
        self.mesh['waypoints'] = wpts[~wpts['index'].isnull()]
        self.mesh['waypoints']['index'] = self.mesh['waypoints']['index'].astype(int)
        self.mesh['waypoints'] =  self.mesh['waypoints'].to_json()


        # ==== Printing Configuration and Information

        if self.verbose:
            print('================================================')
            print('================================================')
            print('============= CONFIG INFORMATION  ============')
            print('================================================')
            print('================================================')
            print(' ---- Processing for Objective Function = {} ----'.format(self.config['Route_Info']['objective_function']))
            print(json.dumps(self.config, indent=4, sort_keys=True))



        self.mesh['waypoints'] =  pd.read_json(self.mesh['waypoints'])


    def to_json(self):
        '''
            Outputing the information in JSON format
        '''

        self.mesh['waypoints'] = self.mesh['waypoints'].to_dict()
        return json.dumps(self.mesh)


    def _dijkstra_paths(self,start_waypoints,end_waypoints):
        '''
            FILL
        '''

        geojson = {}
        geojson['type'] = "FeatureCollection"

        paths = []
        wpts_s = self.mesh['waypoints'][self.mesh['waypoints']['Name'].isin(start_waypoints)]
        wpts_e = self.mesh['waypoints'][self.mesh['waypoints']['Name'].isin(end_waypoints)]
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

                        for vrbl in self.config['Route_Info']['path_variables']:
                            if vrbl == 'traveltime':
                                continue
                            path['properties'][vrbl] = np.cumsum(np.r_[path['properties']['traveltime'][0], np.diff(path['properties']['traveltime'])]*self.dijkstra_info[wpt_a_name].loc[path_indices,'{}'.format(vrbl)].to_numpy()).tolist()
                        path['properties']['traveltime'] = path['properties']['traveltime'].tolist()



                        paths.append(path)

                    except:
                        print('Failure to construct path from Dijkstra information')


        geojson['features'] = paths
        return geojson


    def _objective_value(self,variable,source_graph,neighbour_graph,traveltime):
        '''
            FILL
        '''
        if variable == 'traveltime':
            return np.array([source_graph['shortest_traveltime'] + traveltime[0],source_graph['shortest_traveltime'] + np.sum(traveltime)])
        else:
            return np.array([source_graph['shortest_{}'.format(variable)] +\
                    traveltime[0]*source_graph['{}'.format(variable)],
                    source_graph['shortest_{}'.format(variable)] +\
                    traveltime[0]*source_graph['{}'.format(variable)] +\
                    traveltime[1]*neighbour_graph['{}'.format(variable)]])


    def _neighbour_cost(self,wpt_name,minimum_objective_index):
        '''
            FILL
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
            value = self._objective_value(self.config['Route_Info']['objective_function'],source_graph,neighbour_graph,traveltime)
            if value[1] < neighbour_graph['shortest_{}'.format(self.config['Route_Info']['objective_function'])]:
                for vrbl in self.config['Route_Info']['path_variables']:
                    value = self._objective_value(vrbl,source_graph,neighbour_graph,traveltime)
                    neighbour_graph['shortest_{}'.format(vrbl)] = value[1]
                    neighbour_graph['path_{}'.format(vrbl)]   = source_graph['path_{}'.format(vrbl)] + list(value)
                neighbour_graph['pathIndex']  = source_graph['pathIndex']  + [indx]
                neighbour_graph['pathPoints'] = source_graph['pathPoints'] +[list(crossing_points)] +[list(cell_points)]
                self.dijkstra_info[wpt_name].loc[indx] = neighbour_graph

        self.dijkstra_info[wpt_name].loc[minimum_objective_index] = source_graph


    def _dijkstra(self,wpt_name):
        '''
            FILL
        '''
        # Including only the End Waypoints defined by the user
        wpts = self.mesh['waypoints'][self.mesh['waypoints']['Name'].isin(self.end_waypoints)]
        
        # Initalising zero traveltime at the source location
        source_index = int(self.mesh['waypoints'][self.mesh['waypoints']['Name'] == wpt_name]['index'])
        

        for vrbl in self.config['Route_Info']['path_variables']:
            self.dijkstra_info[wpt_name].loc[source_index,'shortest_{}'.format(vrbl)] = 0.0
        self.dijkstra_info[wpt_name].loc[source_index,'pathIndex'].append(source_index)
        
        # # Updating Dijkstra as long as all the waypoints are not visited or for full graph
        if self.config['Route_Info']['early_stopping_criterion']:
            stopping_criterion_indices = wpts['index']
        else:
            stopping_criterion_indices = self.dijkstra_info[wpt_name].index

        while (self.dijkstra_info[wpt_name].loc[stopping_criterion_indices,'positionLocked'] == False).any():

            # Determining the index of the minimum objective function that has not been visited
            minimum_objective_index = self.dijkstra_info[wpt_name][self.dijkstra_info[wpt_name]['positionLocked']==False]['shortest_{}'.format(self.config['Route_Info']['objective_function'])].idxmin()
  
            # Finding the cost of the nearest neighbours from the source cell (Sc)
            self._neighbour_cost(wpt_name,minimum_objective_index)

            # Updating Position to be locked
            self.dijkstra_info[wpt_name].loc[minimum_objective_index,'positionLocked'] = True

    def compute_routes(self):
        '''
            FILL
        '''

        # Subsetting the waypoints
        if len(self.source_waypoints) == 0:
            self.source_waypoints = list(self.mesh['waypoints']['Name'])
        if len(self.end_waypoints) == 0:
            self.end_waypoints = list(self.mesh['waypoints']['Name'])

        # Removing end and source


        # Initialising the Dijkstra Info Dictionary
        for wpt in self.source_waypoints:
            self.dijkstra_info[wpt] = copy.copy(self.neighbour_graph)

        # 
        if self.verbose:
            print('================================================')
            print('================================================')
            print('============= Dijkstr Path Creation ============')
            print('================================================')
            print('================================================')

        for wpt in self.source_waypoints:
            if self.verbose:
                print('=== Processing Waypoint = {} ==='.format(wpt))
            # try:
            self._dijkstra(wpt)
            # except:
            #     continue

        # Using Dijkstra Graph compute path and meta information to all end_waypoints
        self.paths = self._dijkstra_paths(self.source_waypoints,self.end_waypoints)

        # if self.config['Route_Info']['save_dijkstra_graphs']:
        #     print('Currently not operational - Come back soon :) ')
        #     #JDS - Add in saving option for the full dijkstra graphs.
        # self._save_paths(self.config['Route_Info']['Paths_Filename'])
        self.mesh['paths'] = self.paths

        for ii in range(len(self.mesh['paths']['features'])):
            self.mesh['paths']['features'][ii]['properties']['times'] = [str(ii) for ii in (pd.to_datetime(self.mesh['config']['Mesh_info']['Region']['startTime']) + pd.to_timedelta(self.mesh['paths']['features'][ii]['properties']['traveltime'],unit='days'))]
    

    def compute_smoothed_routes(self):
            '''
                FILL
            '''
            maxiter     = self.config['Route_Info']['smooth_path']['max_iteration_number']
            minimumDiff = self.config['Route_Info']['smooth_path']['minimum_difference']

            # 
            SmoothedPaths = []
            geojson = {}
            geojson['type'] = "FeatureCollection"

            if type(self.paths) == type(None):
                raise Exception('Paths not constructed, please re-run path construction')
            Pths = copy.deepcopy(self.paths)['features']  

            if self.verbose:
                print('================================================')
                print('================================================')
                print('========= Determining Smoothed Paths ===========')
                print('================================================')
                print('================================================')




            for ii in range(len(Pths)):
                    Path = Pths[ii]
                    counter =0 

                    if self.verbose:
                        print('===Smoothing {}'.format(Path['properties']['name']))

                    nc          = NewtonianCurve(self.dijkstra_info[Path['properties']['from']],self.config,maxiter=1,zerocurrents=self.zero_currents)
                    nc.pathIter = maxiter

                    org_path_points = np.array(Path['geometry']['coordinates'])
                    org_cellindices = np.array(Path['properties']['CellIndices'])

                    # -- Generating a dataframe of the case information -- 
                    Points      = np.concatenate([org_path_points[0,:][None,:],org_path_points[1:-1:2],org_path_points[-1,:][None,:]])
                    cellIndices = np.concatenate([[org_cellindices[0]],[org_cellindices[0]],org_cellindices[1:-1:2],[org_cellindices[-1]],[org_cellindices[-1]]])
                    cellDijk    = [nc.neighbour_graph.loc[ii] for ii in cellIndices]
                    nc.CrossingDF  = pd.DataFrame({'cx':Points[:,0],'cy':Points[:,1],'cellStart':cellDijk[:-1],'cellEnd':cellDijk[1:]})

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

                    #while iter < nc.pathIter:
                    if self.verbose:
                        pbar = tqdm(np.arange(nc.pathIter))
                    else:
                        pbar = np.arange(nc.pathIter)



                    # Determining the computational time averaged across all pairs
                    self.allDist = []
                    self.allDist2 = []
                    for iter in pbar:
                        nc.previousDF = copy.deepcopy(nc.CrossingDF)
                        id = 0
                        while id <= (len(nc.CrossingDF) - 3):
                            try:
                                nc.triplet = nc.CrossingDF.iloc[id:id+3]
                        
                                nc._updateCrossingPoint()
                                self.nc = nc

                                # -- Horseshoe Case Detection -- 
                                nc._horseshoe()

                                # -- Removing reseverse cases                    
                                nc._reverseCase()
                            except:
                                break

                            id+=1+nc.id

                        # if  id <= (len(nc.CrossingDF) - 3):
                        #     print('Path Smoothing Failed!')
                        #     break


                        self.nc = nc

                        try:
                            nc._mergePoint()
                            self.nc = nc
                            iter+=1
                        except:
                            self.nc = nc
                            iter+=1
                            
                        # Stop optimisation if the points are within some minimum difference
                        if (len(nc.previousDF) == len(nc.CrossingDF)):
                            Dist = np.mean(np.sqrt((nc.previousDF['cx'].astype(float) - nc.CrossingDF['cx'].astype(float))**2 + (nc.previousDF['cy'].astype(float) - nc.CrossingDF['cy'].astype(float))**2))
                            self.allDist.append(Dist)
                            if self.verbose:
                                pbar.set_description("Mean Difference = {}".format(Dist))

                            if (Dist==np.min(self.allDist)) and len(np.where(abs(self.allDist - np.min(self.allDist)) < 1e-3)[0]) > 20:
                                if self.verbose:
                                    print('{} iterations - dDist={}  - Terminated from Horshoe repreating'.format(iter,Dist))
                                
                                break
                            if (Dist < minimumDiff) and (Dist!=0.0):
                                if self.verbose:
                                    print('{} iterations - dDist={}'.format(iter,Dist))
                                break
                        else:
                            if 'Dist' in locals():
                                self.allDist2.append(Dist)
                                if (np.sum((np.array(self.allDist2) - Dist)[-5:]) < 1e-6) and (iter>50) and len(self.allDist2)>20:
                                    print('{} iterations - dDist={}  - Terminated as value stagnated for more than 5 iterations'.format(iter,Dist))
                                    break



                    if self.verbose:
                        print('{} iterations'.format(iter))

                    # Determining the traveltime 

                    TravelTimeLegs,pathIndex = nc.objective_function()
                    FuelLegs = TravelTimeLegs*self.neighbour_graph['fuel'].loc[pathIndex]


                    SmoothedPath ={}
                    SmoothedPath['type'] = 'Feature'
                    SmoothedPath['geometry'] = {}
                    SmoothedPath['geometry']['type'] = "LineString"
                    SmoothedPath['geometry']['coordinates'] = nc.CrossingDF[['cx','cy']].to_numpy().tolist()            
                    SmoothedPath['properties'] = {}
                    SmoothedPath['properties']['from'] = Path['properties']['from']
                    SmoothedPath['properties']['to']   = Path['properties']['to']
                    SmoothedPath['properties']['traveltime'] = np.cumsum(TravelTimeLegs).tolist() 
                    SmoothedPath['properties']['fuel'] = np.cumsum(FuelLegs).tolist()
                    SmoothedPaths.append(SmoothedPath)

                    geojson['features'] = SmoothedPaths
                    self.smoothed_paths = geojson

                    self.mesh['paths'] = self.smoothed_paths

                    for ii in range(len(self.mesh['paths']['features'])):
                        self.mesh['paths']['features'][ii]['properties']['times'] = [str(ii) for ii in (pd.to_datetime(self.mesh['config']['Mesh_info']['Region']['startTime']) + pd.to_timedelta(self.mesh['paths']['features'][ii]['properties']['traveltime'],unit='days'))]

                    # with open(self.config['Route_Info']['Smoothpaths_Filename'], 'w') as fp:
                    #     json.dump(self.smoothed_paths, fp)
                    