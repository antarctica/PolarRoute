'''
    This section of the codebase is used for construction of route paths using the 
    environmental mesh between a series of user defined waypoints
'''

import copy, json, ast, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely import wkt
from shapely.geometry.polygon import Point
import geopandas as gpd
import logging

from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from polar_route.crossing import NewtonianDistance
from polar_route.crossing_smoothing import Smoothing,PathValues,find_edge
from polar_route.config_validation.config_validator import validate_route_config
from polar_route.config_validation.config_validator import validate_waypoints

def _flattenCases(id,mesh):
    neighbour_case = []
    neighbour_indx = []
    neighbours = mesh['neighbour_graph'][id]
    for case in neighbours.keys():
        for neighbour in neighbours[case]:
            neighbour_case.append(int(case))
            neighbour_indx.append(int(neighbour))
    return neighbour_case, neighbour_indx

def _initialise_dijkstra_graph(dijkstra_graph):
    '''
        Initialising dijkstra graph information in a standard form

        Args:
            dijkstra_graph (pd.dataframe) - Pandas dataframe of the dijkstra graph construction

        Outputs:
            dijkstra_graph_dict (dict) - Dictionary comprising dijkstra graph with keys based on cellbox id. 
                                         Each entry is a dictionary of the cellbox environmental and dijkstra information. 


    '''

    dijkstra_graph_dict = {}
    for idx,cell in dijkstra_graph.iterrows():
        dijkstra_graph_dict[cell.name] = {}
        dijkstra_graph_dict[cell.name]['id'] = cell.name
        for key in cell.keys():
            entry = cell[key]
            if type(entry) == list:
                entry = np.array(entry)
            dijkstra_graph_dict[cell.name][key] = entry
    return  dijkstra_graph_dict


def _initialise_dijkstra_route(dijkstra_graph,dijkstra_route):
    '''
        Initialising dijkstra route info a standard path form

        Args:
            dijkstra_graph (dict) - Dictionary comprising dijkstra graph with keys based on cellbox id.
                                    Each entry is a dictionary of the cellbox environmental and dijkstra information.

            dijkstra_route (dict) - Dictionary of a GeoJSON entry for the dijkstra route

        Outputs:
            aps (list, [find_edge, ...]) - A list of adjacent cell pairs where each entry is of type find_edge including information on
                                        .crossing, .case, .start, and .end see 'find_edge' for more information
    '''

    org_path_points = np.array(dijkstra_route['geometry']['coordinates'])
    org_cellindices = np.array(dijkstra_route['properties']['CellIndices'])
    org_cellcases= np.array(dijkstra_route['properties']['cases'])

    # -- Generating a dataframe of the case information -- 
    Points      = np.concatenate([org_path_points[0,:][None,:],org_path_points[1:-1:2],org_path_points[-1,:][None,:]])
    cellIndices = np.concatenate([[org_cellindices[0]],[org_cellindices[0]],org_cellindices[1:-1:2],[org_cellindices[-1]],[org_cellindices[-1]]])
    cellcases = np.concatenate([[org_cellcases[0]],[org_cellcases[0]],org_cellcases[1:-1:2],[org_cellcases[-1]],[org_cellcases[-1]]])

    cellDijk    = [dijkstra_graph[ii] for ii in cellIndices]
    cells  = cellDijk[1:-1]
    cases  = cellcases[1:-1]
    aps = []
    for ii in range(len(cells)-1):
        aps += [find_edge(cells[ii],cells[ii+1],cases[ii+1])]

    # #-- Setting some backend information
    start_waypoint = Points[0,:]
    end_waypoint   = Points[-1,:]

    return aps, start_waypoint,end_waypoint

def _json_str(input):
    '''
        Load JSON object either from dict or from file

        Input:
            input (dict or string) - JSON file/dict 
    
        Output:
            output (dict) - Dictionary from JSON object
    '''
    if type(input) is dict:
        output = input
    elif type(input) is str:
        try:
            with open(input, 'r') as f:
                output = json.load(f)
        except:
            raise Exception("Unable to load '{}', please check path name".format(input))
    return output

def _pandas_dataframe_str(input):
    if (type(input) is dict) or (type(input) is pd.core.frame.DataFrame):
        output = input
    elif type(input) is str:
        try:
            output = pd.read_csv(input)
        except:
            raise Exception("Unable to load '{}', please check path name".format(input))
    return output





class RoutePlanner:
    """
        ---

        RoutePlanner optimises the route paths between a series of waypoints. 
        The routes are constructed in a two stage process:

        compute_routes: uses a mesh based Dijkstra method to determine the optimal routes 
                        between a series of waypoint.

        compute_smoothed_routes: smooths the compute_routes using information from the environmental mesh
                                to determine mesh independent optimal route paths


        ---

        Attributes:
            waypoints (dict): A dictionary of the waypoints supplied by the user in the region
                of interest. The waypoints are of the form:

               {\n
                    "Name":{\n
                        '0':"Falklands",\n
                        '1':"Rothera",\n
                        ...\n
                    },\n
                    "Lat":{\n
                        '0':-52.6347222222,
                        '1':-75.26722,\n
                        ...\n
                    },\n
                    "Long":{\n
                        ...\n
                    },\n
                    "index":{\n
                        ...\n
                    }\n
               }



            paths (geojson): A GeoJSON of all paths constructed. The paths are in the form:

                {\n
                    'types':'FeatureCollection',\n
                    "features":{[\n
                        'type':'feature',\n
                        'geometry':{\n
                            'type': 'LineString',

                            'coordinates': [[-27.21694, -75.26722],\n
                                            [-27.5, -75.07960297382266],\n
                                            [-27.619238882768894, -75.0],\n
                                            ...]\n
                        },
                        'properties':{\n
                            'from': 'Halley',\n
                            'to': 'Rothera',\n
                            'traveltime': [0.0,\n
                                        0.03531938671648596,\n
                                        0.050310986633880575,\n
                                        ...],\n
                            'fuel': [0.0,\n
                                    0.9648858923588642,\n
                                    1.3745886107069096,\n
                                    ...],\n
                            'times': ['2017-01-01 00:00:00',
                                    '2017-01-01 00:50:51.595036800',
                                    '2017-01-01 01:12:26.869276800',
                                    ...]\n
                        }\n
                    ]}\n
                }\n

        ---

    """

    def __init__(self, mesh, config, waypoints, cost_func=NewtonianDistance):

        """
            Constructs the routes from information given in the config file.

            Args:

                mesh (dict or string of filepath): mesh based JSON containing the cellbox information and neighbourhood graph

                config (dict or string of filepath): config JSON which defines the attributes required for the route construction. 
                    Sections required for the route construction are as follows\n
                    \n
                    {\n
                        "objective_function": (string) currently either 'traveltime' or 'fuel',\n
                        "path_variables": list of (string),\n
                        "waypoints_path": (string),\n
                        "source_waypoints": list of (string),\n
                        "end_waypoints": list of (string),\n
                        "vector_names": list of (string),\n
                        "zero_currents": (boolean),\n
                        "variable_speed" (boolean),\n
                        "time_unit" (string),\n
                        "early_stopping_criterion" (boolean),\n
                        "save_dijkstra_graphs": (boolean),\n
                        "smooth_path":{\n
                            "max_iteration_number":(int),\n
                            "minimum_difference":(float),\n
                         }\n
                    }\n

                cost_func (func): Crossing point cost function for Dijkstra Path creation. For development purposes only!
        """
        validate_route_config(config)
        validate_waypoints(waypoints)
        # Load in the current cell structure & Optimisation Info̦
        self.mesh             = _json_str(mesh)
        self.config           = _json_str(config)
        waypoints_df          = _pandas_dataframe_str(waypoints)
        source_waypoints_df   = waypoints_df[waypoints_df['Source'] == "X"]
        des_waypoints_df      = waypoints_df[waypoints_df['Destination'] == "X"]

        self.source_waypoints = list(source_waypoints_df['Name'])
        self.end_waypoints    = list(des_waypoints_df['Name'])

        # Case indices
        self.indx_type = np.array([1, 2, 3, 4, -1, -2, -3, -4])

        # Creating a blank path construct
        self.paths          = None
        self.smoothed_paths = None
        self.dijkstra_info = {}

        # ====== Loading Mesh & Neighbour Graph ======
        # Formatting the Mesh and Neighbour Graph to the right form
        self.neighbour_graph = pd.DataFrame(self.mesh['cellboxes']).set_index('id')
        self.neighbour_graph['geometry'] = self.neighbour_graph['geometry'].apply(wkt.loads)
        self.neighbour_graph = gpd.GeoDataFrame(self.neighbour_graph, crs='EPSG:4326', geometry='geometry')

        # Removing any point not in accessible positions
        self.neighbour_graph = self.neighbour_graph.loc[list(self.mesh['neighbour_graph'].keys())]

        # Using neighbour graph to determine neighbour indices and cases
        self.neighbour_graph['case'], self.neighbour_graph['neighbourIndex'] = zip(*self.neighbour_graph.apply(lambda row: _flattenCases(row.name, self.mesh), axis=1))

        self.neighbour_graph.index = self.neighbour_graph.index.astype(int)

        # Renaming the vector columns
        self.neighbour_graph = self.neighbour_graph.rename(columns={self.config['vector_names'][0]: "Vector_x", 
                                                                    self.config['vector_names'][1]: "Vector_y"})

        # ====== Speed Function Checking ======
        # Checking if Speed defined in file
        if 'speed' not in self.neighbour_graph:
            print(self.neighbour_graph.columns)
            raise Exception('Vessel Speed not in the mesh information ! Please run vessel performance')
        
        # ======= Sea-Ice Concentration ======
        if 'SIC' not in self.neighbour_graph:
            self.neighbour_graph['SIC'] = 0.0

        # ====== Objective Function Information ======
        #  Checking if objective function is in the dijkstra            
        if self.config['objective_function'] != 'traveltime':
            if self.config['objective_function'] not in self.neighbour_graph:
                raise Exception("Objective Function require '{}' column in mesh dataframe".format(self.config['objective_function']))

        # ===== Dijkstra Graph =====
        # Adding the required columns needed for the dijkstra graph
        self.neighbour_graph['positionLocked']          = False
        for vrbl in self.config['path_variables']:
            self.neighbour_graph['shortest_{}'.format(vrbl)]    = np.inf
        self.neighbour_graph['neighbourTravelLegs']     = [list() for x in range(len(self.neighbour_graph.index))]
        self.neighbour_graph['neighbourCrossingPoints'] = [list() for x in range(len(self.neighbour_graph.index))]
        self.neighbour_graph['pathIndex']               = [list() for x in range(len(self.neighbour_graph.index))]
        for vrbl in self.config['path_variables']:
            self.neighbour_graph['path_{}'.format(vrbl)] = [list() for x in range(len(self.neighbour_graph.index))]
        self.neighbour_graph['pathPoints']               = [list() for x in range(len(self.neighbour_graph.index))]

        # ====== Defining the cost function ======
        self.cost_func       = cost_func

        # ====== Outlining some constant values ======
        self.unit_time       = self.config['time_unit']
        self.zero_currents   = self.config['zero_currents']
        self.variable_speed  = self.config['variable_speed']
        if type(self.variable_speed) == float:
            logging.info(' Defining a constant speed map = {}'.format(self.variable_speed))
            self.neighbour_graph['speed'] = self.variable_speed
            cbxs = pd.DataFrame(self.mesh['cellboxes'])
            cbxs['speed'] = self.variable_speed
            self.mesh['cellboxes'] = cbxs.to_dict('records')

        # ====== Waypoints ======
        self.mesh['waypoints'] = waypoints_df

        # Initialising Waypoints positions and cell index
        wpts = self.mesh['waypoints']
        wpts['index'] = np.nan
        for idx,wpt in wpts.iterrows():
            indices = self.neighbour_graph[self.neighbour_graph['geometry'].contains(Point(wpt[['Long','Lat']]))].index
            # Waypoint is not within a mesh cell, but could still be on the edge of one. So perturbing the position slightly to the north-east and checking again. 
            #If this is not the case then the waypoint is not within the navigable domain and will continue
            if len(indices) == 0:
                try:
                    indices = mesh[(mesh['geometry'].contains(Point(wpt[['Long','Lat']]+1e-5)))].index
                except:
                    continue
            if len(indices) == 0:
                continue
            if len(indices) > 1:
                raise Exception('Waypoint lies in multiple cell boxes. Please check mesh ! ')
            else:
                wpts['index'].loc[idx] = int(indices[0])

        self.mesh['waypoints'] = wpts[~wpts['index'].isnull()]
        self.mesh['waypoints']['index'] = self.mesh['waypoints']['index'].astype(int)
        self.mesh['waypoints'] =  self.mesh['waypoints'].to_json()

        # ==== Printing Configuration and Information
        self.mesh['waypoints'] =  pd.read_json(self.mesh['waypoints'])

        # # ===== Running the route planner for the given information
        # if ("dijkstra_only" in self.config) and self.config['dijkstra_only']:
        #     self.compute_routes()
        # else:
        #     self.compute_routes()
        #     self.compute_smoothed_routes()


        # # === Saving file to output or saving it to variable output
        # output = self.to_json()
        # if ('output' in self.config) and (type(self.config['output']) == str):
        #     with open(self.config['output'], 'w') as f:
        #         json.dump(output,f)
        # else:
        #     self.output = output



    def to_json(self):
        '''
            Outputting the information in JSON format
        '''
        mesh = copy.copy(self.mesh)
        mesh['waypoints'] = mesh['waypoints'].to_dict()
        output_json = json.loads(json.dumps(mesh))
        del mesh
        return output_json

    def _dijkstra_paths(self, start_waypoints, end_waypoints):
        """
            Hidden function. Given internal variables and start and end waypoints this function
            returns a GEOJSON formatted path dict object

            INPUTS:
                start_waypoints: Start waypoint names (list)
                end_waypoints: End waypoint names (list)
        """

        geojson = dict()
        geojson['type'] = "FeatureCollection"

        paths = []
        wpts_s = self.mesh['waypoints'][self.mesh['waypoints']['Name'].isin(start_waypoints)]
        wpts_e = self.mesh['waypoints'][self.mesh['waypoints']['Name'].isin(end_waypoints)]


        for _, wpt_a in wpts_s.iterrows():
            wpt_a_name  = wpt_a['Name']
            wpt_a_index = int(wpt_a['index'])
            wpt_a_loc   = [[wpt_a['Long'],wpt_a['Lat']]]
            for _, wpt_b in wpts_e.iterrows():
                wpt_b_name  = wpt_b['Name']
                wpt_b_index = int(wpt_b['index'])
                wpt_b_loc   = [[wpt_b['Long'],wpt_b['Lat']]]
                if not wpt_a_name == wpt_b_name:
                    try:
                        graph = self.dijkstra_info[wpt_a_name]
                        path = dict()
                        path['type'] = "Feature"
                        path['geometry'] = {}
                        path['geometry']['type'] = "LineString"
                        path_points = (np.array(wpt_a_loc+list(np.array(graph['pathPoints'].loc[wpt_b_index])[:-1, :])+wpt_b_loc))
                        path['geometry']['coordinates'] = path_points.tolist()

                        path['properties'] = {}
                        path['properties']['name'] = 'Route Path - {} to {}'.format(wpt_a_name, wpt_b_name)
                        path['properties']['from'] = '{}'.format(wpt_a_name)
                        path['properties']['to'] = '{}'.format(wpt_b_name)

                        cellIndices  = np.array(graph['pathIndex'].loc[wpt_b_index])
                        path_indices = np.array([cellIndices[0]] + list(np.repeat(cellIndices[1:-1], 2)) + [cellIndices[-1]])
                        path['properties']['CellIndices'] = path_indices.tolist()

                        cases = []

                        # Determine cases for cell pairs along the path
                        for idx in range(len(cellIndices) -1):
                            cellStart = graph.loc[cellIndices[idx]]
                            cellEnd = graph.loc[cellIndices[idx+1]]
                            case = cellStart['case'][np.where(np.array(cellStart['neighbourIndex']) == cellEnd.name)[0][0]]
                            cases+=[case]

                        start_case = cases[0]
                        end_case = cases[-1]

                        # Full list of cases for each leg (centre to crossing point and crossing point to centre)
                        path_cases = list(np.repeat(cases, 2))

                        # Applying in-cell correction for travel-time
                        cost_func    = self.cost_func(source_graph=self.dijkstra_info[wpt_a_name].loc[path_indices[0]],
                                                      neighbour_graph=self.dijkstra_info[wpt_a_name].loc[path_indices[0]],
                                                      unit_shipspeed='km/hr', unit_time=self.unit_time, zerocurrents=self.zero_currents,
                                                      case=start_case)
                        tt_start = cost_func.waypoint_correction(path_points[0, :], path_points[1, :])
                        cost_func    = self.cost_func(source_graph=self.dijkstra_info[wpt_a_name].loc[path_indices[-1]],
                                                      neighbour_graph=self.dijkstra_info[wpt_a_name].loc[path_indices[0]],
                                                      unit_shipspeed='km/hr', unit_time=self.unit_time, zerocurrents=self.zero_currents,
                                                      case=end_case)

                        tt_end = cost_func.waypoint_correction(path_points[-1, :], path_points[-2, :])
                        path['properties']['traveltime']     = np.array(graph['path_traveltime'].loc[wpt_b_index])
                        path['properties']['traveltime']     = (path['properties']['traveltime'] - path['properties']['traveltime'][0]) + tt_start
                        path['properties']['traveltime'][-1] = (path['properties']['traveltime'][-2] + tt_end)
                        path['properties']['cases'] = [int(p) for p in path_cases]

                        for vrbl in self.config['path_variables']:
                            if vrbl == 'traveltime':
                                continue
                            traveltime_diff = np.r_[path['properties']['traveltime'][0], np.diff(path['properties']['traveltime'])]
                            variable_vals = graph.loc[path_indices, '{}'.format(vrbl)]
                            case_vals = pd.Series(data=[v[np.where(self.indx_type==path_cases[i])[0][0]] for i, v in enumerate(variable_vals)],
                                                  index=variable_vals.index)
                            variable_diff = traveltime_diff*case_vals
                            path['properties'][vrbl] = np.cumsum(variable_diff.to_numpy()).tolist()
                            path['properties']['total_'+vrbl] = path['properties'][vrbl][-1]
                        path['properties']['traveltime'] = path['properties']['traveltime'].tolist()
                        path['properties']['total_traveltime'] = path['properties']['traveltime'][-1]
                        paths.append(path)

                    except:
                        logging.warning('{} to {} - Failed to construct path direct in the dijkstra information'.format(wpt_a_name,wpt_b_name))

        geojson['features'] = paths
        return geojson



    def _objective_value(self, variable, source_graph, neighbour_graph, traveltime, case):
        """
            Hidden variable. Returns the objective value between two cellboxes.
        """
        if variable == 'traveltime':
            objs = np.array([source_graph['shortest_traveltime'] + traveltime[0],source_graph['shortest_traveltime'] + np.sum(traveltime)])
            return objs
        else:
            if type(source_graph['{}'.format(variable)]) == list and len(source_graph['{}'.format(variable)]) != 1:
                idx = np.where(self.indx_type==case)[0][0]
                objs = np.array([source_graph['shortest_{}'.format(variable)] + traveltime[0]*source_graph['{}'.format(variable)][idx],source_graph['shortest_{}'.format(variable)] + traveltime[0]*source_graph['{}'.format(variable)][idx] + traveltime[1]*neighbour_graph['{}'.format(variable)][idx]])
                return objs
            else:
                objs = np.array([source_graph['shortest_{}'.format(variable)] + traveltime[0]*source_graph['{}'.format(variable)], source_graph['shortest_{}'.format(variable)] + traveltime[0]*source_graph['{}'.format(variable)] + traveltime[1]*neighbour_graph['{}'.format(variable)]])
                return objs

    def _neighbour_cost(self, wpt_name, minimum_objective_index):
        """
            Determines the neighbour cost from a source cellbox to all of its neighbours.
            These are then used to update the edge values in the dijkstra graph.
        """
        # Determining the nearest neighbour index for the cell
        source_graph   = self.dijkstra_info[wpt_name].loc[minimum_objective_index]
        # Reset these arrays to empty so no duplicates are produced
        source_graph['neighbourTravelLegs'] = []
        source_graph['neighbourCrossingPoints'] = []

        # Looping over idx
        for idx in range(len(source_graph['case'])):
            indx = source_graph['neighbourIndex'][idx]

            neighbour_graph = self.dijkstra_info[wpt_name].loc[indx]
            case = source_graph['case'][idx]

            # Applying Newton curve to determine crossing point
            cost_func    = self.cost_func(source_graph=source_graph, neighbour_graph=neighbour_graph, case=case,
                                          unit_shipspeed='km/hr', unit_time=self.unit_time,
                                          zerocurrents=self.zero_currents)
            # Updating the Dijkstra graph with the new information
            traveltime, crossing_points,cell_points,case = cost_func.value()

            source_graph['neighbourTravelLegs'].append(traveltime)
            source_graph['neighbourCrossingPoints'].append(np.array(crossing_points))

            # Using neighbourhood cost determine objective function value
            value = self._objective_value(self.config['objective_function'], source_graph,neighbour_graph, traveltime, case)
            if value[1] < neighbour_graph['shortest_{}'.format(self.config['objective_function'])]:
                for vrbl in self.config['path_variables']:
                    value = self._objective_value(vrbl, source_graph, neighbour_graph,traveltime, case)
                    neighbour_graph['shortest_{}'.format(vrbl)] = value[1]
                    neighbour_graph['path_{}'.format(vrbl)]   = source_graph['path_{}'.format(vrbl)] + list(value)
                neighbour_graph['pathIndex']  = source_graph['pathIndex']  + [indx]
                neighbour_graph['pathPoints'] = source_graph['pathPoints'] + [list(crossing_points)] + [list(cell_points)]
                self.dijkstra_info[wpt_name].loc[indx] = neighbour_graph

        self.dijkstra_info[wpt_name].loc[minimum_objective_index] = source_graph

    def _dijkstra(self, wpt_name):
        """
            Runs dijkstra across the whole of the domain.
        """
        # Including only the End Waypoints defined by the user
        wpts = self.mesh['waypoints'][self.mesh['waypoints']['Name'].isin(self.end_waypoints)]
        
        # Initialising zero traveltime at the source location
        source_index = int(self.mesh['waypoints'][self.mesh['waypoints']['Name'] == wpt_name]['index'])

        for vrbl in self.config['path_variables']:
            self.dijkstra_info[wpt_name].loc[source_index, 'shortest_{}'.format(vrbl)] = 0.0
        self.dijkstra_info[wpt_name].loc[source_index, 'pathIndex'].append(source_index)
        
        # # Updating Dijkstra as long as all the waypoints are not visited or for full graph
        if self.config['early_stopping_criterion']:
            stopping_criterion_indices = wpts['index']
        else:
            stopping_criterion_indices = self.dijkstra_info[wpt_name].index

        while (self.dijkstra_info[wpt_name].loc[stopping_criterion_indices, 'positionLocked'] == False).any():

            # Determining the index of the minimum objective function that has not been visited
            minimum_objective_index = self.dijkstra_info[wpt_name][self.dijkstra_info[wpt_name]['positionLocked'] == False]['shortest_{}'.format(self.config['objective_function'])].idxmin()
  
            # Finding the cost of the nearest neighbours from the source cell (Sc)
            self._neighbour_cost(wpt_name,minimum_objective_index)

            # Updating Position to be locked
            self.dijkstra_info[wpt_name].loc[minimum_objective_index, 'positionLocked'] = True

    def compute_routes(self):
        """
            Computes the Dijkstra Paths between waypoints. 
            `waypoints` and `paths` are appended to output JSON
        """

        # Subsetting the waypoints
        if len(self.source_waypoints) == 0:
            raise Exception('No source waypoints defined that are accessible')
        if len(self.end_waypoints) == 0:
            raise Exception('No destination waypoints defined that are accessible')

        # Initialising the Dijkstra Info Dictionary
        for wpt in self.source_waypoints:
            self.dijkstra_info[wpt] = copy.copy(self.neighbour_graph)

        # 
        logging.info('============= Dijkstra Path Creation ============')
        logging.info(' - Objective = {} '.format(self.config['objective_function']))

        for wpt in self.source_waypoints:
            logging.info('--- Processing Waypoint = {} ---'.format(wpt))
            if len(self.mesh['waypoints'][self.mesh['waypoints']['Name'] == wpt]) == 0:
                logging.warning('{} not in accessible waypoints, continuing'.format(wpt))
                continue
            elif (len(self.mesh['waypoints']['Name'] == wpt) > 0) and (len(self.mesh['waypoints']['Name'] != wpt) == 0):
                logging.warning('{} is accessible but has no destination waypoints, continuing'.format(wpt))
                continue
            else:
                self._dijkstra(wpt)


        # Using Dijkstra Graph compute path and meta information to all end_waypoints
        self.paths = self._dijkstra_paths(self.source_waypoints, self.end_waypoints)
        self.mesh['paths'] = self.paths

    def compute_smoothed_routes(self,blocked_metric='SIC',debugging=False):
        """
            Using the previously constructed Dijkstra paths smooth the paths to remove mesh features 
            `paths` will be updated in the output JSON
        """
        # ====== Routes info =======
        if len(self.paths['features']) == 0:
            raise Exception('Paths not constructed as there were no dijkstra paths created')
        routes = copy.deepcopy(self.paths)['features']  


        # ====== Determining route info ======
        if 'smoothing_max_iterations' in self.config:
            max_iterations = self.config['smoothing_max_iterations']
        else:
            max_iterations = 2000
        if 'smoothing_blocked_sic' in self.config:
            blocked_sic = self.config['smoothing_blocked_sic']
        else:
            blocked_sic = 10.0
        if 'smoothing_merge_separation' in self.config:
            merge_separation = self.config['smoothing_merge_separation']
        else:
            merge_separation = 1e-3
        if 'smoothing_converged_sep' in self.config:
            converged_sep = self.config['smoothing_converged_sep']
        else:
            converged_sep = 1e-3

        logging.info('========= Determining Smoothed Paths ===========\n')
        geojson = {}
        SmoothedPaths = []
        for route in routes:
            logging.info('---Smoothing {}'.format(route['properties']['name']))
            dijkstra_graph = self.dijkstra_info[route['properties']['from']]
            self.initialise_dijkstra_graph = _initialise_dijkstra_graph(dijkstra_graph)
            self.route = route
            self.adjacent_pairs,self.start_waypoint,self.end_waypoint = _initialise_dijkstra_route(self.initialise_dijkstra_graph,self.route)

            sf = Smoothing(self.initialise_dijkstra_graph,
                           self.adjacent_pairs,
                           self.start_waypoint,self.end_waypoint,
                           blocked_metric=blocked_metric,
                           max_iterations=max_iterations,
                           blocked_sic = blocked_sic,
                           merge_separation=merge_separation,
                           converged_sep=converged_sep)
            self.sf = sf
            self.sf.forward()

            
            # ------ Smooth Path Values -----
            # Given a smoothed route path now determine the along path parameters.
            pv             = PathValues()
            path_info      = pv.objective_function(sf.aps,sf.start_waypoint,sf.end_waypoint)
            variables      = path_info['variables']
            TravelTimeLegs = variables['traveltime']['path_values']
            DistanceLegs   = variables['distance']['path_values'] 
            pathIndex      = variables['cell_index']['path_values'] 
            FuelLegs       = variables['fuel']['path_values'] 
            SpeedLegs      = variables['speed']['path_values'] 



            # ------ Saving Output in a standard form to be saved ------
            SmoothedPath ={}
            SmoothedPath['type'] = 'Feature'
            SmoothedPath['geometry'] = {}
            SmoothedPath['geometry']['type'] = "LineString"
            SmoothedPath['geometry']['coordinates'] = path_info['path'].tolist()            
            SmoothedPath['properties'] = {}
            SmoothedPath['properties']['from']       = route['properties']['from']
            SmoothedPath['properties']['to']         = route['properties']['to']
            SmoothedPath['properties']['traveltime'] = list(TravelTimeLegs)
            SmoothedPath['properties']['total_traveltime'] = SmoothedPath['properties']['traveltime'][-1]
            SmoothedPath['properties']['fuel']       = list(FuelLegs)
            SmoothedPath['properties']['total_fuel'] = SmoothedPath['properties']['fuel'][-1]
            SmoothedPath['properties']['distance']   = list(DistanceLegs)
            SmoothedPath['properties']['speed']      = list(SpeedLegs)
            SmoothedPaths += [SmoothedPath]

        geojson['type'] = "FeatureCollection"
        geojson['features'] = SmoothedPaths
        self.smoothed_paths = geojson
        self.mesh['paths'] = self.smoothed_paths