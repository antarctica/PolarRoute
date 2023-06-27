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

from polar_route.crossing import NewtonianDistance, NewtonianCurve
from polar_route.utils import _json_str

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
         


       

        ---

    """

    def __init__(self, env_mesh, config, cost_func=NewtonianDistance):

        """
            Constructs the routes from information given in the config file.

            Args:

                env_mesh(EnvironmentMesh): mesh object that contains the mesh's cellboxes information and neighbourhood graph

                config (dict or string of filepath): config JSON which defines the attributes required for the route construction. 
                    Sections required for the route construction are as follows\n
                    \n
                    {\n
                        "objective_function": (string) currently either 'traveltime' or 'fuel',\n
                        "path_variables": list of (string),\n
                        "waypoints_path": (string),\n
                        "source_waypoints": list of (string),\n
                        "end_waypoints": list of (string),\n
                        "vector_names": (list of (string),\n
                        "zero_currents": (boolean),\n
                        "variable_speed" (boolean),\n
                        "time_unit" (string),\n
                        "early_stopping_criterion" (boolean),\n
                        "save_dijkstra_graphs": (boolean),\n
                        "smooth_path":{\n
                            "max_iteration_number":(int),\n
                            "minimum_difference":(float),\n
                    }\n

                cost_func (func): Crossing point cost function for Dijkstra Path creation. For development purposes only!
        """

        # Load in the current cell structure & Optimisation Info̦
        self.mesh             = _json_str(env_mesh)
        self.config           = _json_str(config)
     

        # Case indices
        self.indx_type = np.array([1, 2, 3, 4, -1, -2, -3, -4])

        # Creating a blank path construct
        self.paths          = None
        self.smoothed_paths = None
        self.dijkstra_info = {}

        # ====== Loading Mesh & Neighbour Graph ======
     

        # ====== Speed Function Checking ======
        # Checking if Speed defined in file
        if 'speed' not in self.neighbour_graph:
            print(self.neighbour_graph.columns)
            raise Exception('Vessel Speed not in the mesh information ! Please run vessel performance')

        # ====== Objective Function Information ======
        #  Checking if objective function is in the cellgrid            
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
            #If this is not the case then the waypoint is not within the navitagatable domain and will continue
            if len(indices) == 0:
                try:
                    indices = mesh[(mesh['geometry'].contains(Point(wpt[['Long','Lat']]+1e-5)))].index
                except:
                    continue
            if len(indices) == 0:
                continue
            if len(indices) > 1:
                raise Exception('Wapoint lies in multiple cell boxes. Please check mesh ! ')
            else:
                wpts['index'].loc[idx] = int(indices[0])

    



    def to_json(self):
        '''
            Outputing the information in JSON format
        '''
        mesh = copy.copy(self.mesh)
        mesh['waypoints'] = mesh['waypoints'].to_dict()
        mesh['config']['route_info'] = self.config
        output_json = json.loads(json.dumps(mesh, indent=4))
        del mesh
        return output_json

    def _dijkstra_paths(self, start_waypoints, end_waypoints):
        """
            Hidden function. Given internal variables and start and end waypoints this function
            returns a GEOJSON formated path dict object

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
                        path['properties']['traveltime'] = path['properties']['traveltime'].tolist()
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

   