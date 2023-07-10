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

from polar_route.mesh_generation.environment_mesh import EnvironmentMesh
from polar_route.source_waypoint import SourceWaypoint
from polar_route.waypoint import waypoint
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
            mesh(EnvironmentMesh): mesh object that contains the mesh's cellboxes information and neighbourhood graph
            cost_func (func): Crossing point cost function for Dijkstra Path creation.
            config (Json): JSON object that contains the attributes required for the route construction. 

        ---

    """

    def __init__(self, mesh_file, config_file, cost_func=NewtonianDistance):

        """
            Constructs the routes from information given in the config file.

            Args:

                mesh_file(string): the path to the mesh json file that contains the mesh's cellboxes information and neighbourhood graph

                config_file (string): the path to the config JSON file which defines the attributes required for the route construction. 
                    Sections required for the route construction are as follows\n
                    \n
                    {\n
                        "objective_function": (string) currently either 'traveltime' or 'fuel',\n
                        "path_variables": list of (string),\n
                        "vector_names": (list of (string),\n
                        "zero_currents": (boolean),\n
                        "time_unit" (string),\n
                        "early_stopping_criterion" (boolean),\n
                    }\n

                cost_func (func): Crossing point cost function for Dijkstra Path creation. For development purposes only!
        """

        # Load in the current cell structure & Optimisation Info̦
        self.env_mesh = EnvironmentMesh.load_from_json (_json_str(mesh_file))
        self.config = _json_str(config_file)
        # check that the provided mesh has current information
        if 'uC' not in self.env_mesh.agg_cellboxes[0].agg_data or 'vC' not in self.env_mesh.agg_cellboxes[0].agg_data  :
            raise ValueError('The env mesh cellboxes do not have current information and it is a prerequisite for the route planner!')
        # check if speed defined in the env mesh
        if 'speed' not in self.env_mesh.agg_cellboxes[0].agg_data:
            raise ValueError('Vessel Speed not in the mesh information ! Please run vessel performance')
        
        #  check if objective function is in the env mesh (ex. speed)            
        if self.config['objective_function'] != 'traveltime':
            if self.config['objective_function'] not in self.env_mesh.agg_cellboxes[0].agg_data:
                raise ValueError("Objective Function '{}' requires  the mesh cellboxex to have '{}' in the aggregated data".format(self.config['objective_function'], self.config['objective_function']))

        # ====== Defining the cost function ======
        self.cost_func       = cost_func
    

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

    def _dijkstra(self, wp , end_wps):
        """
            Runs dijkstra across the whole of the domain.
            Args:
                wp (Waypoint): object contains the lat, long information of the source waypoint
                end_wps(List(Waypoint)): a list of the end waypoints
        """
        
        # # Updating Dijkstra as long as all the waypoints are not visited or for full graph
        if not self.config['early_stopping_criterion']:
          #TODO: handle
          pass

        source_wp = SourceWaypoint (wp)
        while not source_wp.is_all_visited (end_wps):   

            # Determining the index of the minimum objective function that has not been visited
           # get the neighbours
        #    TODO: update
            self.env_mesh.neighbour_graph [source_wp.get_cellbox_indx()]
            self.env_mesh.agg_cellboxes[source_wp.get_cellbox_indx()][self.config['objective_function']]  # find which of tnhme is not visited and has the minimal obj_func
            minimum_objective_index = self.dijkstra_info[wpt_name][self.dijkstra_info[wpt_name]['positionLocked'] == False]['shortest_{}'.format(self.config['objective_function'])].idxmin()
  
            # Finding the cost of the nearest neighbours from the source cell (Sc)
            self._neighbour_cost(wpt_name,minimum_objective_index)

            # Updating Position to be locked
            self.dijkstra_info[wpt_name].loc[minimum_objective_index, 'positionLocked'] = True

    def compute_routes(self, waypoints):
        """
            Computes the Dijkstra Paths between waypoints. 
            Args: 
                waypoints (List <(src_wp, dest_wp)>): a list of pair of source and dest waypoints
            Returns:
                routes (List<Route>): a list of the computed routes     
        """
        for wp_pair in waypoints:
            if not self._is_valid_wp_pair(wp_pair[0], wp_pair[1]):
                waypoints.remove (wp_pair)
        
        if len(waypoints) == 0:
            raise ValueError('No waypoint pair defined that is accessible')
    

        # 
        logging.info('============= Dijkstra Path Creation ============')
        logging.info(' - Objective = {} '.format(self.config['objective_function']))
        end_wps =  [dest for source, dest in waypoints ]
        for wpt_pair in waypoints:
            logging.info('--- Processing Waypoint = {} ---'.format(wpt))
            self._dijkstra(wpt_pair[0], end_wps)


        # Using Dijkstra Graph compute path and meta information to all end_waypoints
        return self._dijkstra_paths(self.source_waypoints, self.end_waypoints)  # returning the constructed routes
    
def _is_valid_wp_pair (self , source , dest):
    """
            Determines if the provided waypoint pair is valid (both lie within the bounds of the env mesh).
            Args:
                source (Waypoint): waypoint object that encapsulates the source's lat and long information
                dest (Waypoint): waypoint object that encapsulates the dest's lat and long information
            Returns:
                is_valid (Boolean): true if both source and dest are within the env mesh bounds and false otherwise
     
    """
    #TODO: what to do if the provided source is not inside a cellbox but close from a cellbox?
    source_id = -1
    dest_id = -1
    for indx in range (len(self.env_mesh.agg_cellboxes)):
        if self.env_mesh.agg_cellboxes[indx].contains_point (source.get_latitude() , source.get_longtitude()):
            source_indx = indx
            source.set_cellbox_indx(source_indx)
        if self.env_mesh.agg_cellboxes[indx].contains_point (dest.get_latitude() , dest.get_longtitude()):
            dest_indx = indx
            dest.set_cellbox_indx(dest_indx)
    if source_id == -1:
          logging.warning('{} not in accessible waypoints'.format(source.get_name()))
          return False
    if dest_id == -1:
         logging.warning('{} is accessible but has no destination waypoints'.format(dest.get_name()))
         return False
    
    return True