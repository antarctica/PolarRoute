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
from polar_route.route import Route
from polar_route.source_waypoint import SourceWaypoint
from polar_route.waypoint import Waypoint
from polar_route.segment import Segment
from polar_route.routing_info import RoutingInfo
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
            raise ValueError('The env mesh cellboxes do not have current data and it is a prerequisite for the route planner!')
        if 'SIC' not in self.env_mesh.agg_cellboxes[0].agg_data :
            logging.warning('The env mesh does not have SIC data')
        
        # check if speed defined in the env mesh
        if 'speed' not in self.env_mesh.agg_cellboxes[0].agg_data:
            raise ValueError('Vessel Speed not in the mesh information ! Please run vessel performance')
        
        #  check if objective function is in the env mesh (ex. speed)            
        if self.config['objective_function'] != 'traveltime':
            if self.config['objective_function'] not in self.env_mesh.agg_cellboxes[0].agg_data:
                raise ValueError("Objective Function '{}' requires  the mesh cellboxex to have '{}' in the aggregated data".format(self.config['objective_function'], self.config['objective_function']))

        #TODO: index cellboxes with the id for easier access
        self.cellboxes_lookup = {self.env_mesh.agg_cellboxes[i].get_id(): self.env_mesh.agg_cellboxes[i] for i in range (len(self.env_mesh.agg_cellboxes))}
        # ====== Defining the cost function ======
        self.cost_func       = cost_func
       # Case indices
        self.indx_type = np.array([1, 2, 3, 4, -1, -2, -3, -4])

    def _dijkstra_paths(self, start_waypoints, end_waypoints):
        """
            Hidden function. Given internal variables and start and end waypoints this function
            returns a list of routes

            Args:
                start_waypoints (list<Waypoint>): list of the start waypoint
                end_waypoints (list<Waypoint>): list of the end waypoint 
            Return:
                routes(list<Route>): list of the constructed routes
        """
        routes = []
        for i, s_wp in enumerate(start_waypoints):
                route_segments = []
                e_wp = end_waypoints[i]
                cases = []
                while not s_wp.equals(e_wp):
                     routing_info = s_wp.get_routing_info(e_wp.get_id())
                     route_segments.append (routing_info.get_path())
                     cases.append(self.env_mesh.neighbour_graph.get_neighbour_case( routing_info.get_node_index() , e_wp.get_cellbox_indx()))
                     e_wp = routing_info.get_node_index()
                   
                # reversing segments as we moved from end to start
                route_segments.reverse()
                route = Route (route_segments , s_wp.get_name() , end_waypoints[i].get_name())
                route.set_cases(cases)
                routes.append (route)
                
        return routes



    def _dijkstra(self, wp , end_wps):
        """
            Runs dijkstra across the whole of the domain.
            Args:
                wp (Waypoint): object contains the lat, long information of the source waypoint
                end_wps(List(Waypoint)): a list of the end waypoints
        """
        def find_min_objective (source_wp):
            min_obj = np.inf
            cellbox_indx = -1
            for indx, info in source_wp.routing_info:
                if info.get_obj (self.config['objective_function'])< min_obj:
                    min_obj = info.get_obj (self.config['objective_function'])
                    cellbox_indx = indx
            return cellbox_indx
        def consider_neighbours (source_wp , _id):
            # get neighbours of indx
            source_cellbox = self.env_mesh.agg_cellboxes[source_wp.get_cellbox_indx()]
            neighbour_map = self.env_mesh.neighbour_graph [source_cellbox.get_id()]  #neighbours and cases
            for case, neighbours in neighbour_map:
                if len (neighbours) !=0:
                  for neighbour in neighbours:  
                     #TODO: use the cost function here and define the Segement (based on crossing points and the cost) accordingly
                     edges = self._neighbour_cost(_id, neighbour, case)
                     edges_cost = sum (segment.get_obj(self.config['objective_function']) for segment in edges) 
                     new_cost =  source_wp.get_routing_info(_id)+ edges_cost
                     if new_cost < source_wp.get_routing_info(neighbour).get_obj(self.config['objective_function']):
                         source_wp.update_routing_info (neighbour , RoutingInfo (_id, edges))
            
        # # Updating Dijkstra as long as all the waypoints are not visited or for full graph
        if not self.config['early_stopping_criterion']:
            end_wps= [Waypoint.load_from_cellbox(cellbox) for cellbox in self.env_mesh.agg_cellboxes]
        wp = SourceWaypoint( wp , end_wps)
        while not wp.is_all_visited():
            min_obj_indx = find_min_objective(wp)  # Determining the index of the minimum objective function that has not been visited
            consider_neighbours (wp , min_obj_indx)
            wp.visit (min_obj_indx)

    def _neighbour_cost (self, node_id , neighbour_id , case):

        # Applying Newton distance to determine crossing point between node and its neighbour
        cost_func    = self.cost_func(node_id, neighbour_id, self.cellboxes_lookup , case=case, 
                                          unit_shipspeed='km/hr', unit_time=self.unit_time,
                                          zerocurrents=self.zero_currents)
        # Updating the Dijkstra graph with the new information
        traveltime, crossing_points,cell_points,case = cost_func.value()
        # create segments and set their travel time based on the returned 3 points and the remaining obj accordingly (travel_time * node speed/fuel), and return 
        s1 = Segment (Waypoint(self.cellboxes_lookup[node_id]) , Waypoint (crossing_points[0], crossing_points[1]))
        s1.set_travel_time(traveltime[0])
        #fill segment metrics
        case_indx = np.where(self.indx_type==case)
        s1.set_fuel (s1.travel_time * self.cellboxes_lookup[node_id].agg_data['fuel'][case_indx])
        s1.set_speed (s1.travel_time * self.cellboxes_lookup[node_id].agg_data['speed'][case_indx])
        s2 = Segment( Waypoint (crossing_points[0], crossing_points[1]), Waypoint(self.cellboxes_lookup[neighbour_id]))
        s2.set_travel_time([traveltime[1]])
        s2.set_fuel (s1.travel_time * self.cellboxes_lookup[neighbour_id].agg_data['fuel'][case_indx])
        s2.set_speed (s1.travel_time * self.cellboxes_lookup[neighbour_id].agg_data['speed'][case_indx])

        return [s1,s2]

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
            raise ValueError('Invalid waypoints. No waypoint pair defined that is accessible')
    

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
    #TODO: what to do if a point is on the border of a 2 cellboxes
    def select_cellbox (ids):
        '''
           In case a WP lies on the border of 2 cellboxes,  this method applies the selection criteria between the cellboxes
            Args:
                ids([int]): listt contains the touching cellboxes ids
            Returns:
                selected (int): the id of the selected cellbox
        '''
        # self.env_mesh.neighbour_graph.get_neighbour_case(ids[0], ids[1]) # get the touching casebetween cellboxes
        pass
    source_id = []
    dest_id = []
    for indx in range (len(self.env_mesh.agg_cellboxes)):
        if self.env_mesh.agg_cellboxes[indx].contains_point (source.get_latitude() , source.get_longtitude()):
            source_id. append (indx)
            source.set_cellbox_indx(source_id)
        if self.env_mesh.agg_cellboxes[indx].contains_point (dest.get_latitude() , dest.get_longtitude()):
            dest_id.append( indx)
            dest.set_cellbox_indx(dest_id)
    if len (source_id) == 0:
          logging.warning('{} not in accessible waypoints'.format(source.get_name()))
          return False
    if len(dest_id) ==0:
         logging.warning('{} is accessible but has no destination waypoints'.format(dest.get_name()))
         return False
    
    if len(source_id) > 1: # the source wp is on the border of 2 cellboxes
        source_id = [select_cellbox(source_id)]
    if len(dest_id)>1 :# the dest wp is on the border of 2 cellboxes
        dest_id = [select_cellbox(dest_id)]
    if source_id[0] == dest_id[0]:
        logging.warning('The source {} and destination {} waypoints lie in the same cellbox'.format (source.get_name() , dest.get_name()))
    
    return True