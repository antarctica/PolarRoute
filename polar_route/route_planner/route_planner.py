"""
    This section of the codebase is used for construction of routes using the
    environmental mesh between a series of user defined waypoints
"""

import json
import warnings
import numpy as np
import pandas as pd
from shapely import wkt, Point, LineString, STRtree, Polygon
import geopandas as gpd
import logging
import itertools
from pandas.core.common import SettingWithCopyWarning

from polar_route.route_planner.route import Route
from polar_route.route_planner.source_waypoint import SourceWaypoint
from polar_route.route_planner.waypoint import Waypoint
from polar_route.route_planner.segment import Segment
from polar_route.route_planner.routing_info import RoutingInfo
from polar_route.route_planner.crossing import NewtonianDistance
from polar_route.utils import _json_str, unit_speed
from meshiphi.mesh_generation.environment_mesh import EnvironmentMesh
from meshiphi.mesh_generation.direction import Direction

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# Functions for flexible waypoints, TODO update to work with refactored code
def _mesh_boundary_polygon(mesh):
    """
        Creates a polygon from the mesh boundary
    """
    lat_min = mesh['config']['mesh_info']['region']['lat_min']
    lat_max = mesh['config']['mesh_info']['region']['lat_max']
    long_min = mesh['config']['mesh_info']['region']['long_min']
    long_max = mesh['config']['mesh_info']['region']['long_max']
    p1 = Point([long_min, lat_min])
    p2 = Point([long_min, lat_max])
    p3 = Point([long_max, lat_max])
    p4 = Point([long_max, lat_min])
    return Polygon([p1,p2,p3,p4])


def _adjust_waypoints(point, cellboxes, max_distance=5):
    """
        Moves waypoint to the closest accessible cellbox if it isn't already in one. Allows up to 5 degrees flexibility
        by default.
    """
    # Extract cellboxes from mesh
    cbs = gpd.GeoDataFrame.from_records(cellboxes)
    # Prune inaccessible waypoints from search
    cbs = cbs[cbs['inaccessible'] == False]
    # Find nearest cellbox to point that is accessible
    tree = STRtree(wkt.loads(cbs['geometry']))
    nearest_cb = cbs.iloc[tree.nearest(point)]
    cb_polygon = wkt.loads(nearest_cb['geometry'])
    
    if point.within(cb_polygon):
        logging.debug(f'({point.y},{point.x}) in accessible cellbox')
        return point
    else:
        logging.debug(f'({point.y},{point.x}) not in accessible cellbox')
        # Create a line between CB centre and point
        cb_centre = Point([nearest_cb['cx'],nearest_cb['cy']])
        connecting_line = LineString([point, cb_centre])
        # Extract segment of line inside the accessible cellbox    
        intersecting_line = connecting_line.intersection(cb_polygon)
        
        # Put limit on how far it's allowed to adjust waypoint
        distance_away = connecting_line.length - intersecting_line.length
        if distance_away > max_distance:
            logging.info(f'Waypoint too far from accessible cellbox!')
            return point
        
        # Find where it meets the cellbox boundary
        boundary_point = connecting_line.intersection(cb_polygon.exterior)
        # Draw a small circle around it
        buffered_point = boundary_point.buffer(intersecting_line.length*1e-3)
        # Find point along line that intersects circle
        adjusted_point = buffered_point.exterior.intersection(intersecting_line)
        # Interior point is now a point inside the cellbox 
        # that is not on the boundary
        logging.info(f'({point.y},{point.x}) not accessible cellbox')
        logging.info(f'Adjusted to ({adjusted_point.y},{adjusted_point.x})')
        return adjusted_point


class RoutePlanner:
    """
        RoutePlanner finds the optimal route between a series of waypoints.
        The routes are constructed in a two stage process:

        compute_routes: uses a mesh based Dijkstra method to determine the optimal routes 
                        between a series of waypoint.

        compute_smoothed_routes: smooths the compute_routes using information from the environmental mesh
                                to determine mesh independent optimal route paths

        ---

        Attributes:
            env_mesh (EnvironmentMesh): mesh object that contains the mesh's cellboxes information and neighbour graph.
            cost_func (func): Crossing point cost function for Dijkstra Path creation.
            config (Json): JSON object that contains the attributes required for the route construction. 
            src_wps (list<SourceWaypoint>): a list of the source waypoints that contains all of the dijkstra routing
            information to reuse this information for routes with the same source waypoint.

        ---
    """

    def __init__(self, mesh_file, config_file, cost_func=NewtonianDistance):
        """
            Constructs the routes in the given mesh using information given in the config file.

            Args:

                mesh_file(string): the path to the mesh json file that contains the mesh's cellboxes information and neighbourhood graph

                config_file (string): the path to the config JSON file which defines the attributes required for the route construction. 
                    Sections required for the route construction are as follows\n
                    \n
                    {\n
                        "objective_function": (string) currently either 'traveltime' or 'fuel',\n
                        "path_variables": list of (string),\n
                        "vector_names": list of (string),\n
                        "time_unit" (string),\n
                    }\n

                cost_func (func): Crossing point cost function for Dijkstra Path creation. For development purposes only!
        """
        mesh_json = _json_str(mesh_file)

        self.env_mesh = EnvironmentMesh.load_from_json(mesh_json)
        self.config = _json_str(config_file)
        self.config['unit_shipspeed'] = mesh_json['config']['vessel_info']['unit']

        # validate config and mesh TODO replace with function from config_validation
        mandatory_fields = ["objective_function", "path_variables", "vector_names", "time_unit"]
        for field in mandatory_fields: 
            if field not in self.config:
                 raise ValueError(f'missing configuration: {field} should be set in the provided configuration')
        # check that the provided mesh has vector information (ex. current)
        self.vector_names = self.config['vector_names']
        for name in self.vector_names: 
             if  name not in self.env_mesh.agg_cellboxes[0].agg_data :
                 raise ValueError(f'The env mesh cellboxes do not have {name} data and it is a prerequisite for the '
                                  f'route planner!')
        if 'SIC' not in self.env_mesh.agg_cellboxes[0].agg_data :
            logging.warning('The environment mesh does not have SIC data')
        
        # check if speed is defined in the environment mesh
        if 'speed' not in self.env_mesh.agg_cellboxes[0].agg_data:
            raise ValueError('Vessel Speed not in the mesh information! Please run vessel performance')
        
        #  check if objective function is in the environment mesh (e.g. speed)
        if self.config['objective_function'] != 'traveltime':
            if self.config['objective_function'] not in self.env_mesh.agg_cellboxes[0].agg_data:
                raise ValueError(f"Objective Function '{self.config['objective_function']}' requires  the mesh cellboxes"
                                 f" to have '{self.config['objective_function']}' in the aggregated data")

        self.cellboxes_lookup = {str(self.env_mesh.agg_cellboxes[i].get_id()): self.env_mesh.agg_cellboxes[i]
                                 for i in range(len(self.env_mesh.agg_cellboxes))}
        # ====== Defining the cost function ======
        self.cost_func = cost_func
        # Case indices
        self.src_wps = []

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
                
                s_wp.print_routing_table()
                route_segments = []
                e_wp = end_waypoints[i]
                e_wp_indx = e_wp.get_cellbox_indx()
                cases = []

                if s_wp.get_cellbox_indx() == e_wp_indx: # path should be a straight line within the same cellbox
                   route = Route([Segment(s_wp, e_wp)], s_wp.get_name(), e_wp.get_name(), self.config)
                else:
                    while s_wp.get_cellbox_indx() != e_wp_indx:
                        # print (">>> s_wp_indx >>>", s_wp)
                        # print (">>> e_wp_indx >>>", e_wp_indx)
                        routing_info = s_wp.get_routing_info(e_wp_indx)
                        # Insert segments at the front of the list as we are moving from e_wp to s_wp
                        route_segments.insert(0, routing_info.get_path())
                        neighbour_case = (self.env_mesh.neighbour_graph.get_neighbour_case(
                            self.cellboxes_lookup[routing_info.get_node_index()],
                            self.cellboxes_lookup[e_wp_indx]))
                        # Add case twice to cover travel to/from crossing point
                        for x in range(2):
                            cases.insert(0, neighbour_case)
                        e_wp_indx = routing_info.get_node_index()
                        print("route segments >> ", route_segments[0][0].to_str())
                   
                    route_segments = list(itertools.chain.from_iterable(route_segments))
                    route = Route(route_segments, s_wp.get_name(), e_wp.get_name(), self.config)
                    route.set_cases(cases)
                # correct the first and last segment
                    for s in route_segments:
                        print(">>>|S|>>>> ", s.to_str())
                print(route.segments[0].get_start_wp().get_cellbox_indx())
                
                route._waypoint_correction(self.cellboxes_lookup[route.segments[0].get_start_wp().get_cellbox_indx()], s_wp, route.segments[0].get_end_wp(),  0)
                if len(route.segments) >1:  # make sure we have more one segment as we might have only one segment if the src and destination are within the same cellbox
                    route._waypoint_correction(self.cellboxes_lookup[route.segments[-1].get_end_wp().get_cellbox_indx()], e_wp, route.segments[-1].get_start_wp(), -1)
                routes.append(route)
                print(route.to_json())
                
        return routes

    def _dijkstra(self, wp, end_wps):
        """
            Runs dijkstra across the whole of the domain.cellboxes_lookup
            Args:
                wp (SourceWaypoint): object contains the lat, long information of the source waypoint
                end_wps(List(Waypoint)): a list of the end waypoints
        """
        def find_min_objective(source_wp):
            min_obj = np.inf
            cellbox_indx = -1
            # source_wp.print_routing_table()
            for node_id in source_wp.routing_table.keys():
                if not source_wp.is_visited(str(node_id)) and source_wp.get_obj(node_id, self.config['objective_function']) < min_obj:
                    min_obj = source_wp.get_obj( node_id, self.config['objective_function'])
                    cellbox_indx = node_id
            return str(cellbox_indx)
        def consider_neighbours(source_wp, _id):
            # get neighbours of _id
            neighbour_map = self.env_mesh.neighbour_graph.get_neighbour_map(_id)  #neighbours and cases
            for case, neighbours in neighbour_map.items():
                if len(neighbours) !=0:
                  for neighbour in neighbours:  
                     if not source_wp.is_visited(neighbour): # to avoid cycles
                        edges = self._neighbour_cost(_id, str(neighbour), int(case))
                        edges_cost = sum(segment.get_variable(self.config['objective_function']) for segment in edges)
                        new_cost =  source_wp.get_obj( _id, self.config['objective_function'])+ edges_cost
                        if new_cost < source_wp.get_obj(str(neighbour), self.config['objective_function']):
                            source_wp.update_routing_table(str(neighbour), RoutingInfo(_id, edges))
                
        # # Updating Dijkstra as long as all the waypoints are not visited or for full graph
        print(">>>> src >>>> ", wp.get_cellbox_indx())
        print(">>>> end_wp >>>> ", end_wps[0].get_cellbox_indx())
        while not wp.is_all_visited():
            min_obj_indx = find_min_objective(wp)  # Determining the index of the minimum objective function that has not been visited
            print("min_obj >>> ", min_obj_indx )
            
            consider_neighbours(wp, min_obj_indx)
            wp.visit(min_obj_indx)

    def _neighbour_cost(self, node_id, neighbour_id, case):
        direction = [1, 2, 3, 4, -1, -2, -3, -4]

        # Applying Newton distance to determine crossing point between node and its neighbour
        cost_func    = self.cost_func(node_id, neighbour_id, self.cellboxes_lookup, case=case,
                                          unit_shipspeed='km/hr', unit_time=self.config['time_unit'])
        # Updating the Dijkstra graph with the new information
        traveltime, crossing_points,cell_points,case = cost_func.value()
        # create segments and set their travel time based on the returned 3 points and the remaining obj accordingly(travel_time * node speed/fuel), and return
        s1 = Segment(Waypoint.load_from_cellbox(self.cellboxes_lookup[node_id]), Waypoint(crossing_points[1], crossing_points[0], cellbox_indx=node_id))
        s1.set_travel_time(traveltime[0])
        # print (">>> travel time >> ", traveltime)
        #fill segment metrics
        s1.set_fuel(s1.get_travel_time() * self.cellboxes_lookup[node_id].agg_data['fuel'][direction.index(case)])
        s1.set_distance(s1.get_travel_time() * unit_speed(self.cellboxes_lookup[node_id].agg_data['speed'][direction.index(case)], self.config ['unit_shipspeed']))
        s2 = Segment( Waypoint(crossing_points[1], crossing_points[0], cellbox_indx=neighbour_id), Waypoint.load_from_cellbox(self.cellboxes_lookup[neighbour_id]))
        s2.set_travel_time(traveltime[1])
        s2.set_fuel( s2.get_travel_time() * self.cellboxes_lookup[neighbour_id].agg_data['fuel'][direction.index(case)])
        s2.set_distance(s2.get_travel_time() * unit_speed(self.cellboxes_lookup[neighbour_id].agg_data['speed'][direction.index(case)], self.config ['unit_shipspeed']))

        return [s1,s2]

    def compute_routes(self, waypoints):
        """
            Computes the Dijkstra Paths between waypoints. 
            Args: 
                waypoints (String/Dataframe): DataFrame that contains source and destination waypoints info/string points to the path of a csv file that contains this info
            Returns:
                routes (List<Route>): a list of the computed routes     
        """

        src_wps, end_wps =  self._load_waypoints(waypoints)
        # Waypoint validation, TODO: replace with validate_waypoints function in the future
        src_wps = self._validate_wps(src_wps)
        end_wps =  self._validate_wps(end_wps)
        src_wps = [self.get_source_wp(wp, end_wps) for wp in src_wps]   # creating SourceWaypoint objects
        if len(src_wps) == 0:
            raise ValueError('Invalid waypoints. Inaccessible source waypoints')

        logging.info('============= Dijkstra Path Creation ============')
        logging.info(' - Objective = {} '.format(self.config['objective_function']))
        if len(end_wps) == 0:
            end_wps= [Waypoint.load_from_cellbox(cellbox) for cellbox in self.env_mesh.agg_cellboxes] # full graph, use all the cellboxes ids as destination
        for wp in src_wps:
            logging.info('--- Processing Waypoint = {}'.format(wp.get_name()))
            self._dijkstra(wp, end_wps)

        print(">>> dijkstra done ...")
        # Using Dijkstra Graph compute path and meta information to all end_waypoints
        return self._dijkstra_paths(src_wps, end_wps)  # returning the constructed routes
    
    def _validate_wps(self, wps):
        """
                Determines if the provided waypoint list contains valid (both lie within the bounds of the env mesh).
                Args:
                    wps (list<Waypoint>): list of waypoint object that encapsulates lat and long information
                Returns:
                   Wps (list<Waypoint>): list of waypoint object that encapsulates lat and long information after
                   removing the invalid waypoints
        """
        def select_cellbox(ids):
            """
            In case a WP lies on the border of 2 cellboxes,  this method applies the selection criteria between the
            cellboxes (the current criteria is to select the north-east cellbox).
                Args:
                    ids([int]): list contains the touching cellboxes ids
                Returns:
                    selected (int): the id of the selected cellbox
            """
            print(">>> selecting cellbox ...")
            if (self.env_mesh.neighbour_graph.get_neighbour_case(self.cellboxes_lookup [ids[0]],
                                                                self.cellboxes_lookup [ids[1]]) in
                    [Direction.east, Direction.north_east, Direction.north]):
                return ids[1]
            return ids[0]
      
        valid_wps = wps
        for wp in wps: 
            wp_id = []
            for indx in range(len(self.env_mesh.agg_cellboxes)):
                if (self.env_mesh.agg_cellboxes[indx].contains_point(wp.get_latitude(), wp.get_longitude())
                        and not self.env_mesh.agg_cellboxes[indx].agg_data ['inaccessible']):
                    wp_id. append(self.env_mesh.agg_cellboxes[indx].get_id())
                    wp.set_cellbox_indx(str(self.env_mesh.agg_cellboxes[indx].get_id()))
            if len(wp_id) == 0:
                logging.warning(f'{wp.get_name()} is not an accessible waypoint')
                valid_wps.remove(wp)
        
            if len(wp_id) > 1: # the source wp is on the border of 2 cellboxes
                _id = select_cellbox(wp_id)
                wp.set_cellbox_indx(str(_id))

        return valid_wps

    def get_source_wp(self, src_wp, end_wps):
        for wp in self.src_wps:
            if wp.equals(src_wp):
                wp.set_end_wp(end_wps)
                return wp
        wp = SourceWaypoint(src_wp, end_wps)
        self.src_wps.append(wp)
        return wp
    
    def _load_waypoints(self, waypoints):
            try:
                waypoints_df = waypoints
                if isinstance(waypoints, dict):
                    waypoints_df = pd.DataFrame.from_dict(waypoints)  
                if  isinstance(waypoints, str):
                     waypoints_df = pd.read_csv(waypoints)
                source_waypoints_df = waypoints_df[waypoints_df['Source'] == "X"]
                dest_waypoints_df = waypoints_df[waypoints_df['Destination'] == "X"]
                src_wps = [Waypoint(lat=source['Lat'], long=source['Long'], name=source['Name'])
                           for index, source in source_waypoints_df.iterrows()]
                dest_wps = [Waypoint(lat=dest['Lat'], long=dest['Long'], name=dest['Name'])
                            for index, dest in dest_waypoints_df.iterrows()]
                return  src_wps, dest_wps
            except FileNotFoundError:
                raise ValueError(f"Unable to load '{waypoints}', please check path name")


if __name__ == '__main__':

      config = None
    #   mesh_file = "../tests/regression_tests/example_routes/dijkstra/time/checkerboard.json"
      mesh_file = "add_vehicle.output.json"
      

    #   mesh_file = "grf_reprojection.json"
      wp_file = "../tests/unit_tests/resources/waypoint/waypoints_2.csv"
      route_conf = "../tests/unit_tests/resources/waypoint/route_config.json"
      route_planner= None
      vessel_mesh = None
      with open(mesh_file, "r") as mesh_json:
          #config = json.load(mesh_json)['config']
          vessel_mesh =  json.load(mesh_json)
      #mesh_json = MeshBuilder(config).build_environmental_mesh().to_json()
    #   mesh_json = json.load(mesh_json)
      

    #   vp = VesselPerformanceModeller(mesh_json, config['vessel_info'])
    #   vp.model_accessibility()
    #   vp.model_performance()
    #   info = vp.to_json()
    #   json.dump(info, open('vessel_mesh.json', "w"), indent=4)
    #   with open(route_conf, "r") as config_file:
    #       config = json.load(config_file)
    #   route_planner= RoutePlanner("vessel_mesh.json", route_conf)
      route_planner = RoutePlanner(mesh_file, route_conf)
    # #   src, dest = route_planner._load_waypoints(wp_file)
    # #   route_planner._validate_wps(src)
    # #   route_planner._validate_wps(dest)
    #   routes = route_planner.compute_routes(vessel_mesh['waypoints'])
      routes = route_planner.compute_routes(wp_file)
      print(routes[0].to_json())
