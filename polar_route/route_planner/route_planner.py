"""
    This module is used for construction of routes within an
    environmental mesh between a series of user defined waypoints
"""

import numpy as np
import pandas as pd
from shapely import wkt, Point, LineString, STRtree
import geopandas as gpd
import logging
import itertools
import copy

from polar_route.route_planner.route import Route
from polar_route.route_planner.source_waypoint import SourceWaypoint
from polar_route.route_planner.waypoint import Waypoint
from polar_route.route_planner.segment import Segment
from polar_route.route_planner.routing_info import RoutingInfo
from polar_route.route_planner.crossing import NewtonianDistance
from polar_route.route_planner.crossing_smoothing import Smoothing, FindEdge, PathValues, rhumb_line_distance
from polar_route.config_validation.config_validator import validate_route_config
from polar_route.config_validation.config_validator import validate_waypoints
from polar_route.utils import json_str, unit_speed, pandas_dataframe_str, case_from_angle, timed_call
from meshiphi import Boundary
from meshiphi.mesh_generation.environment_mesh import EnvironmentMesh
from meshiphi.mesh_generation.direction import Direction
from meshiphi.utils import longitude_domain

# Squelching SettingWithCopyWarning
pd.options.mode.chained_assignment = None


def _mesh_boundary_polygon(mesh):
    """
    Creates a polygon from the mesh boundary
    """
    # Defining a tiny value, zero if global mesh
    if (mesh['config']['mesh_info']['region']['long_min'] == -180 or
            mesh['config']['mesh_info']['region']['long_max'] == 180):
        tiny_value = 0
    else:
        tiny_value = 1e-10

    lat_min = mesh['config']['mesh_info']['region']['lat_min'] - tiny_value
    lat_max = mesh['config']['mesh_info']['region']['lat_max'] + tiny_value
    long_min = mesh['config']['mesh_info']['region']['long_min'] - tiny_value
    long_max = mesh['config']['mesh_info']['region']['long_max'] + tiny_value

    bounds = Boundary([lat_min, lat_max], [long_min, long_max])

    return bounds.to_polygon()


# Functions for flexible waypoints
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
        cb_centre = Point([nearest_cb['cx'], nearest_cb['cy']])
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


def flatten_cases(cell_id, neighbour_graph):
    """
        Identifies the cases with neighbours around a given cell and gets the ids of those neighbouring cells.

        Args:
            cell_id (str): The id of the cell to find the neighbours for
            neighbour_graph(dict): The neighbour graph of the mesh

        Returns:
            neighbour_case (list): A list of neighbouring case directions
            neighbour_indx (list): A list of neighbouring cell indices
    """
    neighbour_case = []
    neighbour_indx = []
    neighbours = neighbour_graph[cell_id]
    for case in neighbours.keys():
        for neighbour in neighbours[case]:
            neighbour_case.append(int(case))
            neighbour_indx.append(int(neighbour))
    return neighbour_case, neighbour_indx


def initialise_dijkstra_route(dijkstra_graph, dijkstra_route):
    """
        Initialising dijkstra route into a standard form used for smoothing

        Args:
            dijkstra_graph (dict): Dictionary comprising dijkstra graph with keys based on cellbox id.
                                    Each entry is a dictionary of the cellbox environmental and dijkstra information.

            dijkstra_route (dict): Dictionary of a GeoJSON entry for the dijkstra route

        Returns:
            aps (list<FindEdge>): A list of adjacent cell pairs where each entry is of type FindEdge including
                                  information on .crossing, .case, .start, and .end (see 'find_edge' for more information)

    """

    org_path_points = np.array(dijkstra_route['geometry']['coordinates'])
    org_cell_indices = np.array(dijkstra_route['properties']['CellIndices'])
    org_cell_cases= np.array(dijkstra_route['properties']['cases'])

    # -- Generating a dataframe of the case information --
    points      = np.concatenate([org_path_points[0,:][None,:], org_path_points[1:-1:2], org_path_points[-1,:][None,:]])
    cell_indices = np.concatenate([[org_cell_indices[0]], [org_cell_indices[0]], org_cell_indices[1:-1:2],
                                  [org_cell_indices[-1]], [org_cell_indices[-1]]])
    cell_cases = np.concatenate([[org_cell_cases[0]], [org_cell_cases[0]], org_cell_cases[1:-1:2], [org_cell_cases[-1]],
                                 [org_cell_cases[-1]]])

    cell_dijk = [dijkstra_graph[int(ii)] for ii in cell_indices]
    cells = cell_dijk[1:-1]
    cases = cell_cases[1:-1]
    aps = []
    for ii in range(len(cells) - 1):
        aps += [FindEdge(cells[ii], cells[ii+1], cases[ii+1])]

    # #-- Setting some backend information
    start_waypoint = points[0,:]
    end_waypoint = points[-1,:]

    return aps, start_waypoint, end_waypoint


def _load_waypoints(waypoints):
    """
    Load source and destination waypoints from dict or file

    Args:
        waypoints (dict or str): waypoints dict or path to file to load waypoints from

    Returns:
        src_wps (list): list of source waypoints
        dest_wps (list): list of destination waypoints
    """
    try:
        waypoints_df = waypoints
        if isinstance(waypoints, dict):
            waypoints_df = pd.DataFrame.from_dict(waypoints)
        if isinstance(waypoints, str):
             waypoints_df = pd.read_csv(waypoints)
        source_waypoints_df = waypoints_df[waypoints_df['Source'] == "X"]
        dest_waypoints_df = waypoints_df[waypoints_df['Destination'] == "X"]
        src_wps = [Waypoint(lat=source['Lat'], long=source['Long'], name=source['Name'])
                   for index, source in source_waypoints_df.iterrows()]
        dest_wps = [Waypoint(lat=dest['Lat'], long=dest['Long'], name=dest['Name'])
                    for index, dest in dest_waypoints_df.iterrows()]
        return src_wps, dest_wps
    except FileNotFoundError:
        raise ValueError(f"Unable to load '{waypoints}', please check file path provided")


class RoutePlanner:
    """
        RoutePlanner finds the optimal route between a series of waypoints.
        The routes are constructed in a two stage process:

        **compute_routes**: Uses a mesh based Dijkstra method to determine the optimal routes between a series of waypoints.

        **compute_smoothed_routes**: Smooths the output of **compute_routes** using information from the environmental mesh to determine mesh independent optimal routes.

        ---

        Attributes:
            env_mesh (EnvironmentMesh): mesh object that contains the mesh's cellboxes information and neighbour graph
            cost_func (func): Crossing point cost function for Dijkstra Route creation
            config (Json): JSON object that contains the attributes required for the route construction. 
            src_wps (list<SourceWaypoint>): a list of the source waypoints that contains all of the dijkstra routing
                                            information to reuse this information for routes with the same source
                                            waypoint

        ---
    """

    def __init__(self, mesh_file, config_file, cost_func=NewtonianDistance):
        """
            Constructs the routes within the mesh using parameters provided in the config file.

            Args:

                mesh_file(string): the path to the mesh json file that contains the cellbox information and neighbour graph

                config_file (string): the path to the config JSON file which defines the attributes required for the route construction. 
                    Sections required for the route construction are as follows\n
                    \n
                    {\n
                        "objective_function": (string) currently either 'traveltime', 'battery' or 'fuel',\n
                        "path_variables": list of (string),\n
                        "vector_names": list of (string),\n
                    }\n

                cost_func (func): Crossing point cost function for Dijkstra route construction. For development purposes
                                  only!
        """
        # Load and validate input config
        self.config = json_str(config_file)
        validate_route_config(self.config)

        # Only switch off waypoint adjustment if specified in config
        if 'adjust_waypoints' not in self.config:
            self.config['adjust_waypoints'] = True

        # Set default time unit if not specified in config
        if 'time_unit' not in self.config:
            self.config['time_unit'] = "days"

        # Load mesh json from file or dict
        mesh_json = json_str(mesh_file)

        # Set speed units from config
        self.config['unit_shipspeed'] = mesh_json['config']['vessel_info']['unit']

        # Zeroing currents if vectors names are not defined or zero_currents is defined
        mesh_json = self._zero_currents(mesh_json)
        mesh_json = self._fixed_speed(mesh_json)

        # Initialise EnvironmentMesh object
        self.env_mesh = EnvironmentMesh.load_from_json(mesh_json)

        self.cellboxes_lookup = {str(self.env_mesh.agg_cellboxes[i].get_id()): self.env_mesh.agg_cellboxes[i]
                                 for i in range(len(self.env_mesh.agg_cellboxes))}

        # Check that the provided mesh has vector information (ex. current)
        self.vector_names = self.config['vector_names']
        for name in self.vector_names: 
             if not any(name in cb.agg_data for cb in self.cellboxes_lookup.values()):
                 raise ValueError(f'The env mesh cellboxes do not have {name} data and it is a prerequisite for the '
                                  f'route planner!')
        # Check for SIC data, used in smoothed route construction
        if not any('SIC' in cb.agg_data for cb in self.cellboxes_lookup.values()):
            logging.debug('The environment mesh does not have SIC data')
        
        # Check if speed is defined in the environment mesh
        if not any('speed' in cb.agg_data for cb in self.cellboxes_lookup.values()):
            raise ValueError('Vessel speed not in the mesh information! Please run vessel performance')
        
        # Check if objective function is in the environment mesh (e.g. speed)
        if self.config['objective_function'] != 'traveltime':
            if not any(self.config['objective_function'] in cb.agg_data for cb in self.cellboxes_lookup.values()):
                raise ValueError(f"Objective Function '{self.config['objective_function']}' requires the mesh cellboxes"
                                 f" to have '{self.config['objective_function']}' in the aggregated data")

        # ====== Defining the cost function ======
        self.cost_func = cost_func

        # Define attributes
        self.src_wps = []
        self.waypoints_df = None
        self.routes_dijkstra = []
        self.routes_smoothed = []
        self.neighbour_legs = {}

    def _splitting_around_waypoints(self, waypoints_df):
        """
            Applying splitting around waypoints if this is defined in config. This is applied
            inplace.

            Args:
                waypoints_df (pd.DataFrame): Pandas DataFrame of Waypoint locations
            Applied to terms:
                self.config (dict): PolarRoute config file\n
                self.env_mesh (EnvironmentMesh): The EnvironmentMesh object for the relevant mesh

        """
        if ('waypoint_splitting' in self.config) and (self.config['waypoint_splitting']):
            logging.info(' Splitting around waypoints !')
            wps_points = [(entry['Lat'], entry['Long']) for _, entry in waypoints_df.iterrows()]
            self.env_mesh.split_points(wps_points)
            # Rebuild lookup with new env_mesh
            self.cellboxes_lookup = {str(self.env_mesh.agg_cellboxes[i].get_id()): self.env_mesh.agg_cellboxes[i]
                                     for i in range(len(self.env_mesh.agg_cellboxes))}

    def _zero_currents(self, mesh):
        """
            Applying zero currents to mesh

            Args:
                mesh (JSON): MeshiPhi Mesh input
            Returns:
                mesh (JSON): MeshiPhi Mesh Corrected
        """

        # Zeroing currents if both vectors are defined and zeroed
        if ('zero_currents' in self.config) and ("vector_names" in self.config):
            if self.config['zero_currents']:
                logging.info('Zero currents set in config for this mesh!')
                for idx, cell in enumerate(mesh['cellboxes']):
                    cell[self.config['vector_names'][0]] = 0.0
                    cell[self.config['vector_names'][1]] = 0.0
                    mesh['cellboxes'][idx] = cell

        # If no vectors are defined then add zero currents to mesh
        if 'vector_names' not in self.config:
            self.config['vector_names'] = ['Vector_x', 'Vector_y']
            logging.info('No vector_names defined in config. Zeroing currents in mesh !')
            for idx, cell in enumerate(mesh['cellboxes']):
                cell[self.config['vector_names'][0]] = 0.0
                cell[self.config['vector_names'][1]] = 0.0
                mesh['cellboxes'][idx] = cell

        return mesh

    def _fixed_speed(self, mesh):
        """
            Applying max speed for all cellboxes that are accessible

            Args:
                mesh (JSON): MeshiPhi Mesh input
            Returns:
                mesh (JSON): MeshiPhi Mesh Corrected
        """

        # Setting speed to a fixed value if specified in the config
        if 'fixed_speed' in self.config:
            if self.config['fixed_speed']:
                logging.info('Setting all speeds to max speed for this mesh!')
                max_speed = mesh['config']['vessel_info']['max_speed']
                for idx, cell in enumerate(mesh['cellboxes']):
                    if 'speed' in cell.keys():
                        cell['speed'] = [max_speed,
                                         max_speed,
                                         max_speed,
                                         max_speed,
                                         max_speed,
                                         max_speed,
                                         max_speed,
                                         max_speed]
                        mesh['cellboxes'][idx] = cell
                    else:
                        continue

        return mesh

    def _dijkstra_routes(self, start_waypoints, end_waypoints):
        """
            Hidden function. Given internal variables and start and end waypoints this function
            returns a list of routes

            Args:
                start_waypoints (list<Waypoint>): list of start waypoints
                end_waypoints (list<Waypoint>): list of end waypoints
            Return:
                routes (list<Route>): list of the constructed routes
        """
        routes = []

        # Loop over all source waypoints
        for s_wp in start_waypoints:
            s_wp.log_routing_table()
            # Loop over all end waypoints
            for e_wp in end_waypoints:
                # Don't try to calculate route between waypoints at the same location
                if s_wp.equals(e_wp):
                    # No need to log this for the "route" from a waypoint to itself
                    if s_wp.get_name() != e_wp.get_name():
                        logging.info(f"Route from {s_wp.get_name()} to {e_wp.get_name()} not calculated, these waypoints"
                                     f" are identical")
                    continue
                route_segments = []
                e_wp_indx = e_wp.get_cellbox_indx()
                cases = []
                no_route_found = False
                # Handle case where route starts and ends in the same cell
                if s_wp.get_cellbox_indx() == e_wp_indx:
                   # Route should be a straight line within the same cellbox
                   route = Route([Segment(s_wp, e_wp)], s_wp.get_name(), e_wp.get_name(), self.config)
                   route.source_waypoint = s_wp
                else:
                    while s_wp.get_cellbox_indx() != e_wp_indx:
                        # logging.debug(">>> s_wp_indx >>>", s_wp)
                        # logging.debug(">>> e_wp_indx >>>", e_wp_indx)
                        routing_info = s_wp.get_routing_info(e_wp_indx)
                        # If no route found break out of loop and skip this case
                        if routing_info.get_node_index() == -1:
                            logging.warning(f'{s_wp.get_name()} to {e_wp.get_name()} - Failed to construct Dijkstra route')
                            no_route_found = True
                            break
                        # Insert segments at the front of the list as we are moving from e_wp to s_wp
                        route_segments.insert(0, routing_info.get_path())
                        neighbour_case = (self.env_mesh.neighbour_graph.get_neighbour_case(
                            self.cellboxes_lookup[routing_info.get_node_index()],
                            self.cellboxes_lookup[e_wp_indx]))
                        # If no neighbour case found try global mesh method for cells touching anti-meridian
                        if neighbour_case == 0:
                            neighbour_case = (self.env_mesh.neighbour_graph.get_global_mesh_neighbour_case(
                                self.cellboxes_lookup[routing_info.get_node_index()],
                                self.cellboxes_lookup[e_wp_indx]))
                        # Add case twice to cover travel to/from crossing point
                        for x in range(2):
                            cases.insert(0, neighbour_case)
                        e_wp_indx = routing_info.get_node_index()
                        logging.debug(f"route segments >> {route_segments[0][0].to_str()}")
                   
                    route_segments = list(itertools.chain.from_iterable(route_segments))
                    route = Route(route_segments, s_wp.get_name(), e_wp.get_name(), self.config)
                    route.source_waypoint = s_wp
                    route.set_cases(cases)
                    for s in route_segments:
                        logging.debug(f">>>|S|>>>> {s.to_str()}")

                if no_route_found:
                    continue

                logging.debug(route.segments[0].get_start_wp().get_cellbox_indx())
                # Correct the first and last segment of the route
                route.waypoint_correction(self.cellboxes_lookup[route.segments[0].get_start_wp().get_cellbox_indx()],
                                          s_wp, route.segments[0].get_end_wp(), 0)
                # Check we have more one segment before correcting the last segment,
                # as we might have only one segment if the src and destination waypoints are within the same cellbox
                if len(route.segments) > 1:
                    route.waypoint_correction(self.cellboxes_lookup[route.segments[-1].get_end_wp().get_cellbox_indx()],
                                              e_wp, route.segments[-1].get_start_wp(), -1)
                routes.append(route)
                logging.debug(route.to_json())
                
        return routes

    def _dijkstra(self, wp, end_wps):
        """
            Runs Dijkstra's algorithm across the whole of the domain

            Args:
                wp (SourceWaypoint): object contains the lat, long information of the source waypoint
                end_wps(List(Waypoint)): a list of the end waypoints
        """

        def find_min_objective(source_wp):
            """
            Finds the index of the unvisited cell in the source waypoint's routing table with the minimum cost for
            the given objective function
            Args:
                source_wp (SourceWaypoint): the SourceWaypoint object corresponding to the initial location

            Returns:
                cellbox_indx (str): the index of the minimum cost unvisited cell from the routing table
            """
            min_obj = np.inf
            cellbox_indx = -1
            for node_id in source_wp.routing_table.keys():
                if (not source_wp.is_visited(node_id) and
                        source_wp.get_obj(node_id, self.config['objective_function']) < min_obj):
                    min_obj = source_wp.get_obj(node_id, self.config['objective_function'])
                    cellbox_indx = str(node_id)
            return cellbox_indx

        def consider_neighbours(source_wp, _id):
            """
            Get neighbours of the cell at the input index and update the routing table of the source waypoint
            Args:
                source_wp (SourceWaypoint): the relevant source waypoint
                _id (str): index of the cell to get the neighbours for

            """
            neighbour_map = self.env_mesh.neighbour_graph.get_neighbour_map(_id) # neighbours and cases for node _id
            for case, neighbours in neighbour_map.items():
                if len(neighbours) != 0:
                  for neighbour in neighbours:
                    edges = self._neighbour_cost(_id, str(neighbour), int(case))
                    edges_cost = sum(segment.get_variable(self.config['objective_function']) for segment in edges)
                    new_cost = source_wp.get_obj( _id, self.config['objective_function']) + edges_cost
                    if new_cost < source_wp.get_obj(str(neighbour), self.config['objective_function']):
                        source_wp.update_routing_table(str(neighbour), RoutingInfo(_id, edges))
                
        # Updating Dijkstra as long as all the waypoints are not visited or for full graph
        for end_wp in end_wps:
            if wp.equals(end_wp):
                continue
            logging.info(f"Destination waypoint: {end_wp.get_name()}")
            while not wp.is_visited(end_wp.get_cellbox_indx()):
                # Determine the index of the cell with the minimum objective function cost that has not yet been visited
                min_obj_indx = find_min_objective(wp)
                logging.debug(f"min_obj >>> {min_obj_indx}")
                # If min_obj_indx is -1 then no route possible, and we stop search for this waypoint
                if min_obj_indx == -1:
                    break
                consider_neighbours(wp, min_obj_indx)
                wp.visit(min_obj_indx)

    def _neighbour_cost(self, node_id, neighbour_id, case):
        """
        Determines the neighbour cost when travelling from the cell at node_id to the cell at neighbour_id given the
        specified case.

        Args:
            node_id (str): the id of the initial cellbox
            neighbour_id (str): the id of the neighbouring cellbox
            case (int): the case of the transition from node_id to neighbour_id

        Returns:
            neighbour_segments (list<Segment>): a list of segments that form the legs of the route from node_id to neighbour_id

        """
        direction = [1, 2, 3, 4, -1, -2, -3, -4]

        # Applying Newton distance to determine crossing point between node and its neighbour
        cost_func = self.cost_func(node_id, neighbour_id, self.cellboxes_lookup, case=case,
                                    unit_shipspeed='km/hr', time_unit=self.config['time_unit'])
        # Updating the Dijkstra graph with the new information
        traveltime, crossing_points, cell_points, case = cost_func.value()
        # Save travel time and crossing point values for use in smoothing
        self.neighbour_legs[node_id+"to"+neighbour_id] = (traveltime, crossing_points)

        # Create segments and set their travel time based on the returned 3 points and the remaining obj accordingly (travel_time * node speed/fuel)
        s1 = Segment(Waypoint.load_from_cellbox(self.cellboxes_lookup[node_id]), Waypoint(crossing_points[1],
                                                                                          crossing_points[0],
                                                                                          cellbox_indx=node_id))
        s2 = Segment(Waypoint(crossing_points[1], crossing_points[0], cellbox_indx=neighbour_id),
                     Waypoint.load_from_cellbox(self.cellboxes_lookup[neighbour_id]))

        # Fill segment metrics
        s1.set_travel_time(traveltime[0])
        if 'fuel' in self.config['path_variables']:
            s1.set_fuel(s1.get_travel_time() * self.cellboxes_lookup[node_id].agg_data['fuel'][direction.index(case)])
        if 'battery' in self.config['path_variables']:
            s1.set_battery(s1.get_travel_time() * self.cellboxes_lookup[node_id].agg_data['battery'][direction.index(case)])
        s1.set_distance(s1.get_travel_time() * unit_speed(self.cellboxes_lookup[node_id].agg_data['speed'][direction.index(case)],
                                                          self.config['unit_shipspeed']))

        s2.set_travel_time(traveltime[1])
        if 'fuel' in self.config['path_variables']:
            s2.set_fuel(s2.get_travel_time() * self.cellboxes_lookup[neighbour_id].agg_data['fuel'][direction.index(case)])
        if 'battery' in self.config['path_variables']:
            s2.set_battery(
                s2.get_travel_time() * self.cellboxes_lookup[node_id].agg_data['battery'][direction.index(case)])
        s2.set_distance(s2.get_travel_time() * unit_speed(self.cellboxes_lookup[neighbour_id].agg_data['speed'][direction.index(case)],
                                                          self.config['unit_shipspeed']))

        neighbour_segments = [s1, s2]

        return neighbour_segments

    @timed_call
    def compute_routes(self, waypoints):
        """
            Computes the Dijkstra routes between waypoints.

            Args: 
                waypoints (String/Dataframe): DataFrame that contains source and destination waypoints info or a string
                                              pointing to the path of a csv file that contains this info
            Returns:
                routes (List<Route>): a list of the computed routes
        """
        waypoints_df = pandas_dataframe_str(waypoints)
        # Handle common issues with case/whitespace in Source/Destination fields
        waypoints_df['Source'] = [s.strip().upper() if type(s) == str else np.nan for s in waypoints_df['Source']]
        waypoints_df['Destination'] = [s.strip().upper() if type(s) == str else np.nan for s in waypoints_df['Destination']]
        # Validate input waypoints format
        validate_waypoints(waypoints_df)

        self.waypoints_df = waypoints_df

        # Move waypoint to the closest accessible cellbox, if it isn't in one already
        mesh_boundary = _mesh_boundary_polygon(self.env_mesh.to_json())
        for idx, row in waypoints_df.iterrows():
            point = Point([row['Long'], row['Lat']])
            # Only allow waypoints within an existing mesh
            assert point.within(mesh_boundary), \
                f"Waypoint {row['Name']} outside of mesh boundary! {point}"
            if self.config['adjust_waypoints']:
                logging.debug("Adjusting waypoints in inaccessible cells to nearest accessible location")
                adjusted_point = _adjust_waypoints(point, self.env_mesh.to_json()['cellboxes'])

                waypoints_df.loc[idx, 'Long'] = adjusted_point.x
                waypoints_df.loc[idx, 'Lat'] = adjusted_point.y
            else:
                logging.debug("Skipping waypoint adjustment")

        # Split around waypoints if specified in the config
        self._splitting_around_waypoints(waypoints_df)

        # Load source and destination waypoints
        src_wps, end_wps = _load_waypoints(waypoints_df)
        # Waypoint validation for route planning
        src_wps = self._validate_wps(src_wps)
        end_wps = self._validate_wps(end_wps)

        # Create SourceWaypoint objects
        src_wps = [SourceWaypoint(wp, end_wps) for wp in src_wps]
        self.src_wps.append(src_wps)
        if len(src_wps) == 0:
            raise ValueError('Invalid waypoints. Inaccessible source waypoints')

        logging.info('============= Dijkstra Route Creation ============')
        logging.info(f" - Objective = {self.config['objective_function']}")
        if len(end_wps) == 0:
            end_wps = [Waypoint.load_from_cellbox(cellbox) for cellbox in self.env_mesh.agg_cellboxes] # full graph, use all the cellboxes ids as destination
        for wp in src_wps:
            logging.info('--- Processing Source Waypoint = {}'.format(wp.get_name()))
            self._dijkstra(wp, end_wps)

        # Using Dijkstra graph compute route and meta information to all end_waypoints
        routes = self._dijkstra_routes(src_wps, end_wps)
        logging.info("Dijkstra routing complete...")
        self.routes_dijkstra = routes
        # Returning the constructed routes
        return routes

    @timed_call
    def compute_smoothed_routes(self, blocked_metric='SIC'):
        """
            Uses the previously constructed Dijkstra routes and smooths them to remove mesh features
            `paths` will be updated in the output JSON
        """

        # ====== Routes info =======
        # Check whether any Dijkstra routes exist before running smoothing
        if len(self.routes_dijkstra) == 0:
            raise Exception('Smoothed routes not constructed as there were no Dijkstra routes created')
        routes = copy.deepcopy(self.routes_dijkstra)

        # ====== Determining route info ======
        # Get smoothing parameters from config or set default values
        max_iterations = self.config.get('smoothing_max_iterations', 2000)
        blocked_sic = self.config.get('smoothing_blocked_sic', 10.0)
        merge_separation = self.config.get('smoothing_merge_separation', 1e-3)
        converged_sep = self.config.get('smoothing_converged_sep', 1e-3)

        logging.info('========= Determining Smoothed Routes ===========')
        geojson = {}
        smoothed_routes = []

        mesh_json = self.env_mesh.to_json()
        neighbour_graph = mesh_json['neighbour_graph']
        cellboxes = mesh_json['cellboxes']

        for route in routes:
            route_json = route.to_json()

            # Handle straight line route within same cell
            if len(route_json['properties']['CellIndices']) == 1:
                logging.info(f"--- Skipping smoothing for {route_json['properties']['name']}, direct route within a"
                             f" single cell")
                # Set and remove some additional info for final output to match smoothed routes
                start_location = route_json['geometry']['coordinates'][0]
                end_location = route_json['geometry']['coordinates'][1]
                route_cell = route_json['properties']['CellIndices'][0]
                route_case = case_from_angle(start_location, end_location)
                route_json['properties']['distance'] = [0., rhumb_line_distance(start_location, end_location)]
                route_json['properties']['speed'] = [0., self.cellboxes_lookup[route_cell].agg_data['speed'][route_case]]
                for var in self.config['path_variables']:
                    route_json['properties'][var].insert(0, 0.)
                del route_json['properties']['cases']
                del route_json['properties']['CellIndices']
                del route_json['properties']['name']
                # Add straight line route to list of outputs
                smoothed_routes += [route_json]
                continue

            logging.info(f"--- Smoothing {route_json['properties']['name']}")

            initialised_dijkstra_graph = self.initialise_dijkstra_graph(cellboxes, neighbour_graph, route)
            adjacent_pairs, source_wp, end_wp = initialise_dijkstra_route(initialised_dijkstra_graph, route_json)

            sf = Smoothing(initialised_dijkstra_graph,
                           adjacent_pairs,
                           source_wp, end_wp,
                           blocked_metric=blocked_metric,
                           max_iterations=max_iterations,
                           blocked_sic = blocked_sic,
                           merge_separation=merge_separation,
                           converged_sep=converged_sep)

            sf.forward()

            # ------ Smoothed Route Values -----
            # Given a smoothed route now determine the parameters along the route.
            pv = PathValues(self.config['path_variables'])
            path_info = pv.objective_function(sf.aps, sf.start_waypoint, sf.end_waypoint)
            # Ensure all coordinates are in domain -180:180
            path_info['path'][:, 0] = longitude_domain(path_info['path'][:, 0])
            variables = path_info['variables']
            travel_time_legs = variables['traveltime']['path_values']
            distance_legs = variables['distance']['path_values']
            speed_legs = variables['speed']['path_values']

            # ------ Saving Output in a standard form to be saved ------
            smoothed_route = dict()
            smoothed_route['type'] = 'Feature'
            smoothed_route['geometry'] = {}
            smoothed_route['geometry']['type'] = "LineString"
            smoothed_route['geometry']['coordinates'] = path_info['path'].tolist()
            smoothed_route['properties'] = {}
            smoothed_route['properties']['from'] = route_json['properties']['from']
            smoothed_route['properties']['to'] = route_json['properties']['to']
            smoothed_route['properties']['traveltime'] = list(travel_time_legs)
            smoothed_route['properties']['total_traveltime'] = smoothed_route['properties']['traveltime'][-1]
            smoothed_route['properties']['distance'] = list(distance_legs)
            smoothed_route['properties']['speed'] = list(speed_legs)

            if 'fuel' in self.config['path_variables']:
                fuel_legs = variables['fuel']['path_values']
                smoothed_route['properties']['fuel'] = list(fuel_legs)
                smoothed_route['properties']['total_fuel'] = smoothed_route['properties']['fuel'][-1]
            if 'battery' in self.config['path_variables']:
                battery_legs = variables['battery']['path_values']
                smoothed_route['properties']['battery'] = list(battery_legs)
                smoothed_route['properties']['total_battery'] = smoothed_route['properties']['battery'][-1]

            smoothed_routes += [smoothed_route]

            logging.info('Smoothing complete in {} iterations'.format(sf.jj))

        geojson['type'] = "FeatureCollection"
        geojson['features'] = smoothed_routes
        self.routes_smoothed = geojson
        return self.routes_smoothed

    def initialise_dijkstra_graph(self, cellboxes, neighbour_graph, route, path_index=False):
        """
            Initialising dijkstra graph information in a standard form used for the smoothing

            Args:
                cellboxes (list): List of cells with environmental and vessel performance info
                neighbour_graph (dict): Neighbour graph for the mesh
                route (Route): Route object for the route to be smoothed
                path_index (bool): Option to generate the pathIndex array that can be used to generate new dijkstra routes

            Returns:
                dijkstra_graph_dict (dict): Dictionary comprising dijkstra graph with keys based on cellbox id.
                                             Each entry is a dictionary of the cellbox environmental and dijkstra information.

        """
        dijkstra_graph_dict = dict()
        for cell in cellboxes:
            if cell['inaccessible']:
                continue
            cell_id = int(cell['id'])
            dijkstra_graph_dict[cell_id] = cell
            if 'SIC' not in cell:
                dijkstra_graph_dict[cell_id]['SIC'] = 0.0
            dijkstra_graph_dict[cell_id]['id'] = cell_id
            dijkstra_graph_dict[cell_id]['Vector_x'] = dijkstra_graph_dict[cell_id][self.config['vector_names'][0]]
            dijkstra_graph_dict[cell_id]['Vector_y'] = dijkstra_graph_dict[cell_id][self.config['vector_names'][1]]
            cases, neighbour_index = flatten_cases(str(cell_id), neighbour_graph)
            dijkstra_graph_dict[cell_id]['case'] = np.array(cases)
            dijkstra_graph_dict[cell_id]['neighbourIndex'] = np.array(neighbour_index)
            neighbour_travel_legs = []
            neighbour_crossing_points = []
            for i, neighbour in enumerate(neighbour_index):
                leg_id = str(cell_id) + "to" + str(neighbour)
                if leg_id in self.neighbour_legs:
                    neighbour_travel_legs.append(self.neighbour_legs[leg_id][0])
                    neighbour_crossing_points.append(self.neighbour_legs[leg_id][1])
            dijkstra_graph_dict[cell_id]['neighbourTravelLegs'] = np.array(neighbour_travel_legs)
            dijkstra_graph_dict[cell_id]['neighbourCrossingPoints'] = np.array(neighbour_crossing_points)
            if path_index:
                dijkstra_graph_dict[cell_id]['pathIndex'] = route.source_waypoint.get_path_nodes(str(cell_id))

        return dijkstra_graph_dict

    def _validate_wps(self, wps):
        """
            Determines if the provided waypoint list contains valid waypoints (i.e. both lie within the bounds of
            the env mesh).

            Args:
                wps (list<Waypoint>): list of waypoint objects that encapsulates lat and long information

            Returns:
                Wps (list<Waypoint>): list of waypoint objects that encapsulates lat and long information after
                removing any invalid waypoints
        """
        def select_cellbox(ids):
            """
            In case a WP lies on the border of 2 cellboxes, this method applies the selection criteria between the
            cellboxes (the current criteria is to select the north-east cellbox).
                Args:
                    ids([int]): list contains the touching cellboxes ids
                Returns:
                    selected (int): the id of the selected cellbox
            """
            logging.debug(">>> selecting cellbox for waypoint on boundary...")
            if (self.env_mesh.neighbour_graph.get_neighbour_case(self.cellboxes_lookup[ids[0]],
                                                                self.cellboxes_lookup[ids[1]]) in
                    [Direction.east, Direction.north_east, Direction.north]):
                return ids[1]
            return ids[0]
      
        valid_wps = wps
        for wp in wps: 
            wp_id = []
            for indx in range(len(self.env_mesh.agg_cellboxes)):
                if (self.env_mesh.agg_cellboxes[indx].contains_point(wp.get_latitude(), wp.get_longitude())
                        and not self.env_mesh.agg_cellboxes[indx].agg_data['inaccessible']):
                    wp_id. append(self.env_mesh.agg_cellboxes[indx].get_id())
                    wp.set_cellbox_indx(str(self.env_mesh.agg_cellboxes[indx].get_id()))
            if len(wp_id) == 0:
                logging.warning(f'{wp.get_name()} is not an accessible waypoint')
                valid_wps.remove(wp)
        
            if len(wp_id) > 1: # the source wp is on the border of 2 cellboxes
                _id = select_cellbox(wp_id)
                wp.set_cellbox_indx(str(_id))

        return valid_wps

    def to_json(self):
        """
        Output all information from the RoutePlanner object in json format

        Returns:
            output_json (dict): the full mesh and route information in json format

        """
        output_json = self.env_mesh.to_json()
        output_json['config']['route_info'] = self.config
        output_json['waypoints'] = self.waypoints_df.to_dict()

        if self.routes_smoothed:
            output_json['paths'] = self.routes_smoothed
        elif self.routes_dijkstra:
            output_json['paths'] = {"type": "FeatureCollection", "features": []}
            output_json['paths']['features'] = [dr.to_json() for dr in self.routes_dijkstra]
        else:
            output_json['paths'] = []

        return output_json
