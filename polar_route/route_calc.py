import json
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt, distance
from shapely.geometry import Point, LineString, MultiLineString, Polygon
from polar_route.utils import gpx_route_import
from polar_route.route_planner.crossing import traveltime_in_cell
from polar_route.route_planner.crossing_smoothing import rhumb_line_distance, dist_around_globe


# Define ordering of cases in array data
case_indices = np.array([1, 2, 3, 4, -1, -2, -3, -4])


def traveltime_distance(cellbox, wp, cp, speed='speed', vector_x='uC', vector_y='vC', case=0):
    """
        Calculate travel time and distance for two points.

        Args:
            cellbox (dict): the cell containing the line segment
            wp (ndarray): the start point
            cp (ndarray): the end point
            speed (str): the key for speed
            vector_x (str): the key for the x vector component
            vector_y (str): the key for the y vector component
            case (int): case giving the index of the speed array

        Returns:
            traveltime (float): the time to travel the line segment
            distance (float): the distance along the line segment
    """

    idx = np.where(case_indices==case)[0][0]
    # Conversion factors from lat/long degrees to metres
    m_long = 111.321 * 1000
    m_lat = 111.386 * 1000
    x = dist_around_globe(cp[0], wp[0]) * m_long * np.cos(wp[1] * (np.pi / 180))
    y = (cp[1] - wp[1]) * m_lat
    if (vector_x in cellbox) and (vector_y in cellbox):
        su = cellbox[vector_x]
        sv = cellbox[vector_y]
    else:
        su = 0
        sv = 0
    ssp = cellbox[speed][idx] * (1000 / (60 * 60))
    try:
        traveltime = traveltime_in_cell(x, y, su, sv, ssp)
        dist = rhumb_line_distance(cp, wp)
    except:
        traveltime = 0
        dist = 0

    return traveltime, dist


def case_from_angle(start, end):
    """
        Determine the direction of travel between two points and return the associated case

        Args:
            start (ndarray): the coordinates of the start point within the cell
            end (ndarray):  the coordinates of the end point within the cell

        Returns:
            case (int): the case to use to select variable values from a list
    """

    direct_vec = [end[0]-start[0], end[1]-start[1]]
    direct_ang = np.degrees(np.arctan2(direct_vec[0], direct_vec[1]))

    case = None

    # Angular ranges corresponding to directional cases used in the route planner
    # Cases cover 45 degree angular segments running clockwise from 1 at NE through to -4 at N
    # See Sec. 6.1.2 in the docs for more info

    if -22.5 <= direct_ang < 22.5:
        case = -4
    elif 22.5 <= direct_ang < 67.5:
        case = 1
    elif 67.5 <= direct_ang < 112.5:
        case = 2
    elif 112.5 <= direct_ang < 157.5:
        case = 3
    elif 157.5 <= abs(direct_ang) <= 180:
        case = 4
    elif -67.5 <= direct_ang < -22.5:
        case = -3
    elif -112.5 <= direct_ang < -67.5:
        case = -2
    elif -157.5 <= direct_ang < -112.5:
        case = -1

    return case


def load_route(route_file):
    """
        Load route information from file

        Args:
            route_file (str): Path to user defined route in json, csv or gpx format

        Returns:
            df (Dataframe): Dataframe with route info
            from_wp (str): Name of start waypoint
            to_wp (str) Name of end waypoint

    """
    logging.info(f"Loading route from: {route_file}")
    # Loading route from csv file
    if route_file[-3:] == "csv":
        df = pd.read_csv(route_file)
        to_wp = df['Name'].iloc[-1]
        from_wp = df['Name'].iloc[0]
    # Loading route from geojson file
    elif route_file[-4:] == "json":
        with open(route_file, "r") as f:
            route_json = json.load(f)
        route_coords = route_json['features'][0]['geometry']['coordinates']
        to_wp = route_json['features'][0]['properties']['to']
        from_wp = route_json['features'][0]['properties']['from']
        longs = [c[0] for c in route_coords]
        lats = [c[1] for c in route_coords]
        df = pd.DataFrame()
        df['Long'] = longs
        df['Lat'] = lats
    elif route_file[-3:] == "gpx":
        route_json = gpx_route_import(route_file)
        route_coords = route_json['features'][0]['geometry']['coordinates']
        to_wp = route_json['features'][0]['properties']['to']
        from_wp = route_json['features'][0]['properties']['from']
        longs = [c[0] for c in route_coords]
        lats = [c[1] for c in route_coords]
        df = pd.DataFrame()
        df['Long'] = longs
        df['Lat'] = lats
    else:
        logging.warning("Invalid route input! Please supply either a csv, gpx or geojson file with the route waypoints.")
        return None

    logging.info(f"Route start waypoint: {from_wp}")
    logging.info(f"Route end waypoint: {to_wp}")
    logging.debug(f"Route has {len(df)} waypoints")
    df['id'] = 1
    df['order'] = np.arange(len(df))

    return df, from_wp, to_wp


def load_mesh(mesh_file):
    """
        Load mesh from file into GeoDataFrame
        Args:
            mesh_file (str): Path to mesh with vehicle information

        Returns:
            mesh (GeoDataFrame): Mesh in GeoDataFrame format
    """
    logging.info(f"Loading mesh from: {mesh_file}")
    # Loading mesh information
    with open(mesh_file, 'r') as fp:
        info = json.load(fp)
    mesh = pd.DataFrame(info['cellboxes'])

    if (not any('uC' in cb for cb in mesh)) or (not any('vC' in cb for cb in mesh)):
        logging.info("No data for currents in mesh, setting default value to zero!")

    mesh['geometry'] = mesh['geometry'].apply(wkt.loads)
    mesh = gpd.GeoDataFrame(mesh, crs='EPSG:4326', geometry='geometry')

    region = info['config']['mesh_info']['region']
    region_poly = Polygon(((region['long_min'], region['lat_min']), (region['long_min'], region['lat_max']),
                           (region['long_max'], region['lat_max']), (region['long_max'], region['lat_min'])))

    return mesh, region_poly


def find_intersections(df, mesh):
    """
        Find crossing points of the route and the cells in the mesh
        Args:
            df (DataFrame): Route info in dataframe format
            mesh (GeoDataFrame): Mesh in GeoDataFrame format

        Returns:
            track_points (dict): Dictionary of crossing points and cell ids
    """

    track_waypoints = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Long'], df['Lat']))
    tracks = track_waypoints.sort_values(by=['order']).groupby(['id'])['geometry'].apply(
        lambda x: LineString(x.tolist()))
    tracks = gpd.GeoDataFrame(tracks, crs='EPSG:4326', geometry='geometry')

    line_segs_first_points = []
    line_segs_last_points = []
    line_segs_mid_points = []
    line_segs_cell_id = []

    # Find crossing points of the route and the cells in the mesh
    for idx in range(len(mesh)):
        # Skip cells with no intersection
        if not tracks['geometry'].iloc[0].intersects(mesh['geometry'].iloc[idx]):
            continue
        tp = tracks['geometry'].iloc[0].intersection(mesh['geometry'].iloc[idx])
        # Check for multi line strings in case of complex intersections
        if type(tp) == MultiLineString:
            pnts = []
            for l in tp.geoms:
                for c in l.coords:
                    pnts.append(Point(c))
        else:
            pnts = [Point(point) for point in tp.coords]
        if len(pnts) <= 1:
            continue
        line_segs_first_points.append(pnts[0])
        line_segs_mid_points.append(pnts[1:-1])
        line_segs_last_points.append(pnts[-1])
        line_segs_cell_id.append(idx)

    track_points = pd.DataFrame(
        {'startPoints': line_segs_first_points, 'midPoints': line_segs_mid_points, 'endPoints': line_segs_last_points,
         'cellID': line_segs_cell_id})

    return track_points


def order_track(df, track_points):
    """
        Order crossing points into a track along the route
        Args:
            df (DataFrame): Route info in dataframe format
            track_points (dict): Dictionary of crossing points and cell ids

        Returns:
            user_track (DataFrame): DataFrame of ordered crossing points and cell ids
    """

    start_point = Point(df.iloc[0]['Long'], df.iloc[0]['Lat'])
    end_point = Point(df.iloc[-1]['Long'], df.iloc[-1]['Lat'])

    pathing = True
    track_id = np.where(track_points['startPoints'] == start_point)[0][0]
    path_point = []
    cell_ids = []

    # Loop through crossing points to order them into a track along the route
    while pathing:
        start_point_segment = track_points['startPoints'].iloc[track_id]
        end_point_segment = track_points['endPoints'].iloc[track_id]
        path_point.append(start_point_segment)
        cell_ids.append(track_points['cellID'].iloc[track_id])

        if len(track_points['midPoints'].iloc[track_id]) != 0:
            for midpnt in track_points['midPoints'].iloc[track_id]:
                path_point.append(midpnt)
                cell_ids.append(track_points['cellID'].iloc[track_id])

        if  distance(end_point_segment,end_point) < 0.05:
            path_point.append(end_point_segment)
            cell_ids.append(track_points['cellID'].iloc[track_id])
            pathing = False
        else:
            track_id     = np.argmin([distance(entry,end_point_segment) for entry in track_points['startPoints']])
            track_misfit = min([distance(entry,end_point_segment) for entry in track_points['startPoints']])
            if track_misfit >= 0.05:
                raise Exception(f'Path Segment not adding - ID={track_id},Misfit={track_misfit},distance from'
                                f' end={distance(end_point_segment,end_point)}')

    user_track = pd.DataFrame({'Point': path_point, 'CellID': cell_ids})
    return user_track

def route_calc(route_file, mesh_file):
    """
        Function to calculate the fuel/time cost of a user defined route in a given mesh

        Args:
            route_file (str): Path to user defined route
            mesh_file (str): Path to mesh with vehicle information

        Returns:
            user_path (dict): User defined route in geojson format with calculated cost information
    """

    # Load route info and waypoint names from file
    df, from_wp, to_wp = load_route(route_file)

    # Load mesh info from file
    mesh, region_poly = load_mesh(mesh_file)

    # Check route waypoints contained in mesh bounds
    for idx in range(len(df)):
        if region_poly.contains(Point((df.iloc[idx]['Long'],df.iloc[idx]['Lat']))):
            continue
        else:
            logging.warning(f"Mesh does not contain waypoint located at Lat: {df.iloc[idx]['Lat']} "
                            f"Long: {df.iloc[idx]['Long']} !")
            return None

    # Find points where route crosses mesh
    track_points = find_intersections(df, mesh)

    # Loop through crossing points to order them into a track along the route
    user_track = order_track(df, track_points)
    logging.debug(f"Route has {len(user_track)} crossing points")

    # Initialise segment costs with zero values at start point of path
    traveltimes = [0.0]
    distances = [0.0]
    cellboxes = [mesh.iloc[user_track['CellID'].iloc[0]]]
    cases = [1]

    # Calculate cost of each segment in the path
    for idx in range(len(user_track)-1):
        start_point = np.array((user_track['Point'].iloc[idx].xy[0][0], user_track['Point'].iloc[idx].xy[1][0]))
        end_point = np.array((user_track['Point'].iloc[idx+1].xy[0][0], user_track['Point'].iloc[idx+1].xy[1][0]))
        cell_box = mesh.iloc[user_track['CellID'].iloc[idx]]
        case = case_from_angle(start_point, end_point)
        # Check for inaccessible cells on user defined route
        if cell_box['inaccessible']:
            logging.warning(f"This route crosses an inaccessible cell! Cell located at Lat: {cell_box['cy']} "
                         f"Long: {cell_box['cx']}")
            logging.info("Trying with speed and fuel from previous cells, reroute for more accurate results")
            i = 0
            # Go back along path to find previous accessible cell
            while cell_box['inaccessible']:
                i += 1
                cell_box = mesh.iloc[user_track['CellID'].iloc[idx-i]]

        traveltime_s, distance_m = traveltime_distance(cell_box, start_point, end_point, speed='speed', vector_x='uC',
                                                   vector_y='vC', case=case)
        traveltime = ((traveltime_s / 60) / 60) / 24
        segment_distance = distance_m / 1000
        traveltimes.append(traveltime)
        distances.append(segment_distance)
        cellboxes.append(cell_box)
        cases.append(case)


    logging.debug(f"Route crosses {len(set([c['id'] for c in cellboxes]))} different cellboxes")
    # Find cumulative values along path
    path_points = user_track['Point']
    path_traveltimes = np.cumsum(traveltimes)
    path_distances = np.cumsum(distances)
    path_fuels = [traveltimes[idx] * cellboxes[idx]['fuel'][np.where(case_indices==cases[idx])[0][0]] for idx in range(len(traveltimes))]
    path_fuel = np.cumsum(path_fuels)

    # Put path values into geojson format
    path_geojson = pd.DataFrame()
    path_geojson['geometry'] = [LineString(path_points)]
    path_geojson['traveltime'] = [path_traveltimes.tolist()]
    path_geojson['distance'] = [(path_distances * 1000).tolist()]
    path_geojson['fuel'] = [path_fuel.tolist()]
    path_geojson['to'] = to_wp
    path_geojson['from'] = from_wp
    path_geojson = gpd.GeoDataFrame(path_geojson, crs='EPSG:4326', geometry='geometry')

    user_path = json.loads(path_geojson.to_json())

    return user_path