import json
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from shapely.geometry import LineString


def traveltime_in_cell(xdist, ydist, u, v, s):
    """
        Calculate travel time in cell.
    """
    dist = np.sqrt(xdist ** 2 + ydist ** 2)
    cval = np.sqrt(u ** 2 + v ** 2)

    dotprod = xdist * u + ydist * v
    diffsqrs = s ** 2 - cval ** 2

    # if (dotprod**2 + diffsqrs*(dist**2) < 0)
    if diffsqrs == 0.0:
        if dotprod == 0.0:
            return np.inf
            # raise Exception(' ')
        else:
            if ((dist ** 2) / (2 * dotprod)) < 0:
                return np.inf
                # raise Exception(' ')
            else:
                traveltime = dist * dist / (2 * dotprod)
                return traveltime

    traveltime = (np.sqrt(dotprod ** 2 + (dist ** 2) * diffsqrs) - dotprod) / diffsqrs
    if traveltime < 0:
        traveltime = np.inf
    return traveltime, dist


def traveltime_distance(cellbox, wp, cp, speed='speed', vector_x='uC', vector_y='vC'):
    """
        Calculate travel time and distance
    """
    case = 0
    m_long = 111.321 * 1000
    m_lat = 111.386 * 1000
    x = (cp[0] - wp[0]) * m_long * np.cos(wp[1] * (np.pi / 180))
    y = (cp[1] - wp[1]) * m_lat
    su = cellbox[vector_x]
    sv = cellbox[vector_y]
    ssp = cellbox[speed][case] * (1000 / (60 * 60))
    traveltime, distance = traveltime_in_cell(x, y, su, sv, ssp)
    return traveltime, distance


def route_calc(route_file, mesh_file):
    """
    Function to calculate the fuel/time cost of a user defined route in a given mesh
    Args:
        route_file (str): Path to user defined route
        mesh_file (str): Path to mesh with vehicle information

    Returns:
        user_path (dict): User defined route in geojson format with calculated cost information

    """
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
    else:
        logging.warning("Invalid route input! Please supply either a csv or geojson file with the route waypoints.")
        return None

    df['id'] = 1
    df['order'] = np.arange(len(df))

    track_waypoints = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Long'], df['Lat']))
    tracks = track_waypoints.sort_values(by=['order']).groupby(['id'])['geometry'].apply(
        lambda x: LineString(x.tolist()))
    tracks = gpd.GeoDataFrame(tracks, crs='EPSG:4326', geometry='geometry')


    # Loading mesh information
    with open(mesh_file, 'r') as fp:
        info = json.load(fp)
    mesh = pd.DataFrame(info['cellboxes'])
    mesh['geometry'] = mesh['geometry'].apply(wkt.loads)
    mesh = gpd.GeoDataFrame(mesh, crs='EPSG:4326', geometry='geometry')

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

    start_point = Point(df.iloc[0]['Long'], df.iloc[0]['Lat'])
    end_point = Point(df.iloc[-1]['Long'], df.iloc[-1]['Lat'])

    pathing = True
    track_id = np.where(track_points['startPoints'] == start_point)[0][0]
    path_point = []
    cell_ids = []

    # Loop through crossing points to order them into a track along the route
    while pathing:
        try:
            start_point_segment = track_points['startPoints'].iloc[track_id]
            end_point_segment = track_points['endPoints'].iloc[track_id]
            path_point.append(start_point_segment)
            cell_ids.append(track_points['cellID'].iloc[track_id])

            if len(track_points['midPoints'].iloc[track_id]) != 0:
                for midpnt in track_points['midPoints'].iloc[track_id]:
                    path_point.append(midpnt)
                    cell_ids.append(track_points['cellID'].iloc[track_id])

            if end_point_segment == end_point:
                pathing = False
            track_id = np.where(track_points['startPoints'] == end_point_segment)[0][0]
        except IndexError:
            pathing = False
            path_point.append(end_point_segment)
            cell_ids.append('NaN')

    user_track = pd.DataFrame({'Point': path_point, 'CellID': cell_ids})

    # Initialise segment costs with zero values at start point of path
    traveltimes = [0.0]
    distances = [0.0]
    cellboxes = [mesh.iloc[user_track['CellID'].iloc[0]]]

    # Calculate cost of each segment in the path
    for idx in range(len(user_track)-1):
        start_point = np.array((user_track['Point'].iloc[idx].xy[0][0], user_track['Point'].iloc[idx].xy[1][0]))
        end_point = np.array((user_track['Point'].iloc[idx+1].xy[0][0], user_track['Point'].iloc[idx+1].xy[1][0]))
        cell_box = mesh.iloc[user_track['CellID'].iloc[idx]]
        # Check for inaccessible cells on user defined route
        if cell_box['inaccessible']:
            logging.warning(f"This route crosses an inaccessible cell! Cell located at Lat: {cell_box['cy']} "
                         f"Long: {cell_box['cx']}. Please reroute around it.")
            return None

        traveltime_s, distance_m = traveltime_distance(cell_box, start_point, end_point, speed='speed', vector_x='uC',
                                                   vector_y='vC')
        traveltime = ((traveltime_s / 60) / 60) / 24
        distance = distance_m / 1000
        traveltimes.append(traveltime)
        distances.append(distance)
        cellboxes.append(cell_box)

    # Find cumulative values along path
    path = pd.DataFrame(cellboxes).reset_index(drop=True)
    path['path_points'] = user_track['Point']
    path['path_traveltimes'] = np.cumsum(traveltimes)
    path['path_distances'] = np.cumsum(distances)
    path_fuels = [traveltimes[idx] * path['fuel'][idx][0] for idx in range(len(traveltimes))]
    path['path_fuel'] = np.cumsum(path_fuels)

    # Put path values into geojson format
    path_geojson = pd.DataFrame()
    path_geojson['geometry'] = [LineString(path['path_points'])]
    path_geojson['traveltime'] = [path['path_traveltimes'].tolist()]
    path_geojson['distance'] = [(path['path_distances'] * 1000).tolist()]
    path_geojson['fuel'] = [path['path_fuel'].tolist()]
    path_geojson['to'] = to_wp
    path_geojson['from'] = from_wp
    path_geojson = gpd.GeoDataFrame(path_geojson, crs='EPSG:4326', geometry='geometry')

    user_path = json.loads(path_geojson.to_json())

    return user_path