import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from shapely.geometry import LineString
import logging

def traveltime_in_cell(xdist, ydist, U, V, S):
    """
        Calculate travel time in cell.
    """
    dist = np.sqrt(xdist ** 2 + ydist ** 2)
    cval = np.sqrt(U ** 2 + V ** 2)

    dotprod = xdist * U + ydist * V
    diffsqrs = S ** 2 - cval ** 2

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


def traveltime_distance(cellBox, Wp, Cp, Speed='speed', Vector_x='uC', Vector_y='vC'):
    """
        Calculate travel time and distance
    """
    case = 0
    m_long = 111.321 * 1000
    m_lat = 111.386 * 1000
    x = (Cp[0] - Wp[0]) * m_long * np.cos(Wp[1] * (np.pi / 180))
    y = (Cp[1] - Wp[1]) * m_lat
    Su = cellBox[Vector_x]
    Sv = cellBox[Vector_y]
    Ssp = cellBox[Speed][case] * (1000 / (60 * 60))
    traveltime, distance = traveltime_in_cell(x, y, Su, Sv, Ssp)
    return traveltime, distance

def route_calc(route_file, mesh_file):
    df = pd.read_csv(route_file)
    df['id'] = 1
    df['order'] = np.arange(len(df))

    track_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Long'], df['Lat']))
    tracks = track_points.sort_values(by=['order']).groupby(['id'])['geometry'].apply(lambda x: LineString(x.tolist()))
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

    for idx in range(len(mesh)):
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

    user_track = pd.DataFrame(
        {'startPoints': line_segs_first_points, 'midPoints': line_segs_mid_points, 'endPoints': line_segs_last_points,
         'cellID': line_segs_cell_id})

    start_point = Point(df.iloc[0]['Long'], df.iloc[0]['Lat'])
    end_point = Point(df.iloc[-1]['Long'], df.iloc[-1]['Lat'])

    pathing = True
    track_id = np.where(user_track['startPoints'] == start_point)[0][0]
    path_point = []
    cell_ids = []
    while pathing:
        try:
            start_point_segment = user_track['startPoints'].iloc[track_id]
            end_point_segment = user_track['endPoints'].iloc[track_id]
            path_point.append(start_point_segment)
            cell_ids.append(user_track['cellID'].iloc[track_id])

            if len(user_track['midPoints'].iloc[track_id]) != 0:
                for midpnt in user_track['midPoints'].iloc[track_id]:
                    path_point.append(midpnt)
                    cell_ids.append(user_track['cellID'].iloc[track_id])

            if end_point_segment == end_point:
                pathing = False
            track_id = np.where(user_track['startPoints'] == end_point_segment)[0][0]
        except:
            pathing = False
            path_point.append(end_point_segment)
            cell_ids.append('NaN')

    user_track = pd.DataFrame({'Point': path_point, 'CellID': cell_ids})
    track_line_string = LineString(user_track['Point'])

    traveltimes = list()
    distances = list()
    cellboxes = list()

    cellboxes.append(mesh.iloc[user_track['CellID'].iloc[0]])
    traveltimes.append(0.0)
    distances.append(0.0)
    for idx in range(len(user_track)-1):
        start_point = np.array((user_track['Point'].iloc[idx].xy[0][0], user_track['Point'].iloc[idx].xy[1][0]))
        end_point = np.array((user_track['Point'].iloc[idx+1].xy[0][0], user_track['Point'].iloc[idx+1].xy[1][0]))
        cell_box = mesh.iloc[user_track['CellID'].iloc[idx]]
        traveltime, distance = traveltime_distance(cell_box, start_point, end_point, Speed='speed', Vector_x='uC',
                                                   Vector_y='vC')
        traveltime = ((traveltime / 60) / 60) / 24
        distance = distance / 1000
        traveltimes.append(traveltime)
        distances.append(distance)
        cellboxes.append(cell_box)

    path = pd.DataFrame(cellboxes).reset_index(drop=True)
    path['path_points'] = user_track['Point']
    path['path_traveltimes'] = np.cumsum(traveltimes)
    path['path_distances'] = np.cumsum(distances)
    path_fuels = []
    for idx in range(len(traveltimes)):
        path_fuels.append(traveltimes[idx] * path['fuel'][idx][0])
    path['path_fuel'] = np.cumsum(path_fuels)

    path_geojson = pd.DataFrame()  # path[['path_points','path_traveltimes','path_distances','path_fuel']]
    path_geojson['geometry'] = [LineString(path['path_points'])]
    path_geojson['traveltime'] = [path['path_traveltimes'].tolist()]
    path_geojson['distance'] = [(path['path_distances'] * 1000).tolist()]
    path_geojson['fuel'] = [path['path_fuel'].tolist()]
    path_geojson['to'] = df['Name'].iloc[-1]
    path_geojson['from'] = df['Name'].iloc[0]
    path_geojson = gpd.GeoDataFrame(path_geojson, crs='EPSG:4326', geometry='geometry')
    return json.loads(path_geojson.to_json())