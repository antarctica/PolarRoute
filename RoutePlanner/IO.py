from netCDF4 import Dataset
import numpy as np
import pandas as pd
from glob import glob
import json
import numpy as np

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def LoadIcePoints(NetCDF,startDate,endDate):
    bsos = Dataset(NetCDF)

    Dates = pd.to_datetime('2012-12-01') + pd.to_timedelta(bsos['time'][:],unit='S')

    DateRange     = [startDate,endDate]
    DateRangeIndx = [np.argmin(abs(Dates - pd.to_datetime(DateRange[0]))),np.argmin(abs(Dates - pd.to_datetime(DateRange[1])))] 

    XC,YC = np.meshgrid(bsos['XC'][:].data,bsos['YC'][:].data)

    icePoints = pd.DataFrame({'time':pd.to_datetime(DateRangeIndx[0]),
                            'long':XC.flatten(),'lat':YC.flatten(),
                            'iceArea':np.mean(bsos['SIarea'][DateRangeIndx[0]:DateRangeIndx[1],...].data,axis=0).flatten(),
                            'depth':bsos['Depth'][:,:].data.flatten()})
    return icePoints

def LoadCurrentPoints(NetCDF):
    sose = Dataset(NetCDF)
    currentPoints = pd.DataFrame({'long':sose['lon'][...].data.flatten(),'lat':sose['lat'][...].data.flatten(),'uC':sose['uC'][...].data.flatten(),'vC':sose['vC'][...].data.flatten()})
    currentPoints['time'] = ''
    return currentPoints


def GreatCircle(Start_p,End_p):
    import pyproj
    import numpy as np
    startlong, startlat = Start_p
    endlong, endlat     = End_p
    startlong = startlong-360
    endlong   = endlong-360

    # calculate distance between points
    g = pyproj.Geod(ellps='WGS84')
    (az12, az21, dist) = g.inv(startlong, startlat, endlong, endlat)

    # calculate line string along path with segments <= 1 km
    lonlats = g.npts(startlong, startlat, endlong, endlat,
                    1 + int(dist / 1000))

    lonlats = np.array(lonlats)
    lonlats[:,0] = lonlats[:,0]+360

    return lonlats


def SDAPosition(PATH):
    #PATH = '/Users/jsmith/Documents/Research/Researcher_BAS/RoutePlanning/SDADT-Positions'
    pts = [];tms = [];hding=[]
    for file in glob('{}/*.json'.format(PATH)):
        try :
            data = json.load(open(file))
            for pt in data['features']:
                pts.append(pt['geometry']['coordinates'])
                tms.append(pt['properties']['timestamp'])
                hding.append(pt['properties']['NavigationHeading'])
        except:
            continue
    tms = pd.to_datetime(tms)
    pts = np.array(pts)

    Path = pd.DataFrame({'Time':tms,'Long':pts[:,0]+360,'Lat':pts[:,1],'Heading':hding})
    Path = Path[(Path['Long'] != 0) & (Path['Lat'] != 0)]
    Path = Path.sort_values('Time').reset_index(drop=True)
    return Path



#Paths to GeoJSON
import numpy as np
import copy

def PathsJSON(Paths):
    GeoJSON = {}
    GeoJSON['type'] = "FeatureCollection"
    GeoJSON['features'] = []
    Pths = copy.deepcopy(Paths)
    for path in Pths:
        if np.isinf(path['Time']):
            continue
        
        points = path['Path']['Points']
        points[:,0] = points[:,0]-360

        pt = {}
        pt['type']     = 'Feature'
        pt['geometry'] = {}
        pt['geometry']['type'] = 'LineString'
        pt['geometry']['coordinates'] = path['Path']['Points'].tolist()
        GeoJSON['features'].append(pt)
    return GeoJSON

def JSON2Paths(file):
    import json
    import numpy as np
    with open(file) as json_file:
        data = json.load(json_file)

    Paths = []
    for feature in data['features']:
        path = {}
        path['from'] = feature['properties']['from']
        path['to']   = feature['properties']['to']
        path['Time'] = feature['properties']['Travel Time (d)']


        points = np.array(feature['geometry']['coordinates'])
        points[:,0] = points[:,0]+360
        path['Path'] ={}
        path['Path']['Points'] =points
        Paths.append(path)
    return Paths
    
def WaypointsJSON(Waypoints):
    GeoJSON = {}
    GeoJSON['type'] = "FeatureCollection"
    GeoJSON['features'] = []
    for idx,wpt in Waypoints.iterrows():
        pt = {}
        pt['type']     = 'Feature'
        pt['geometry'] = {}
        pt['geometry']['type'] = 'Point'

        loc = wpt[['Long','Lat']]
        loc[0] = loc[0]-360
        pt['geometry']['coordinates'] = loc.tolist()
        pt['properties']={}
        pt['properties']['name'] = wpt['Name']
        GeoJSON['features'].append(pt)  
    return GeoJSON

def MeshJSON(cellGrid):
    GeoJSON = {}
    GeoJSON['type'] = "FeatureCollection"
    GeoJSON['features'] = []
    for ii in range(len(cellGrid.cellBoxes)):
        bounds      = np.array(cellGrid.cellBoxes[ii].getBounds())
        bounds[:,0] = bounds[:,0]-360
        pt = {}
        pt['type']     = 'Feature'
        pt['geometry'] = {}
        pt['geometry']['type'] = 'Polygon'
        pt['geometry']['coordinates'] = bounds.tolist()
        GeoJSON['features'].append(pt)  
    return GeoJSON
    

from RoutePlanner.CellBox import CellBox
def MeshDF(cellGrid):
    from shapely.geometry import Polygon
    import geopandas as gpd
    Shape   = []; IceArea = []; IsLand  = []; dpth=[];vec=[]; CentroidCx=[];CentroidCy=[];Index=[]
    for idx,c in enumerate(cellGrid.cellBoxes):
        if isinstance(c, CellBox):
            bounds = np.array(c.getBounds())
            Shape.append(Polygon(bounds))
            IceArea.append(c.iceArea()*100)
            if cellGrid._j_grid:
                IsLand.append(c.isLandM())
            else:
                IsLand.append(c.containsLand())
            dpth.append(c.depth())
            vec.append([c.getuC(),c.getvC()])
            CentroidCx.append(c.cx)
            CentroidCy.append(c.cy)
            Index.append(int(idx))
    Polygons = pd.DataFrame()
    Polygons['geometry'] = Shape
    Polygons['Ice Area'] = IceArea
    Polygons['Land']     = IsLand
    Polygons['Cx']       = CentroidCx
    Polygons['Cy']       = CentroidCy
    Polygons['Vector']   = vec
    Polygons['Depth']    = dpth
    Polygons['Index']    = Index
    Polygons = gpd.GeoDataFrame(Polygons,crs={'init': 'epsg:4326'}, geometry='geometry')
    Polygons['Land'][np.isnan(Polygons['Ice Area'])] = True

    Polygons['Land'][Polygons['Cy'] < -78.0] = True
    return Polygons