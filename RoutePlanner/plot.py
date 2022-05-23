import copy
import json
import pandas as pd
import numpy as np
import sys
from branca.colormap import linear
import folium
from shapely import wkt
import geopandas as gpd
from folium import plugins

sys.path.insert(0, 'folium')
sys.path.insert(0, 'branca')

import branca
import folium

# Adapted from: https://nbviewer.org/gist/BibMartin/f153aa957ddc5fadc64929abdee9ff2e
from branca.element import MacroElement
from jinja2 import Template

class BindColormap(MacroElement):
    """Binds a colormap to a given layer.

    Parameters
    ----------
    colormap : branca.colormap.ColorMap
        The colormap to bind.
    """
    def __init__(self, layer, colormap):
        super(BindColormap, self).__init__()
        self.layer = layer
        self.colormap = colormap
        self._template = Template(u"""
        {% macro script(this, kwargs) %}
            {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
            {{this._parent.get_name()}}.on('layeradd', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
                }});
            {{this._parent.get_name()}}.on('layerremove', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'none';
                }});
        {% endmacro %}
        """)  # noqa


class InteractiveMap:
    def __init__(self,config):
        self.config = config

        self.basemap = config["Interactive_Map"]['Basemap_Info']
        self.layers  = config['Interactive_Map']['Layers']


        # Initialising the basemap
        self._basemap()


        # Defining the layers to plot
        for layer in self.layers:
            # try:
                if layer['Type'] == 'Paths':
                    self._paths(layer)
                if layer['Type'] == 'Maps':
                    self._maps(layer) 
                if layer['Type'] == 'Points':
                    self._points(layer)
                if layer['Type'] == 'MeshInfo':
                    self._meshInfo(layer)

            # except:
            #     print('Issue Plotting Layer')
                    
        # Adding in the layer control
        self._layer_control()

        # Save
        if 'Output_Filename' in self.config['Interactive_Map']:
            self.map.save(self.config['Interactive_Map']['Output_Filename'])


    def _basemap(self):
        '''
            Defining the basemap structure
        '''
        title_html = '''
            <h1 style="color:#003b5c;font-size:16px">
            &ensp;<img src='https://i.ibb.co/JH2zknX/Small-Logo.png' alt="BAS-colour-eps" border="0" style="width:40px;height:40px;"> 
            <img src="https://i.ibb.co/XtZdzDt/BAS-colour-eps.png" alt="BAS-colour-eps" border="0" style="width:179px;height:40px;"> 
            &ensp;|&ensp; RoutePlanner &ensp;|&ensp; {}
            </h1>
            </body>
            '''.format(self.basemap['Title'])   
        self.map = folium.Map(location=self.basemap['Map_Centre'],zoom_start=self.basemap['Map_Zoom'],tiles=None)
        
        # 
        bsmap = folium.FeatureGroup(name='BaseMap')
        folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}.png',attr="toner-bcg", name='Basemap').add_to(bsmap)
        bsmap.add_to(self.map)
        bsmap = folium.FeatureGroup(name='Dark BaseMap',show=False)
        folium.TileLayer(tiles="https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png",attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',name='darkmatter').add_to(bsmap)
        bsmap.add_to(self.map)



        self.map.get_root().html.add_child(folium.Element(title_html))
        
    def _paths(self,info):
        '''
            Plotting a paths type object
        '''

        #map,PathPoints=True,PathName='Transit Time', colorLine=True
        
        # Defining the feature groups to add
        pths        = folium.FeatureGroup(name='{}'.format(info['Name']),show=info['Show'])
        if info['Path_Points']:
            pths_points = folium.FeatureGroup(name='{} - Path Points'.format(info['Name']),show=info['Show'])
            
        # Loading the path information
        with open('paths_traveltime.json', 'r') as f:
            geojson = json.load(f)
            paths = geojson['features']

        # Determining max values of all paths
        if info['Colorline']:
            max_val = 0
            for path in copy.deepcopy(paths):
                if np.array(path['properties'][info['Data_Name']]).max() > max_val:
                    max_val = np.array(path['properties'][info['Data_Name']]).max()

        # Determining max travel-times of all paths
        for path in paths:
            points   = np.array(path['geometry']['coordinates'])
            if 'Data_Name' in info:
                data_val = np.array(path['properties'][info['Data_Name']])
            else:
                data_val = np.array(len(points))

            points[:,0] = points[:,0]
            points = points[:,::-1]

            if info['Colorline']:
                folium.PolyLine(points,color="black", weight=info['Line_Width'], opacity=1).add_to(pths)
                colormap = linear._colormaps[info['Cmap']].scale(0,max_val)
                colormap.caption = '{} ({})'.format(info['Data_Name'],info['Data_Units'])
                folium.ColorLine(points,data_val,colormap=colormap,nb_steps=50, weight=3.5, opacity=1).add_to(pths)



            else:
                folium.PolyLine(points,color=info['Color'], weight=info['Line_Width'], opacity=1).add_to(pths)


            if info['Path_Points']:
                for idx in range(len(points)):
                    loc = [points[idx,0],points[idx,1]]
                    folium.Marker(
                        location=loc,
                        icon=folium.plugins.BeautifyIcon(icon='circle',
                                                    border_color='transparent',
                                                    background_color='transparent',
                                                    border_width=1,
                                                    text_color=info['Color'],
                                                    inner_icon_style='margin:0px;font-size:0.8em')
                    ).add_to(pths_points)
        
        if info['Path_Points']:
            pths_points.add_to(self.map)
        if info['Colorline']:
            self.map.add_child(pths)
            self.map.add_child(colormap)
            self.map.add_child(BindColormap(pths,colormap))
        else:
            pths.add_to(self.map)


    def _points(self,info):

        wpts = folium.FeatureGroup(name='{}'.format(info['Name']),show=info['Show'])
        wpts_name = folium.FeatureGroup(name='{} - Names'.format(info['Name']),show=info['Show'])

        dataframe_points = pd.read_csv(info['filename'])


        for id,wpt in dataframe_points.iterrows():
            loc = [wpt['Lat'], wpt['Long']]
            folium.Marker(
                location=loc,
                icon=plugins.BeautifyIcon(icon='circle',
                                                border_color='transparent',
                                                background_color='transparent',
                                                border_width=info['Border_Width'],
                                                text_color=info['Color'],
                                                inner_icon_style='margin:0px;font-size:0.8em'),
                popup="<b>{} [{:4f},{:4f}]</b>".format(wpt['Name'],loc[0],loc[1]),
            ).add_to(wpts)    

            folium.Marker(
                        location=loc,
                            icon=folium.features.DivIcon(
                                icon_size=(250,36),
                                icon_anchor=(0,0),
                                html='<div style="font-size: {}pt">{}</div>'.format(info['Font_Size'],wpt['Name']),
                                ),
            ).add_to(wpts_name)


        wpts.add_to(self.map)
        wpts_name.add_to(self.map)


    def _maps(self,info):
        '''
            Plotting a map type object
        '''

        dataframe_pandas = pd.read_csv(info['filename'])
        dataframe_pandas['geometry'] = dataframe_pandas['geometry'].apply(wkt.loads)
        dataframe_geo = gpd.GeoDataFrame(dataframe_pandas,crs='EPSG:4326', geometry='geometry')


        feature_info = folium.FeatureGroup(name='{}'.format(info['Name']),show=info['Show'])

        if dataframe_geo[info['Data_Name']].dtype == 'bool':
            folium.GeoJson(
                dataframe_geo[dataframe_geo[info['Data_Name']]==True],
                style_function=lambda x: {
                    'fillColor': info['Fill_Color'],
                    'color': info['Line_Color'],
                    'weight': info['Line_Width'],
                    'fillOpacity': info['Fill_Opacity']
                    }
            ).add_to(feature_info)

            feature_info.add_to(self.map)

        if dataframe_geo[info['Data_Name']].dtype == float:

            if ('Fill_trim_min' not in info):
                info['Fill_trim_min'] = dataframe_geo[info['Data_Name']].min()
            if ('Fill_trim_max' not in info):
                info['Fill_trim_max'] = dataframe_geo[info['Data_Name']].max()            

            dataframe_geo = dataframe_geo[(dataframe_geo[info['Data_Name']] >= info['Fill_trim_min']) &
                                        (dataframe_geo[info['Data_Name']] <= info['Fill_trim_max'])]

            if info['Fill_cmap_use']:

                if 'Fill_cmap_opacity' not in info:
                    info['Fill_cmap_opacity'] = False

                if info['Fill_cmap_opacity']:
                    folium.GeoJson(
                        dataframe_geo,
                        style_function=lambda x: {
                                'fillColor': info['Fill_Color'],
                                'color': info['Line_Color'],
                                'weight': info['Line_Width'],
                                'fillOpacity': x['properties'][info['Data_Name']]/info['Fill_cmap_opacity_scalar']
                            }
                    ).add_to(feature_info)
                    self.map.add_child(feature_info)

                else:
                    if 'Fill_cmap_min' in info:
                        cmin = info['Fill_cmap_min']
                    else:
                        cmin = dataframe_geo[info['Data_Name']].min()
                    if 'Fill_cmap_max' in info:
                        cmax = info['Fill_cmap_max']
                    else:
                        cmax = dataframe_geo[info['Data_Name']].max()

                    colormap = linear._colormaps[info['Cmap']].scale(cmin,cmax)
                    colormap.caption = '{} ({})'.format(info['Data_Name'],info['Data_Units'])
                    folium.GeoJson(
                        dataframe_geo,
                        style_function=lambda x: {
                                'fillColor': colormap(x['properties'][info['Data_Name']]),
                                'color': info['Line_Color'],
                                'weight': info['Line_Width'],
                                'fillOpacity': info['Fill_Opacity']
                            }
                    ).add_to(feature_info)
                    self.map.add_child(feature_info)
                    self.map.add_child(colormap)
                    self.map.add_child(BindColormap(feature_info,colormap))
            else:
                folium.GeoJson(
                    dataframe_geo,
                    style_function=lambda x: {
                        'fillColor': info['Fill_Color'],
                        'color': info['Line_Color'],
                        'weight': info['Line_Width'],
                        'fillOpacity': info['Fill_Opacity']
                        }
                ).add_to(feature_info)

                self.map.add_child(feature_info)

    #   {
    #     "Type": "MeshInfo",
    #     "Show":false,
    #     "Name": "Mesh Info",
    #     "filename": "./cellgrid_dataframe.csv",
    #     "Line_Width": 0.3,
    #     "Line_Color": "gray",
    #     "Fill_Color": "white",
    #     "Fill_Opacity": 0.0,
    #     "Data_Names":["cell_info"]
    #   },

    def _meshInfo(self,info):
        dataframe_pandas = pd.read_csv(info['filename'])
        dataframe_pandas['geometry'] = dataframe_pandas['geometry'].apply(wkt.loads)
        dataframe_geo = gpd.GeoDataFrame(dataframe_pandas,crs='EPSG:4326', geometry='geometry')


        feature_info = folium.FeatureGroup(name='{}'.format(info['Name']),show=info['Show'])

        folium.GeoJson(dataframe_geo,

            style_function= lambda x: {
                    'fillColor': 'black',
                    'color': 'gray',
                    'weight': 0.5,
                    'fillOpacity': 0.
                },

            tooltip=folium.GeoJsonTooltip(
                fields=[dataframe_geo.columns.intersection(info['Data_Names'])],
                aliases=[dataframe_geo.columns.intersection(info['Data_Names'])],
                localize=True
            )
        ).add_to(feature_info) 
        self.map.add_child(feature_info)

    def _layer_control(self):
        folium.LayerControl(collapsed=True).add_to(self.map)

    def show(self):
        return self.map


# def MapTimeMesh(map,Polygons):
#     # ==== Plotting Ice ==== 
#     colormap = linear.Paired_06.scale(0,120).to_step(12)
#     colormap.caption = 'Fuel Usage (Tonnes)'
#     iceInfo = folium.FeatureGroup(name='Fuel Usage')
#     folium.GeoJson(
#         Polygons,
#         style_function=lambda x: {
#                 'fillColor': colormap(x['properties']['Fuel']),
#                 'color': 'gray',
#                 'weight': 0.5,
#                 'fillOpacity': 0.5

#             }
#     ).add_to(iceInfo)
#     iceInfo.add_to(map)
#     map.add_child(colormap)
#     return map

# import matplotlib.pylab as plt
# from matplotlib.patches import Polygon
# import numpy as np
# #from simplekml import Kml, Color, Style
# from RoutePlanner.IO import SDAPosition, MeshDF
# from RoutePlanner.CellBox import CellBox
# import folium
# import pandas as pd

# #  =============
# from branca.colormap import linear
# import folium
# import copy
# from folium.plugins import TimestampedGeoJson

# # Example Maps http://leaflet-extras.github.io/leaflet-providers/preview/
# #icons can be found at https://fontawesome.com/icons

# def MapJavaPaths(file,TT,map,color='blue',PathPoints=True):
#     import json
#     import numpy as np
#     import folium
#     import copy    

#     with open(file, 'r') as f:
#         JavaGeo = json.load(f)['allpaths']['January']

#     PathPts = []
#     for path in JavaGeo:
#         if any(path['from'] in s for s in TT.source_waypoints) and any(path['to'] in s for s in TT.end_waypoints):
#             pts=[]
#             for jj in path['path']:
#                 try:
#                     pts.append([jj['lon'],-jj['lat']])
#                 except:
#                     continue
#             pts = np.array(pts)
#             PathPts.append(pts)


#     pathMap        = folium.FeatureGroup(name='Java Paths')
#     for path in copy.deepcopy(PathPts):
#         points = path
#         points = points[:,::-1]
#         folium.PolyLine(points,color=color, weight=2.0, opacity=1).add_to(pathMap)
#     pathMap.add_to(map)

#     return map

# def MapWaypoints(DF,map,color='k'):
#     wpts = folium.FeatureGroup(name='WayPoints')
#     wpts_name = folium.FeatureGroup(name='WayPoint Names',show=False)
#     for id,wpt in DF.iterrows():
#         loc = [wpt['Lat'], wpt['Long']]
#         folium.Marker(
#             location=loc,
#             icon=folium.plugins.BeautifyIcon(icon='circle',
#                                             border_color='transparent',
#                                             background_color='transparent',
#                                             border_width=1,
#                                             text_color=color,
#                                             inner_icon_style='margin:0px;font-size:0.8em'),
#             popup="<b>{} [{},{}]</b>".format(wpt['Name'],loc[0],loc[1]),
#         ).add_to(wpts)    

#         folium.Marker(
#                     location=loc,
#                         icon=folium.features.DivIcon(
#                             icon_size=(250,36),
#                             icon_anchor=(0,0),
#                             html='<div style="font-size: 10pt">{}</div>'.format(wpt['Name']),
#                             ),
#          ).add_to(wpts_name)


#     wpts.add_to(map)
#     wpts_name.add_to(map)
#     return map


# def MapResearchSites(DF,map):
#     wpts = folium.FeatureGroup(name='Research Stations')
#     for id,wpt in DF.iterrows():
#         loc = [wpt['Lat'], wpt['Long']]
#         folium.Marker(
#             location=loc,
#             name='Research Sites',
#             icon=folium.plugins.BeautifyIcon(icon='circle',
#                                             border_color='transparent',
#                                             background_color='transparent',
#                                             border_width=1,
#                                             text_color='orange',
#                                             inner_icon_style='margin:0px;font-size:0.8em'),
#             popup="<b>{} [{},{}]</b>".format(wpt['Name'],loc[0],loc[1])
#         ).add_to(wpts)
#     wpts.add_to(map)    
#     return map

# def MapPaths(Paths,map,PathPoints=True,PathName='Transit Time', colorLine=True):
#     Pths        = folium.FeatureGroup(name='{}'.format(PathName))
#     Pths_points = folium.FeatureGroup(name='{} - Path Points'.format(PathName))

#     # Determining max travel-times of all paths
#     maxTT = 0
#     for path in copy.deepcopy(Paths):
#         if path['Time'] > maxTT:
#             maxTT = path['Time']

#     for path in copy.deepcopy(Paths):
#         points = path['Path']['Points']
#         Times  = path['Path']['Time']
#         points[:,0] = points[:,0]
#         points = points[:,::-1]

#         folium.PolyLine(points,color="black", weight=5.0, opacity=1).add_to(Pths)
#         if colorLine:
#             colormap = linear.viridis.scale(0,maxTT).to_step(100)
#             colormap.caption = 'Transit Time (Days)'
#             colormap.background = 'white'
#             folium.ColorLine(points,Times,colormap=colormap,nb_steps=50, weight=3.5, opacity=1).add_to(Pths)
#         for idx in range(len(points)):
#             loc = [points[idx,0],points[idx,1]]
#             folium.Marker(
#                 location=loc,
#                 icon=folium.plugins.BeautifyIcon(icon='circle',
#                                             border_color='transparent',
#                                             background_color='transparent',
#                                             border_width=1,
#                                             text_color='black',
#                                             inner_icon_style='margin:0px;font-size:0.8em')
#             ).add_to(Pths_points)
#     Pths.add_to(map)
#     if PathPoints:
#         Pths_points.add_to(map)

#     if colorLine:
#         map.add_child(colormap)

#     return map

# def TimeMapPaths(Paths,map,starttime='2014-01-01T00:00:00'):
#     lines=[]
#     for Path in copy.deepcopy(Paths):
#         Points = Path['Path']['Points']
#         Times  = pd.to_datetime(starttime) + pd.to_timedelta(Path['Path']['Time'],unit='D')

#         entry = {}
#         entry['coordinates'] = Points.tolist()
#         entry['dates'] = Times.strftime('%Y-%m-%dT%H:%M:%S').tolist()
#         entry['color'] = 'black'
#         lines.append(entry)


#     features = [
#         {
#             "type": "Feature",
#             "geometry": {
#                 "type": "LineString",
#                 "coordinates": line["coordinates"],
#             },
#             "properties": {
#                 "times": line["dates"],
#                 "style": {
#                     "color": line["color"],
#                     "weight": line["weight"] if "weight" in line else 3,
#                     'color': 'red'
#                 },
#                 'icon': 'circle',
#                 'iconstyle': {'color': 'red','iconSize': [1,1]}
#             },
#         }
#         for line in lines
#     ]

#     TimestampedGeoJson(
#         {
#             "type": "FeatureCollection",
#             "features": features,
#         },
#         period="PT1H",
#         auto_play=False,
#         add_last_point=True,
#         max_speed=100,
#         min_speed=5
#     ).add_to(map)
#     return map

# def TimeMapSDA(PATH,map):
#     #'/Users/jsmith/Documents/Research/Researcher_BAS/RoutePlanning/SDADT-Positions'
#     Info = SDAPosition(PATH)
#     lines=[]
#     Points = Info[['Long','Lat']].to_numpy()
#     Points[:,0] = Points[:,0]-360
#     entry = {}
#     entry['coordinates'] = Points.tolist()
#     entry['dates'] = Info['Time'].dt.strftime('%Y-%m-%dT%H:%M:%S').to_list()
#     entry['color'] = 'blue'
#     lines.append(entry)


#     TMS = Info['Time'].dt.strftime('%Y-%m-%dT%H:%M:%S').to_list()
#     pointfeatures = [
#         {
#             "type": "Feature",
#             "geometry": {
#                 "type": "Point",
#                 "coordinates": pt.tolist(),
#             },
#             'properties': {
#                 'time': TMS[idx],
#                 'style': {'color': ''},
#                 'icon': 'circle',
#                 'iconstyle': {
#                     'fillColor': '#black',
#                     'fillOpacity': 0.8,
#                     'stroke': 'true',
#                     'radius': 2
#                 }
#     },

#         }
#         for idx,pt in enumerate(Points)
#     ]


#     features = [
#         {
#             "type": "Feature",
#             "geometry": {
#                 "type": "LineString",
#                 "coordinates": line["coordinates"],
#             },
#             "properties": {
#                 "times": line["dates"],
#                 "style": {
#                     "weight": line["weight"] if "weight" in line else 3,
#                     'color': 'blue',
#                     "line-dasharray": [0.1, 1.8]
#                 },
#                 'icon': 'circle',
#                 'iconstyle': {'color': 'blue','iconSize': [1,1]}
#             },
#         }
#         for line in lines
#     ]

#     features =  features + pointfeatures

#     TimestampedGeoJson(
#         {
#             "type": "FeatureCollection",
#             "features": features,
#         },
#         period="PT1H",
#         duration="P7D",
#         auto_play=False,
#         add_last_point=True,
#         max_speed=50
#     ).add_to(map)
#     return map


# def MapCurrents(cellGrid,map,show=False,scale=15):
#     import folium
#     from pyproj import Geod
#     def bearing(st,en):
#         import numpy as np
#         long1,lat1 = st
#         long2,lat2 = en
#         dlong = long2-long1
#         dlat  = lat2-lat1
#         vector_1 = [0, 1]
#         vector_2 = [dlong, dlat]
#         if np.linalg.norm(vector_2) == 0:
#             return np.nan
#         unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
#         unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
#         dot_product = np.dot(unit_vector_1, unit_vector_2)
#         angle = np.arccos(dot_product)/(np.pi/180)*np.sign(vector_2[0])
#         if (angle==0) & (np.sign(dlat)==-1):
#             angle=180
#         if angle < 0:
#             angle = angle +360
#         angle
#         return angle

#     cellGrid
#     X=[];Y=[];U=[];V=[];
#     for ii in range(len(cellGrid.cellBoxes)):
#         cellbox = cellGrid.cellBoxes[ii]
#         if not isinstance(cellbox, CellBox):
#             continue

#         X.append(cellbox.cx)
#         Y.append(cellbox.cy)
#         U.append(cellbox.getuC())
#         V.append(cellbox.getvC())
#     Currents = pd.DataFrame({'X':X,'Y':Y,'U':U,'V':V})
#     Currents = Currents.dropna()
#     Currents['X'] = Currents['X']


#     vectors = folium.FeatureGroup(name='Currents',show=show)
#     for idx,vec in Currents.iterrows():
#         loc =[[vec['Y'],vec['X']],[vec['Y']+vec['V']*scale,vec['X']+vec['U']*scale]]
#         folium.PolyLine(loc, color="gray",weight=1.4).add_to(vectors)
#         # get pieces of the line
#         pairs = [(loc[idx], loc[idx-1]) for idx, val in enumerate(loc) if idx != 0]
#         # get rotations from forward azimuth of the line pieces and add an offset of 90°
#         geodesic = Geod(ellps='WGS84')
#         rotations = [geodesic.inv(pair[0][1], pair[0][0], pair[1][1], pair[1][0])[0]+90 for pair in pairs]
#         # create your arrow
#         for pair, rot in zip(pairs, rotations):
#             folium.RegularPolygonMarker(location=pair[0], color='gray', fill=True, fill_color='gray', fill_opacity=1,
#                                         number_of_sides=3, rotation=rot,radius=2,weight=0.8).add_to(vectors)

#     vectors.add_to(map)
#     return map


# def MapMesh(cellGrid,map,threshold=0.8):
#     DF = MeshDF(cellGrid)
#     LandDF = DF[DF['Land'] == True]
#     IceDF  = DF[DF['Land'] == False]
#     ThickIceDF = IceDF[IceDF['Ice Area'] >= threshold*100]
#     ThinIceDF  = IceDF[IceDF['Ice Area'] < threshold*100]

#     # ==== Plotting Ice ==== 
#     iceInfo = folium.FeatureGroup(name='Ice Mesh')
#     folium.GeoJson(
#         IceDF,
#         style_function=lambda x: {
#                 'fillColor': 'white',
#                 'color': 'gray',
#                 'weight': 0.5,
#                 'fillOpacity': x['properties']['Ice Area']/100
#             }
#     ).add_to(iceInfo)
#     folium.GeoJson(
#         ThickIceDF,
#         style_function=lambda x: {
#                 'color': 'red',
#                 'weight': 0.5,
#                 'fillOpacity': 0.0
#             }
#     ).add_to(iceInfo)
#     iceInfo.add_to(map)

#     # ===== Plotting Land =====
#     landInfo = folium.FeatureGroup(name='Land Mesh')
#     folium.GeoJson(
#         LandDF,
#         style_function= lambda x: {
#                 'fillColor': 'black',
#                 'color': 'gray',
#                 'weight': 0.5,
#                 'fillOpacity': 0.3
#             }
#     ).add_to(landInfo)
#     landInfo.add_to(map)

#     # ===== Plotting Mesh Info =====
#     # try:
#     #     bathInfo = folium.FeatureGroup(name='Bathymetry Mesh',show=False)
#     #     colormap = linear.Reds_09.scale(min(ThinIceDF['Depth']),max(ThinIceDF['Depth']))
#     #     folium.GeoJson(
#     #         IceDF,
#     #         style_function=lambda x: {
#     #                 'fillColor': colormap(x['properties']['Depth']),
#     #                 'color': 'gray',
#     #                 'weight': 0.5,
#     #                 'fillOpacity': 0.3
#     #             }
#     #     ).add_to(bathInfo)
#     #     bathInfo.add_to(map)

#     # ===== Plotting Mesh Info =====
#     meshInfo = folium.FeatureGroup(name='Mesh Information',show=False)
#     folium.GeoJson(
#         DF,
#         style_function= lambda x: {
#                 'fillColor': 'black',
#                 'color': 'gray',
#                 'weight': 0.5,
#                 'fillOpacity': 0.
#             },
#         tooltip=folium.GeoJsonTooltip(
#             fields=['Ice Area', 'Land','Cx','Cy','Depth','Vector','Index'],
#             aliases=['Ice Area (%)', 'Land','Centroid Cx [Long]','Centroid Cy [Lat]','Depth(m)','Vector (m/s)','Cell Index'],
#             localize=True
#         ),
#         name='Land Grid'
#     ).add_to(meshInfo)
#     meshInfo.add_to(map)
#     return map

# def MapGeotiff(map,file,subregion=None,name='Modis',resamplingFactor=1):
#     import folium
#     import rasterio as rs
#     from matplotlib import cm
#     from IceMDP.IO import windowFunc
#     import numpy as np
#     '''
#         Issue - Resampling only working on WGS84 projections
#     '''
#     dataset = rs.open(file)

#     if type(subregion)!=type(None):
#         window,bounds = windowFunc(subregion[0], subregion[1], dataset)
#         bnds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
#     else:
#         window=None
#         bnds = [[dataset.bounds[1], dataset.bounds[0]], [dataset.bounds[3], dataset.bounds[2]]]

#     trying =True
#     indx = 1
#     while trying:
#         try:

#             if indx==1:
#                 opc = 0.8
#                 shw = True
#             else:
#                 opc = 0.7
#                 shw = False

#             img = folium.raster_layers.ImageOverlay(
#                 name="{} - Band {}".format(name,indx),
#                 image=dataset.read(indx,window=window)[::resamplingFactor,::resamplingFactor],
#                 bounds=bnds,
#                 opacity=opc,
#                 mercator_project=True,
#                 pixelated=False,
#                 show = shw
#             )
#             img.add_to(map)
#         except:
#             trying=False
#         indx+=1
#     return map


# def BaseMap(TitleText,MapCentre=[-58,-63.7],MapZoom=2.6):
#     title_html = '''
#                 <h1 style="color:#003b5c;font-size:16px">
#                 &ensp;<img src='https://i.ibb.co/JH2zknX/Small-Logo.png' alt="BAS-colour-eps" border="0" style="width:40px;height:40px;"> 
#                 <img src="https://i.ibb.co/XtZdzDt/BAS-colour-eps.png" alt="BAS-colour-eps" border="0" style="width:179px;height:40px;"> 
#                 &ensp;|&ensp; RoutePlanner &ensp;|&ensp; {}
#                 </h1>
#                 </body>
#                 '''.format(TitleText)   
#     map = folium.Map(location=MapCentre,zoom_start=MapZoom,tiles=None)
#     bsmap = folium.FeatureGroup(name='BaseMap')
#     folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}.png',attr="toner-bcg", name='Basemap').add_to(bsmap)
#     bsmap.add_to(map)
#     map.get_root().html.add_child(folium.Element(title_html))
#     return map

# def MapTravelTime(TT,map,source='Halley'):
#     from branca.colormap import linear
#     def DijkstraInfoDF(TT,source='Halley'):
#         from shapely.geometry import Polygon
#         import geopandas as gpd
#         import numpy as np
#         DijkstraInfo = TT.DijkstraInfo[source]
#         Shape   = []; CentroidCx=[];CentroidCy=[];TravelTime=[]
#         for idx in DijkstraInfo.index:
#             c = TT.Mesh.cellBoxes[idx]
#             if isinstance(c, CellBox):
#                 if DijkstraInfo['traveltime'].loc[idx] == np.inf  and DijkstraInfo['positionLocked'].loc[idx] == False:
#                     continue
#                 bounds = np.array(c.getBounds())
#                 Shape.append(Polygon(bounds))
#                 CentroidCx.append(c.cx)
#                 CentroidCy.append(c.cy)
#                 if DijkstraInfo['traveltime'].loc[idx] == np.inf:
#                     TravelTime.append(0.0)
#                 else:
#                     TravelTime.append(float(DijkstraInfo['traveltime'].loc[idx]))
#         Polygons = pd.DataFrame()
#         Polygons['geometry'] = Shape
#         Polygons['Cx']       = CentroidCx
#         Polygons['Cy']       = CentroidCy
#         Polygons['TravelTime']    = TravelTime
#         Polygons = gpd.GeoDataFrame(Polygons,crs={'init': 'epsg:4326'}, geometry='geometry')
#         return Polygons

#     DF = DijkstraInfoDF(TT,source=source)


#     bathInfo = folium.FeatureGroup(name='TravelTime Mesh',show=True)
#     colormap = linear.viridis.scale(min(DF['TravelTime']),max(DF['TravelTime']))
#     folium.GeoJson(
#         DF,
#         style_function=lambda x: {
#                 'fillColor': colormap(x['properties']['TravelTime']),
#                 'color': 'gray',
#                 'weight': 0.1,
#                 'fillOpacity': 1.0
#             }
#     ).add_to(bathInfo)
#     bathInfo.add_to(map)
#     return map



# def LayerControl(map,collapsed=True):
#     folium.LayerControl(collapsed=collapsed).add_to(map)
#     return map

#≠================================
####DEPRICATED


# def PlotPaths(cellGrid,Paths,routepoints=False,figInfo=None,return_ax=False,Waypoints=None):
#         if type(figInfo) == type(None):
#             fig,ax = plt.subplots(1,1,figsize=(15,10))
#             fig.patch.set_facecolor('white')
#             ax.set_facecolor('lightblue')
#         else:
#             fig,ax = figInfo

#         ax = PlotMesh(cellGrid,figInfo=[fig,ax],return_ax=True)

#         for Path in Paths:
#             if Path['Time'] == np.inf:
#                 continue
#             Points = np.array(Path['Path']['Points'])
#             if routepoints:
#                 ax.plot(Points[:,0],Points[:,1],linewidth=1.0,color='k')
#                 ax.scatter(Points[:,0],Points[:,1],30,zorder=99,color='k')
#             else:
#                 ax.plot(Points[:,0],Points[:,1],linewidth=1.0,color='k')


#         if type(Waypoints) != type(None):
#             ax.scatter(Waypoints['Long'],Waypoints['Lat'],50,marker='^',color='k')

#         if return_ax:
#             return ax



# def PlotMeshNeighbours(cellGrid,Xpoint,Ypoint,figInfo=None,return_ax=False):

#     if type(figInfo) == type(None):
#         fig,ax = plt.subplots(1,1,figsize=(15,10))
#         fig.patch.set_facecolor('white')
#         ax.set_facecolor('lightblue')
#     else:
#         fig,ax = figInfo

#     ax = PlotMesh(cellGrid,figInfo=[fig,ax],return_ax=True)
#     cell  = cellGrid.getCellBox(Xpoint,Ypoint)
#     neigh = cellGrid.getNeightbours(cell)
#     for ncell_indx,ncell in neigh.iterrows():
#         ax.add_patch(Polygon(ncell['Cell'].getBounds(), closed = True, fill = False, color = 'Red'))
#         ax.scatter(ncell['Cp'][0],ncell['Cp'][1],50,'b')
#         ax.scatter(ncell['Cell'].cx,ncell['Cell'].cy,50,'r')
#         ax.text(ncell['Cell'].cx+0.1,ncell['Cell'].cy+0.1,ncell['Case'])
#     ax.add_patch(Polygon(cell.getBounds(), closed = True, fill = False, color = 'Black'))
#     ax.scatter(cell.cx,cell.cy,50,'k')
#     ax.scatter(Xpoint,Ypoint,50,'m')

#     # Add in the xlims,ylims to neighbour grid cells

#     if return_ax:
#         return ax




# def Paths2KML(Paths,File):
#     kml = Kml(open=1)
#     for path in Paths:
#         if np.isinf(path['Time']):
#             continue
#         linestring = kml.newlinestring(name="{} -> {}. TravelTime={} days".format(path['from'],path['to'],path['Time']))
#         if type=='Maps':
#             fullPath = path['Path']['Points']
#             fullPath[:,0] = fullPath[:,0]-360
#             fullPath = fullPath[:,::-1]
#             linestring.coords = fullPath
#         else:
#             linestring.coords = path['Path']['Points']
#     kml.save(File)


# def Mesh2KML(cellGrid,File,MaxIce=0.8):
#     kml = Kml(open=1)
#     for ii in range(len(cellGrid.cellBoxes)):
#         cell = kml.newmultigeometry(name="Cell Box {}".format(ii))
#         if type=='Maps':
#             bounds      = np.array(cellGrid.cellBoxes[ii].getBounds())
#             bounds[:,0] = bounds[:,0]-360
#             bounds = bounds[:,::-1]

#         cell.newpolygon(outerboundaryis=cellGrid.cellBoxes[ii].getBounds())
#         cell.style.linestyle.width = 0.1
#         if cellGrid.cellBoxes[ii].containsLand():
#             cell.style.polystyle.color = Color.changealpha("77", Color.green)
#             cell.style.linestyle.color = Color.changealpha("77", Color.green)
#         else:
#             cell.style.linestyle.color = Color.black
#             icearea = cellGrid.cellBoxes[ii].iceArea()
#             if not np.isnan(icearea):
#                 if icearea > MaxIce:
#                     cell.style.linestyle.color = Color.changealpha("77", Color.pink)
#                 else:
#                     cell.style.polystyle.color = Color.changealpha("{}".format(int(cellGrid.cellBoxes[ii].iceArea()*100)), Color.white)
#             else:
#                 cell.style.polystyle.color = Color.changealpha("77", Color.green)
#                 cell.style.linestyle.color = Color.changealpha("77", Color.green)
#     kml.save(File)

# def WayPoints2KML(Waypoints,File):
#     kml = Kml(open=1)
#     style = Style()
#     style.labelstyle.color = Color.red  
#     style.linestyle.color= Color.red
#     style.labelstyle.scale = 0.8  
#     style.iconstyle.icon.href = 'https://maps.google.com/mapfiles/kml/shapes/placemark_circle_highlight.png'
#     for ii,wpt in Waypoints.iterrows():
#         if type=='Maps':
#             pnt = kml.newpoint(name="{}".format(wpt['Name']), coords=[(wpt['Lat'],wpt['Long']-360)])
#         else:
#             pnt = kml.newpoint(name="{}".format(wpt['Name']), coords=[(wpt['Long'],wpt['Lat'])])
#         pnt.style = style
#     kml.save(File)