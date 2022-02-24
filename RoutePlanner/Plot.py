import matplotlib.pylab as plt
from matplotlib.patches import Polygon
import numpy as np
from simplekml import Kml, Color, Style
from RoutePlanner.IO import SDAPosition, MeshDF
import folium
import pandas as pd

#  =============
from branca.colormap import linear
import folium
import copy
from folium.plugins import TimestampedGeoJson

# Example Maps http://leaflet-extras.github.io/leaflet-providers/preview/
#icons can be found at https://fontawesome.com/icons



def MapWaypoints(DF,map):
    wpts = folium.FeatureGroup(name='WayPoints')
    for id,wpt in DF.iterrows():
        loc = [wpt['Lat'], wpt['Long']-360]
        folium.Marker(
            location=loc,
            icon=folium.plugins.BeautifyIcon(icon='circle',
                                            border_color='transparent',
                                            background_color='transparent',
                                            border_width=1,
                                            text_color='red',
                                            inner_icon_style='margin:0px;font-size:0.8em'),
            popup="<b>{} [{},{}]</b>".format(wpt['Name'],loc[0],loc[1])
        ).add_to(wpts)    
    wpts.add_to(map)
    return map

def MapResearchSites(DF,map):
    wpts = folium.FeatureGroup(name='Research Stations')
    for id,wpt in DF.iterrows():
        loc = [wpt['Lat'], wpt['Long']]
        folium.Marker(
            location=loc,
            name='Research Sites',
            icon=folium.plugins.BeautifyIcon(icon='circle',
                                            border_color='transparent',
                                            background_color='transparent',
                                            border_width=1,
                                            text_color='orange',
                                            inner_icon_style='margin:0px;font-size:0.8em'),
            popup="<b>{} [{},{}]</b>".format(wpt['Name'],loc[0],loc[1])
        ).add_to(wpts)
    wpts.add_to(map)    
    return map


def MapPaths(Paths,map,PathPoints=True):
    Pths        = folium.FeatureGroup(name='Paths')
    Pths_points = folium.FeatureGroup(name='Path Points')
    for path in copy.deepcopy(Paths):
        points = path['Path']['Points']
        points[:,0] = points[:,0]-360
        points = points[:,::-1]
        folium.PolyLine(points,color="black", weight=2.5, opacity=1,
                        popup='{}->{}\n Travel-Time = {:.2f}days'.format(path['from'],path['to'],path['Time'])).add_to(Pths)
        for idx in range(len(points)):
            loc = [points[idx,0],points[idx,1]]
            folium.Marker(
                location=loc,
                icon=folium.DivIcon(html=f"""
                    <div><svg>
                        <rect x="0", y="0" width="10" height="10", fill="black", opacity="1.0" 
                    </svg></div>""")
            ).add_to(Pths_points)
    Pths.add_to(map)
    if PathPoints:
        Pths_points.add_to(map)
    return map


def TimeMapPaths(Paths,map,starttime='2014-01-01T00:00:00'):
    lines=[]
    for Path in copy.deepcopy(Paths):
        Points = Path['Path']['Points']
        Points[:,0] = Points[:,0]-360
        Times  = pd.to_datetime(starttime) + pd.to_timedelta(Path['Path']['Time'],unit='D')

        entry = {}
        entry['coordinates'] = Points.tolist()
        entry['dates'] = Times.strftime('%Y-%m-%dT%H:%M:%S').tolist()
        entry['color'] = 'black'
        lines.append(entry)


    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": line["coordinates"],
            },
            "properties": {
                "times": line["dates"],
                "style": {
                    "color": line["color"],
                    "weight": line["weight"] if "weight" in line else 3,
                    'color': 'red'
                },
                'icon': 'circle',
                'iconstyle': {'color': 'red','iconSize': [1,1]}
            },
        }
        for line in lines
    ]

    TimestampedGeoJson(
        {
            "type": "FeatureCollection",
            "features": features,
        },
        period="PT1H",
        auto_play=False,
        add_last_point=True,
    ).add_to(map)
    return map


def TimeMapSDA(PATH,map):
    #'/Users/jsmith/Documents/Research/Researcher_BAS/RoutePlanning/SDADT-Positions'
    Info = SDAPosition(PATH)
    lines=[]
    Points = Info[['Long','Lat']].to_numpy()
    Points[:,0] = Points[:,0]-360
    entry = {}
    entry['coordinates'] = Points.tolist()
    entry['dates'] = Info['Time'].dt.strftime('%Y-%m-%dT%H:%M:%S').to_list()
    entry['color'] = 'blue'
    lines.append(entry)

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": line["coordinates"],
            },
            "properties": {
                "times": line["dates"],
                "style": {
                    "color": line["color"],
                    "weight": line["weight"] if "weight" in line else 3,
                    'color': 'blue'
                },
                'icon': 'circle',
                'iconstyle': {'color': 'blue','iconSize': [1,1]}
            },
        }
        for line in lines
    ]



    TimestampedGeoJson(
        {
            "type": "FeatureCollection",
            "features": features,
        },
        period="PT1H",
        auto_play=False,
        add_last_point=True,
        max_speed=50
    ).add_to(map)
    return map

def MapMesh(cellGrid,map):
    DF = MeshDF(cellGrid)
    LandDF = DF[DF['Land'] == True]
    IceDF  = DF[DF['Land'] == False]
    ThickIceDF = IceDF[IceDF['Ice Area'] >= 0.8*100]
    ThinIceDF  = IceDF[IceDF['Ice Area'] < 0.8*100]

    # ==== Plotting Ice ==== 
    iceInfo = folium.FeatureGroup(name='Ice Mesh')
    folium.GeoJson(
        IceDF,
        style_function=lambda x: {
                'fillColor': 'white',
                'color': 'gray',
                'weight': 0.5,
                'fillOpacity': x['properties']['Ice Area']/100
            }
    ).add_to(iceInfo)
    folium.GeoJson(
        ThickIceDF,
        style_function=lambda x: {
                'color': 'red',
                'weight': 0.5,
                'fillOpacity': 0.1
            }
    ).add_to(iceInfo)
    iceInfo.add_to(map)

    # ===== Plotting Land =====
    landInfo = folium.FeatureGroup(name='Land Mesh')
    folium.GeoJson(
        LandDF,
        style_function= lambda x: {
                'fillColor': 'green',
                'color': 'gray',
                'weight': 0.5,
                'fillOpacity': 0.3
            }
    ).add_to(landInfo)
    landInfo.add_to(map)

    # ===== Plotting Mesh Info =====
    bathInfo = folium.FeatureGroup(name='Bathymetry Mesh',show=False)
    colormap = linear.Reds_09.scale(min(ThinIceDF['Depth']),max(ThinIceDF['Depth']))
    folium.GeoJson(
        IceDF,
        style_function=lambda x: {
                'fillColor': colormap(x['properties']['Depth']),
                'color': 'gray',
                'weight': 0.5,
                'fillOpacity': 0.3
            }
    ).add_to(bathInfo)
    bathInfo.add_to(map)
    # ===== Plotting Mesh Info =====
    meshInfo = folium.FeatureGroup(name='Mesh Information',show=False)
    folium.GeoJson(
        DF,
        style_function= lambda x: {
                'fillColor': 'black',
                'color': 'gray',
                'weight': 0.5,
                'fillOpacity': 0.
            },
        tooltip=folium.GeoJsonTooltip(
            fields=['Ice Area', 'Land','Cx','Cy','Depth','Vector'],
            aliases=['Ice Area (%)', 'Land','Centroid Cx [Long]','Centroid Cy [Lat]','Depth(m)','Vector (m/s)'],
            localize=True
        ),
        name='Land Grid'
    ).add_to(meshInfo)
    meshInfo.add_to(map)
    return map


def BaseMap(location=[-58,-63.7],logo=True,logoPos=[10,90]):
    map = folium.Map(location=location,zoom_start=4.6,tiles=None)
    bsmap = folium.FeatureGroup(name='BaseMap')
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}.png',attr="toner-bcg", name='Basemap').add_to(bsmap)
    bsmap.add_to(map)
    if logo:
        folium.plugins.FloatImage('https://i.ibb.co/dr3TNf7/Large-Logo.jpg',bottom=logoPos[1],left=logoPos[0]).add_to(map)
    return map

def LayerControl(map,collapsed=True):
    folium.LayerControl(collapsed=collapsed).add_to(map)
    return map

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