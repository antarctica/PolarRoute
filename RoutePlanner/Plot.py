import matplotlib.pylab as plt
from matplotlib.patches import Polygon
import numpy as np
from simplekml import Kml, Color, Style


def Mesh(self,figInfo=None,currents=False,return_ax=False,iceThreshold=None):
    """
        plots this cellGrid for display.

        TODO - requires reworking as part of the plotting work-package
    """
    if type(figInfo) == type(None):
        fig,ax = plt.subplots(1,1,figsize=(15,10))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('lightblue')
    else:
        fig,ax = figInfo

    for cellBox in self.cellBoxes:
        if cellBox.containsLand():
            ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = True, facecolor='mediumseagreen'))
            ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = False, edgecolor='gray'))
            continue

        iceArea = cellBox.iceArea()
        if iceArea >= 0.8:
            ax.add_patch(Polygon(cellBox.getBounds(),closed=True,fill=True,color='White'))
            ax.add_patch(Polygon(cellBox.getBounds(),closed=True,fill=True,color='Pink',alpha=0.4))
            ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = False,edgecolor='gray'))
        elif not np.isnan(iceArea):
            ax.add_patch(Polygon(cellBox.getBounds(),closed=True,fill=True,color='White',alpha=iceArea))
            ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = False,edgecolor='gray'))
        else:
            ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = True, facecolor='mediumseagreen'))
            ax.add_patch(Polygon(cellBox.getBounds(), closed = True, fill = False, edgecolor='gray'))

        if currents:
            ax.quiver((cellBox.long+cellBox.width/2),(cellBox.lat+cellBox.height/2),cellBox.getuC()*1000,cellBox.getvC()*1000,scale=2,width=0.002,color='gray')

    ax.set_xlim(self._longMin, self._longMax)
    ax.set_ylim(self._latMin, self._latMax)

    if return_ax:
        return ax


def MeshNeighbours(cellGrid,Xpoint,Ypoint,figInfo=None,return_ax=False):

    if type(figInfo) == type(None):
        fig,ax = plt.subplots(1,1,figsize=(15,10))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('lightblue')
    else:
        fig,ax = figInfo

    ax = Mesh(cellGrid,figInfo=[fig,ax],return_ax=True)
    cell  = cellGrid.getCellBox(Xpoint,Ypoint)
    neigh = cellGrid.getNeightbours(cell)
    for ncell_indx,ncell in neigh.iterrows():
        ax.add_patch(Polygon(ncell['Cell'].getBounds(), closed = True, fill = False, color = 'Red'))
        ax.scatter(ncell['Cp'][0],ncell['Cp'][1],50,'b')
        ax.scatter(ncell['Cell'].cx,ncell['Cell'].cy,50,'r')
        ax.text(ncell['Cell'].cx+0.1,ncell['Cell'].cy+0.1,ncell['Case'])
    ax.add_patch(Polygon(cell.getBounds(), closed = True, fill = False, color = 'Black'))
    ax.scatter(cell.cx,cell.cy,50,'k')
    ax.scatter(Xpoint,Ypoint,50,'m')

    # Add in the xlims,ylims to neighbour grid cells

    if return_ax:
        return ax



def Paths(cellGrid,Paths,routepoints=False,figInfo=None,return_ax=False,Waypoints=None):
        if type(figInfo) == type(None):
            fig,ax = plt.subplots(1,1,figsize=(15,10))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('lightblue')
        else:
            fig,ax = figInfo

        ax = Mesh(cellGrid,figInfo=[fig,ax],return_ax=True)

        for Path in Paths:
            if Path['Time'] == np.inf:
                continue
            Points = np.array(Path['Path']['Points'])
            if routepoints:
                ax.plot(Points[:,0],Points[:,1],linewidth=1.0,color='k')
                ax.scatter(Points[:,0],Points[:,1],30,zorder=99,color='k')
            else:
                ax.plot(Points[:,0],Points[:,1],linewidth=1.0,color='k')


        if type(Waypoints) != type(None):
            ax.scatter(Waypoints['Long'],Waypoints['Lat'],50,marker='^',color='k')

        if return_ax:
            return ax



def Paths2KML(Paths,File):
    kml = Kml(open=1)
    for path in Paths:
        if np.isinf(path['Time']):
            continue
        linestring = kml.newlinestring(name="{} -> {}. TravelTime={} days".format(path['from'],path['to'],path['Time']))
        if type=='Maps':
            fullPath = path['Path']['Points']
            fullPath[:,0] = fullPath[:,0]-360
            fullPath = fullPath[:,::-1]
            linestring.coords = fullPath
        else:
            linestring.coords = path['Path']['Points']
    kml.save(File)


def Mesh2KML(cellGrid,File,MaxIce=0.8):
    kml = Kml(open=1)
    for ii in range(len(cellGrid.cellBoxes)):
        cell = kml.newmultigeometry(name="Cell Box {}".format(ii))
        if type=='Maps':
            bounds      = np.array(cellGrid.cellBoxes[ii].getBounds())
            bounds[:,0] = bounds[:,0]-360
            bounds = bounds[:,::-1]

        cell.newpolygon(outerboundaryis=cellGrid.cellBoxes[ii].getBounds())
        cell.style.linestyle.width = 0.1
        if cellGrid.cellBoxes[ii].containsLand():
            cell.style.polystyle.color = Color.changealpha("77", Color.green)
            cell.style.linestyle.color = Color.changealpha("77", Color.green)
        else:
            cell.style.linestyle.color = Color.black
            icearea = cellGrid.cellBoxes[ii].iceArea()
            if not np.isnan(icearea):
                if icearea > MaxIce:
                    cell.style.linestyle.color = Color.changealpha("77", Color.pink)
                else:
                    cell.style.polystyle.color = Color.changealpha("{}".format(int(cellGrid.cellBoxes[ii].iceArea()*100)), Color.white)
            else:
                cell.style.polystyle.color = Color.changealpha("77", Color.green)
                cell.style.linestyle.color = Color.changealpha("77", Color.green)
    kml.save(File)

def WayPoints2KML(Waypoints,File):
    kml = Kml(open=1)
    style = Style()
    style.labelstyle.color = Color.red  
    style.linestyle.color= Color.red
    style.labelstyle.scale = 0.8  
    style.iconstyle.icon.href = 'https://maps.google.com/mapfiles/kml/shapes/placemark_circle_highlight.png'
    for ii,wpt in Waypoints.iterrows():
        if type=='Maps':
            pnt = kml.newpoint(name="{}".format(wpt['Name']), coords=[(wpt['Lat'],wpt['Long']-360)])
        else:
            pnt = kml.newpoint(name="{}".format(wpt['Name']), coords=[(wpt['Long'],wpt['Lat'])])
        pnt.style = style
    kml.save(File)


#  =============
from folium.plugins import FloatImage, BoatMarker
from branca.colormap import linear

# Example Maps http://leaflet-extras.github.io/leaflet-providers/preview/
#icons can be found at https://fontawesome.com/icons

def MapWaypoints(json,map):
    for pt in json['features']:
        loc = pt['geometry']['coordinates']
        folium.Marker(
        location=[loc[1], loc[0]],
        name='WayPoints',
        # icon=folium.Icon(color="red",icon="cloud"),


        icon=folium.DivIcon(html=f"""
            <div><svg>
                <rect x="0", y="0" width="10" height="10", fill="red", opacity="1.0" 
            </svg></div>"""),


        popup="<b>{} [{},{}]</b>".format(pt['properties']['name'],loc[0],loc[1])
        ).add_to(map)    
    return map

def MapMesh(DF,map):


    LandDF = DF[DF['Land'] == True]

    IceDF  = DF[DF['Land'] == False]
    ThickIceDF = IceDF[IceDF['Ice Area'] >= 0.8*100]
    ThinIceDF  = IceDF[IceDF['Ice Area'] < 0.8*100]

    # ==== Plotting Ice ==== 
    colormap = linear.Reds_09.scale(min(ThinIceDF['Ice Area']),
                                                max(ThinIceDF['Ice Area']))
    style_function = lambda x: {
        'fillColor': colormap(x['properties']['Ice Area']),
        'color': 'gray',
        'weight': 0.5,
        'fillOpacity': 0.3
    }
    folium.GeoJson(
        ThinIceDF,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['Ice Area', 'Land','Centroid','Depth','Vector'],
            aliases=['Ice Area (%)', 'Land','Centroid [Long,Lat]','Depth(m)','Vector (m/s)'],
            localize=True
        ),
        name='Ice Grid'
    ).add_to(map)
    # colormap.add_to(map)
    # colormap.caption = 'Ice Area'
    # colormap.add_to(map)
    style_function = lambda x: {
        'fillColor': 'white',
        'color': 'gray',
        'weight': 0.5,
        'fillOpacity': 0.5
    }
    folium.GeoJson(
        ThickIceDF,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['Ice Area', 'Land','Centroid','Depth','Vector'],
            aliases=['Ice Area (%)', 'Land','Centroid [Long,Lat]','Depth(m)','Vector (m/s)'],
            localize=True
        ),
        name='No-Go Ice Grid'
    ).add_to(map)

    # ===== Plotting Land =====
    style_function = lambda x: {
        'fillColor': 'green',
        'color': 'gray',
        'weight': 0.5,
        'fillOpacity': 0.3
    }
    folium.GeoJson(
        LandDF,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['Ice Area', 'Land','Centroid','Depth','Vector'],
            aliases=['Ice Area (%)', 'Land','Centroid [Long,Lat]','Depth(m)','Vector (m/s)'],
            localize=True
        ),
        name='Land Grid'
    ).add_to(map)

    return map