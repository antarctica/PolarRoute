"""
Outlined in this section we will discuss the usage of the automated plotting functionallity
of the pyRoutePlanner. In this series of class distributions we house our interactive web based plots as well 
static plots ready for publication usage.
representation of input data. In each CellBox we determine the mean and variance of 
the information goverining our nemerical world, this includes and is not limited to:
Ocean Currents, Sea Ice Concentration, Bathemetric depth, whether on land.

Example:
    An example of running this code can be executed by running the following in a ipython/Jupyter Notebook::

        from RoutePlanner import CellBox
        ....

Additional information on constructing document strings using the Google DocString method can be found at
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

Attributes:
    Some of the key attributes that the CellBox comprises are ...

Todo:
    * Adding the addition of ...


"""





import copy
import json
import pandas as pd
import matplotlib
import numpy as np
import sys
from branca.colormap import linear
import folium
from pyproj import transform
from shapely import wkt
import geopandas as gpd
from folium import plugins
from shapely.geometry import Polygon

sys.path.insert(0, 'folium')
sys.path.insert(0, 'branca')

import branca
import folium

# Adapted from: https://nbviewer.org/gist/BibMartin/f153aa957ddc5fadc64929abdee9ff2e
from branca.element import MacroElement
from jinja2 import Template
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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



def deep_search(needles, haystack):
    found = {}
    if type(needles) != type([]):
        needles = [needles]

    if type(haystack) == type(dict()):
        for needle in needles:
            if needle in haystack.keys():
                found[needle] = haystack[needle]
            elif len(haystack.keys()) > 0:
                for key in haystack.keys():
                    result = deep_search(needle, haystack[key])
                    if result:
                        for k, v in result.items():
                            found[k] = v
    elif type(haystack) == type([]):
        for node in haystack:
            result = deep_search(needles, node)
            if result:
                for k, v in result.items():
                    found[k] = v
    return found



from shapely.geometry.polygon import Polygon
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from shapely import wkt
import cartopy.feature as cfeature
import pandas as pd
import cartopy.io.img_tiles as cimgt
import json


class StaticMap:
    def __init__(self, json):
    
        self.json = json
        self.config  = json['config']
        self.basemap = self.config["Static_Map"]['Basemap_Info']
        self.layers  = self.config['Static_Map']['Layers']
        cellboxes    = json['cellboxes']
        


        #Initialising the basemap
        if 'CRS' in self.basemap:
            if self.basemap['CRS'] == 'Mercartor':
                self.ccrs = ccrs.Mercator()
            if self.basemap['CRS'] == 'Orthographic':
                self.ccrs = ccrs.Orthographic(central_longitude=self.basemap['central_longitude'],central_latitude=self.basemap['central_latitude'])
        else:
           self.ccrs = ccrs.Mercator()
        self._basemap()


        # Overlaying the layers
        for idx, layer in enumerate(self.layers):
            self.zorder = idx+1
            if not layer['Show']:
                continue
            #try:
            if layer['Type'] == 'Maps':
                self._maps(layer, cellboxes)
            if layer['Type'] == 'Paths':
                self._paths(layer)
            if layer['Type'] == 'Points':
                self._points(layer)

        if 'Output_Filename' in self.config['Static_Map']:
            plt.savefig(self.config['Static_Map']['Output_Filename'])


    def _basemap(self):
        self.fig = plt.figure(figsize=(15,15))
        matplotlib.rcParams.update({'font.size': 16})

        self.ax = plt.axes(projection=self.ccrs)
        self.ax.set_extent([self.config['Mesh_info']['Region']['longMin']+1e-6,self.config['Mesh_info']['Region']['longMax']-1e-6,self.config['Mesh_info']['Region']['latMin'],self.config['Mesh_info']['Region']['latMax']], crs=ccrs.PlateCarree())
        self.ax.add_image(cimgt.GoogleTiles(), 3)
        self.ax.coastlines(resolution='50m')
        self.ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,linewidth=0.5,linestyle='--')
        self.ax.add_feature(cfeature.BORDERS)
        plt.title(r'Route Planner | {}'.format(self.basemap['Title']),fontsize=14,loc='left',color='blue')

    def _points(self,info):

        if 'filename' in info:
            dataframe_points = pd.read_csv(info['filename'])
        elif 'object' in info:
            dataframe_points = pd.DataFrame(deep_search([info['object']],self.json )[info['object']])
        else:
            print('Please define either a filename or object in the json')

        if ('Name' in info) and info['Name']:
            dataframe_points[info['Name']][dataframe_points[info['Name']] > info['Fill_trim_max']] = np.nan
            dataframe_points[info['Name']][dataframe_points[info['Name']] <= info['Fill_trim_min']] = np.nan
            dataframe_points = dataframe_points.dropna()
            if ('Cmap' in info) and info['Cmap']:
                cmap = linear._colormaps[info['Cmap']].scale(dataframe_points[info['Name']].min(), dataframe_points[info['Name']].max())
                colour = [cmap.rgba_floats_tuple(ii) for ii in np.array(dataframe_points[info['Name']])]
            else:
                colour = info['Color']
            self.ax.scatter(dataframe_points['Long'],dataframe_points['Lat'],info["Size"],c=colour,marker='o',transform=ccrs.PlateCarree(),zorder=self.zorder)
        else:
            self.ax.scatter(dataframe_points['Long'],dataframe_points['Lat'],info["Size"],marker='o',transform=ccrs.PlateCarree(),color=info['Color'],zorder=self.zorder)


    def _paths(self,info):
        '''
            Plotting a paths type object
        '''

        # Loading the path information

        if 'filename' in info:
            with open(info['filename'], 'r') as f:
                geojson = json.load(f)
        elif 'object' in info:
            geojson = deep_search([info['object']],self.json)[info['object']]
        else:
            print('Please define either a filename or object in the json')
        paths = geojson['features']

        # Determining max values of all paths
        if info['Colorline']:
            max_val = 0
            min_val = 0
            for path in copy.deepcopy(paths):
                if np.array(path['properties'][info['Data_Name']]).max() > max_val:
                    max_val = np.array(path['properties'][info['Data_Name']]).max()
                if np.array(path['properties'][info['Data_Name']]).min() < min_val:
                    min_val = np.array(path['properties'][info['Data_Name']]).min()

        # Determining max travel-times of all paths
        for path in paths:
            points   = np.array(path['geometry']['coordinates'])
            if 'Data_Name' in info:
                data_val = np.array(path['properties'][info['Data_Name']])
            else:
                data_val = np.array(len(points))


            if info['Colorline']:
                # Add ColorLines
                x = self.ccrs.transform_points(x=points[:,0], y=points[:,1],
                                                src_crs=ccrs.PlateCarree())
                xcs = np.array([x[:,0],x[:,1]]).T.reshape(-1,1,2) 
                segments = np.concatenate([xcs[:-1], xcs[1:]], axis=1)   
                lc = LineCollection(segments, cmap=info['Cmap'], linewidth=info['Line_Width'],norm=plt.Normalize(vmin=min_val, vmax=max_val),zorder=self.zorder) 
                lc.set_array(path['properties']['traveltime'])                                           
                self.ax.add_collection(lc) 
                
            
            else:
                self.ax.plot(points[:,0],points[:,1],transform=ccrs.PlateCarree(),linewidth=info['Line_Width'],color=info['Color'],alpha=1.,zorder=self.zorder)

            if info['Path_Points']:
                self.ax.scatter(points[:,0],points[:,1],color=info['Color'],)
        
        if info['Colorline']:
            cbaxes = inset_axes(self.ax, '25%', '3%', loc =1)
            cbaxes.set_facecolor([1,1,1,0.7])
            cb=self.fig.colorbar(lc,cax=cbaxes,orientation='horizontal',label='Traveltime') #make colorbar



    def _maps(self,info, cellboxes):
            '''
                Plotting a map type object
            '''
            dataframe_pandas = pd.DataFrame(cellboxes)
            #dataframe_pandas = pd.read_csv(info['filename'])
            dataframe_pandas['geometry'] = dataframe_pandas['geometry'].apply(wkt.loads)
            dataframe_geo = gpd.GeoDataFrame(dataframe_pandas,crs='EPSG:4326', geometry='geometry')

            if dataframe_geo[info['Data_Name']].dtype == 'bool':
                for _,poly in dataframe_geo.iterrows():
                    if poly[info['Data_Name']]:
                        self.ax.add_geometries([poly['geometry']], crs=ccrs.PlateCarree(), edgecolor=info['Line_Color'], facecolor=info['Fill_Color'],alpha=info['Fill_Opacity'],lw=info['Line_Width'],zorder=self.zorder)


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
                        for _,poly in dataframe_geo.iterrows():
                            self.ax.add_geometries([poly['geometry']], crs=ccrs.PlateCarree(), edgecolor=info['Line_Color'], facecolor=info['Fill_Color'],alpha=poly[info['Data_Name']]/info['Fill_cmap_opacity_scalar'],lw=info['Line_Width'],zorder=self.zorder)
                    else:
                        if 'Fill_cmap_min' in info:
                            cmin = info['Fill_cmap_min']
                        else:
                            cmin = dataframe_geo[info['Data_Name']].min()
                        if 'Fill_cmap_max' in info:
                            cmax = info['Fill_cmap_max']
                        else:
                            cmax = dataframe_geo[info['Data_Name']].max()
                            # Define a colormap version
                            cmap = linear._colormaps[info['Cmap']].scale(info['Fill_trim_min'], info['Fill_trim_max'])
                            for indx,row in dataframe_geo.iterrows():
                                data = row[info['Data_Name']]
                                if not np.isnan(data):
                                    colour = cmap.rgba_floats_tuple(data)
                                    self.ax.add_geometries([row['geometry']], facecolor = colour, crs = ccrs.PlateCarree(), alpha = info['Fill_Opacity'],zorder=self.zorder)
                else:
                    if (info['Fill_Opacity'] == 0.0):
                        for _,poly in dataframe_geo.iterrows():
                            x,y = poly['geometry'].exterior.coords.xy
                            self.ax.plot(np.array(x),np.array(y),color=info['Line_Color'],linewidth=info['Line_Width'],transform=ccrs.PlateCarree(),zorder=self.zorder)
    def show(self):
        plt.show()


class InteractiveMap:
    def __init__(self,config):
        self.config = config

        self.basemap = config["Interactive_Map"]['Basemap_Info']
        self.layers  = config['Interactive_Map']['Layers']


        # Initialising the basemap
        self._basemap()


        # Defining the layers to plot
        for layer in self.layers:
            try:
                if layer['Type'] == 'Paths':
                    self._paths(layer)
                if layer['Type'] == 'Maps':
                    self._maps(layer) 
                if layer['Type'] == 'Points':
                    self._points(layer)
                if layer['Type'] == 'Geotiff':
                    self._geotiff(layer)
                if layer['Type'] == 'MeshInfo':
                    self._meshInfo(layer)
            except:
                print('Issue Plotting Layer')
                    
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
        with open(info['filename'], 'r') as f:
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
                #folium.PolyLine(points,color="black", weight=info['Line_Width'], opacity=1).add_to(pths)
                colormap = linear._colormaps[info['Cmap']].scale(0,max_val)
                colormap.caption = '{} ({},Max Value={:.3f})'.format(info['Data_Name'],info['Data_Units'],max_val)
                folium.ColorLine(points,data_val,colormap=colormap,nb_steps=50, weight=info['Line_Width'], opacity=1).add_to(pths)



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
        dataframe_geo = dataframe_geo[['geometry']+list(info['Data_Names'])]

        feature_info = folium.FeatureGroup(name='{}'.format(info['Name']),show=info['Show'])
        folium.GeoJson(dataframe_geo,
            style_function= lambda x: {
                    'fillColor': info['Fill_Color'],
                    'color': info['Line_Color'],
                    'weight': info['Line_Width'],
                    'fillOpacity': info['Fill_Opacity']
                },
            tooltip=folium.GeoJsonTooltip(
                fields=list(dataframe_geo.columns.intersection(info['Data_Names'])),
                aliases=info['Label_Names'],
                localize=True
            )
        ).add_to(feature_info) 
        feature_info.add_to(self.map)
    def _layer_control(self):
        folium.LayerControl(collapsed=True).add_to(self.map)

    def show(self):
        return self.map


    def _geotiff(self,info):
        from rasterio.windows import Window
        import rasterio as rs
        from pyproj import Proj
        from math import floor, ceil
        from rasterio.transform import guard_transform
        import numpy as np

        def windowFunc(lon, lat, dataset):
            p = Proj(dataset.crs)
            t = dataset.transform
            xmin, ymin = p(lon[0], lat[0])
            xmax, ymax = p(lon[1], lat[1])
            col_min, row_min = ~t * (xmin, ymin)
            col_max, row_max = ~t * (xmax, ymax)


            window = Window.from_slices(rows=(floor(row_max), ceil(row_min)),cols=(floor(col_min), ceil(col_max)))
            transform = guard_transform(dataset.transform)
            return window,rs.windows.bounds(window, transform)

        dataset = rs.open(info['filename'])
        resamplingFactor = info['Resampling_Factor']
        window=None
        bnds = [[dataset.bounds[1], dataset.bounds[0]], [dataset.bounds[3], dataset.bounds[2]]]
        trying =True
        indx = 1
        while trying:
            try:
                img = folium.raster_layers.ImageOverlay(
                    name="{} - Band {}".format(info['Name'],indx),
                    image=dataset.read(indx,window=window)[::resamplingFactor,::resamplingFactor],
                    bounds=bnds,
                    opacity=info['Opacity'],
                    mercator_project=True,
                    pixelated=False,
                    show = info['Show']
                )
                img.add_to(self.map)
            except:
                trying=False
            indx+=1