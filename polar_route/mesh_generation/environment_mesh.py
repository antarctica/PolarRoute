import json
import logging
import geopandas as gpd
import pandas as pd
from shapely import wkt
import numpy as np


from polar_route.mesh_generation.jgrid_aggregated_cellbox import JGridAggregatedCellBox
from polar_route.mesh_generation.boundary import Boundary
from polar_route.mesh_generation.aggregated_cellBox import AggregatedCellBox
from polar_route.mesh_generation.neighbour_graph import NeighbourGraph

from osgeo import gdal, ogr

from polar_route.mesh_validation.sampler import Sampler


class EnvironmentMesh:
    """
    a class that defines the environmental mesh structure and contains each cellbox aggregate information


    Attributes:
        bounds (Boundary): the boundaries of this mesh 
        agg_cellboxes (AggregatedCellBox[]): a list of aggregated cellboxes
        neighbour_graph(NeighbourGraph): an object contains each cellbox neighbours information 
        config (dict): conatins the initial config used to build this mesh



    """
    @classmethod
    def load_from_json(cls, mesh_json):
        """
            Constructs an Env.Mesh from a given env-mesh json file to be used by other modules (ex.Vessel Performance Modeller).

            Args:
                mesh_json (dict): a dictionary loaded from an Env-mesh json file of the following format - \n
                    \n
                    {\n
                        "config": {\n
                            "Mesh_info":{\n
                                "Region": {\n
                                    "latMin": (real),\n
                                    "latMax": (real),\n
                                    "longMin": (real),\n
                                    "longMax": (real),\n
                                    "startTime": (string) 'YYYY-MM-DD',\n
                                    "endTime": (string) 'YYYY-MM-DD',\n
                                    "cellWidth": (real),\n
                                    "cellHeight" (real),\n
                                    "splitDepth" (int)\n
                                },\n
                                "Data_sources": [\n
                                    {\n
                                        "loader": (string)\n
                                        "params" (dict)\n
                                    },\n
                                    ...,\n
                                    {...}
                                    ], \n
                                "splitting":\n
                                {\n
                                     "split_depth": (int),\n
                                     "minimum_datapoints": (int)\n
                                }\n
                                "cellboxes": [\n
                                    {\n

                                    },\n
                                    ...,\n
                                    {...}\n

                                ],\n
                                "neighbour_graph": [\n
                                    {\n

                                    },\n
                                    ...,\n
                                    {...}\n
                                ]\n
                            }\n
                        }\n
                    }\n


            Returns:
                EnvironmentMesh: object that contains all the json file mesh information. \n


        """
        config = mesh_json['config']
        cellboxes_json = mesh_json['cellboxes']
        agg_cellboxes = []
        bounds = Boundary.from_json(config)
        # load the agg_cellboxes
        for cellbox_json in cellboxes_json:
            agg_cellbox = AggregatedCellBox.from_json(cellbox_json)
            agg_cellboxes.append(agg_cellbox)
        neighbour_graph = NeighbourGraph.from_json(
            mesh_json['neighbour_graph'])
        obj = EnvironmentMesh(bounds, agg_cellboxes, neighbour_graph, config)
        return obj

    def __init__(self, bounds, agg_cellboxes, neighbour_graph, config):
        """

            Args:
              bounds (Boundary): the boundaries of this mesh 
              agg_cellboxes (AggregatedCellBox[]): a list of aggregated cellboxes
              neighbour_graph(NeighbourGraph): an object contains each cellbox neighbours information 
              config (dict): conatins the initial config used to build this mesh

        """

        self.bounds = bounds
        self.agg_cellboxes = agg_cellboxes
        self.neighbour_graph = neighbour_graph
        self.config = config

    def to_json(self):
        """
            Returns this Mesh converted to a JSON object.

            Returns:
                json: a string representation of the CellGird parseable as a JSON object. The JSON object is of the form -

                    {\n
                        "config": the config used to initialize the Mesh,\n
                        "cellboxes": a list of CellBoxes contained within the Mesh,\n
                        "neighbour_graph": a graph representing the adjacency of CellBoxes within the Mesh\n
                    }\n
        """
        output = dict()
        output['config'] = self.config
        output["cellboxes"] = self.cellboxes_to_json()
        output['neighbour_graph'] = self.neighbour_graph.get_graph()

        return json.loads(json.dumps(output))
    
    def to_geojson(self):
        """
            Returns the cellboxes of this mesh converted to a geoJSON format.

            Returns:
                geojson: The cellboxes of this mesh in a geoJSON format

            NOTE:
                geoJSON format does not contain all the data included in the standard 
                .to_json() format. geoJSON meshes do not contain the configs used to 
                build them, or the neighbour-graph which details how each of the 
                cellboxes are connected together.
        """
        geojson = ""
        mesh_json = self.to_json()

        # Formatting mesh to geoJSON
        mesh_df = pd.DataFrame(mesh_json['cellboxes'])
        mesh_df['geometry'] = mesh_df['geometry'].apply(wkt.loads)
        mesh_gdf = gpd.GeoDataFrame(mesh_df, crs = "EPSG:4326", geometry="geometry")
        geojson = json.loads(mesh_gdf.to_json())

        return geojson

    def to_tif(self, data_name, sampling_resolution, path, projection = "EPSG:4326" , color_map="blues" ):
        """
            generates a representation of the mesh in geotif image format.
            Args:

                data_name(string): the name of the mesh data that will be included in the tif image (ex. SIC, elevation)
                sampling_resolution (tuple): the sampling resolution the geotiff will be generated at (how many pixels in the final image)
                projection (string): the sampling projection used to create the geotiff image
                colour_map (string): the name of the colour_map that will be used to display the scalar value in the generated geotif image (ex. Greys, Greens).
                path (string): the path to save the generated tif image


            NOTE:
                geotif format does not contain all the data included in the standard 
                .to_json() format. It contains only a visual representation of the values specified 
                in 'data_name' argument (ex. SIC, elevation)

        """
        #define the data extent as the mesh bounds
        extent = [self.bounds.get_min_long() , self.bounds.get_min_lat() , self.bounds.get_max_long() , self.bounds.get_max_lat()]

        # create raster band and populate with sampled data of image_size (sampling_resolution)
        # Get GDAL driver GeoTiff
        driver = gdal.GetDriverByName('GTiff')
        
        # Get image dimensions
        nlines = sampling_resolution[0]
        ncols = sampling_resolution[1]
    
        data_type = gdal.GDT_Float32
        #sampling data based on the image size
        sampler = Sampler (2 , nlines*ncols)
        ranges = [[self.bounds.get_min_lat(), self.bounds.get_max_lat()],[self.bounds.get_min_long(), self.bounds.get_min_long() ]]
        samples = sampler.generate_samples(ranges)
        def get_sample_value (sample):
                lat =  sample[0]
                long = sample[1]
                for agg_cellbox in self.agg_cellboxes:
                    if agg_cellbox.contains_point(lat , long):
                        value =  agg_cellbox.agg_data [data_name] #get the agg_value 
                        break  # break to make sure we avoid getting multiple values (for lat and long on the bounds of multiple cellboxes)
          
         
                return value
        data = []
        for sample in samples:
            data = np.append (data, get_sample_value(sample))
        
    
        # Create a temp grid
        #options = ['COMPRESS=JPEG', 'JPEG_QUALITY=80', 'TILED=YES']
        grid_data = driver.Create('grid_data', ncols, nlines, 1, data_type)#, options)
        
        # Write data for each bands
        grid_data.GetRasterBand(1).WriteArray(data)
        
        #create Spatial Reference System
        srs = ogr.SpatialReference()
        srs.ImportFromProj4('+init=EPSG:4326')
        #TODO: handle different projections
       # if projection != "EPSG:4326":
            # map projection to "EPSG:4326"

        def get_geo_transform(extent, nlines, ncols):
            resx = (extent[2] - extent[0]) / ncols
            resy = (extent[3] - extent[1]) / nlines
            return [extent[0], resx, 0, extent[3] , 0, -resy]
        
        # Setup projection and geo-transform
        grid_data.SetProjection(srs.ExportToWkt())
        grid_data.SetGeoTransform(get_geo_transform(extent, nlines, ncols))
        # Save the file
        file_name = path+ 'mesh_'+ data_name +'.tif'
        print(f'Generated GeoTIFF: {file_name}')
        driver.CreateCopy(file_name, grid_data, 0)  
 
        # Close the file
        driver = None
        grid_data = None
 
        # Delete the temp grid
        import os                
        os.remove('grid_data')

       
        
    def cellboxes_to_json(self):
        """
            returns a list of dictionaries containing information about each cellbox
            in this Mesh.
            all cellboxes will include id, geometry, cx, cy, dcx, dcy

            Returns:
                cellboxes (list<dict>): a list of CellBoxes which form the Mesh.
                    CellBoxes are of the form -

                    {
                        "id": (string) ... \n
                        "geometry": (string) POLYGON(...), \n
                        "cx": (float) ..., \n
                        "cy": (float) ..., \n
                        "dcx": (float) ..., \n
                        "dcy": (float) ..., \n
                        \n
                        "value_1": (float) ..., \n
                        ..., \n
                        "value_n": (float) ... \n
                    }
        """

        cellboxes_json = []
        for cellbox in self.agg_cellboxes:

            # Get json for CellBox
            cell = cellbox.to_json()

            cellboxes_json.append(cell)
        return cellboxes_json

    def update_cellbox(self, index, values):
        """
            method that adds values to the dict of a cellbox at certain index (to be used by the vessel perf. module to add the perf. metrics to the cellbox)

            Args:
              index (int): the index of the cellbox to be updated
              values (dict): a dict contains perf. metrics names and values

        """
        if index > -1 or index < len(self.agg_cellboxes):
            self.agg_cellboxes[index].agg_data.update(values)
        else:
            raise ValueError('Invalid cellbox index')

    def save(self, path, format="JSON"):
        """
            Saves this object to a location in local storage. 

            Args:
                path (String): The file location the mesh will be saved to.
                format (String) (optional): The format the mesh will be saved in.
                    If not format is given, default is JSON.
                    Supported formats are:
                        JSON
                        GEOJSON
        """
        

        logging.info(f"- saving the environment mesh to {path}")
        with open(path, 'w') as f:
            if format.upper() == "JSON":
                logging.info(f"Saving mesh in {format} format")
                json.dump(self.to_json(), f)
            elif format.upper() == "GEOJSON":
                logging.info(f"Saving mesh in {format} format")
                json.dump(self.to_geojson(), f)
            else:
                logging.warning(f"Cannot save mesh in a {format} format")

            if isinstance(self.agg_cellboxes[0], JGridAggregatedCellBox):
               dump_path = path.replace (".json" , ".dump")
               with open(dump_path, 'w') as dump_f:
                    self.dump_mesh(dump_f)

    def dump_mesh(self, file):
        """
            creates a string representation of this Mesh which
            is then saved to a file location specified by parameter
            'file'
            for use in j_grid regression testing
        """
        mesh_dump_str = ""
        for cell_box in self.agg_cellboxes:
            if isinstance(cell_box, JGridAggregatedCellBox):
                mesh_dump_str += cell_box.mesh_dump()

        file.write(mesh_dump_str)
        file.close()
