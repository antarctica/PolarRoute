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


from polar_route.mesh_validation.sampler import Sampler
from osgeo import gdal, ogr, osr

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

    def to_tif(self, params,  path):
        """
            generates a representation of the mesh in geotif image format.
            Args:
                params(dict): a dict that might contain the folowing params:
                        data_name(string): the name of the mesh data that will be included in the tif image (ex. SIC, elevation)
                        sampling_resolution (tuple): the sampling resolution the geotiff will be generated at (how many pixels in the final image)
                        projection (int): an int representing the ESPG sampling projection used to create the geotiff image  (default is 4326)
                        colour_table (GDALColorTable): an object that defines the colors used to display the scalar value in the generated geotif image.
                path (string): the path to save the generated tif image


            NOTE:
                geotif format does not contain all the data included in the standard 
                .to_json() format. It contains only a visual representation of the values specified 
                in 'data_name' argument (ex. SIC, elevation)

        """

        def generate_samples ():
            """
                generates uniform lat, long samples covering the image resolution space
                Returns:
                    samples([[lat,long],..]): an array of samples, each item in the array is a 2d array that contains each sample lat and long
            """
            mesh_height = self.bounds.get_lat_max() - self.bounds.get_lat_min()
            mesh_width = self.bounds.get_long_max() - self.bounds.get_long_min()
            pixel_height = mesh_height/ ncols
            pixel_width = mesh_width/ nlines
            samples = []
            for lat in np.arange(self.bounds.get_lat_max(), self.bounds.get_lat_min(), -1*pixel_height): # has to move in this direction as we start rendering from the upper left pixel
                for long in np.arange(self.bounds.get_long_min(), self.bounds.get_long_max(), pixel_width):
                    pixel_lat = lat - 0.5* pixel_height   # centeralize the pixel lat value 
                    pixel_long = long + 0.5*pixel_width   # centeralize the pixel long value
                    samples = np.append (samples , pixel_lat)
                    samples = np.append (samples , pixel_long)
            samples = np.reshape(samples , (nlines* ncols, 2)) # shape the samples in 2d array (each entry in the array holds sample lat and long
            return samples
        def get_sample_value (sample):
                """
                    finds the aggregated cellbox that contains the sample lat and long and returns the value within
                    Args:
                      sample ([lat,long]): an array conatins the sample latitude and longtitude
                    Returns:
                         the aggregated value of 'data_name'(specified in to_tif params) 
            
                 """
                lat =  sample[0]
                long = sample[1]
                value = np.nan
                for agg_cellbox in self.agg_cellboxes:
                    if agg_cellbox.contains_point(lat , long):
                        value =  agg_cellbox.agg_data [data_name] #get the agg_value 
                        break  # break to make sure we avoid getting multiple values (for lat and long on the bounds of multiple cellboxes)
                return value
        def get_geo_transform(extent, nlines, ncols):
            """
                transforms from the image coordinate space (row, column) to the georeferenced coordinate space 
                Returns:
                  GT : array consists of 6 items representing how GDAL would place the top left pixel on the generated Geotiff:
                    GT[0] x-coordinate of the upper-left corner of the upper-left pixel.
                    GT[1] w-e pixel resolution / pixel width.
                    GT[2] row rotation (typically zero).
                    GT[3] y-coordinate of the upper-left corner of the upper-left pixel.
                    GT[4] column rotation (typically zero).
                    GT[5] n-s pixel resolution / pixel height (negative value for a north-up image).

            """
            resx = (extent[2] - extent[0]) / ncols
            resy = (extent[3] - extent[1]) / nlines
            return [extent[0], resx, 0, extent[3] , 0, -resy]
        def validate_params(params):
          """
                validates that the parameters needed for the export are not missing
          """
          if (params == None):
             raise ValueError('Parameters missing! Can not save mesh in tif format with None parameters')
          if ( "data_name" not in params.keys() ):
             raise ValueError('data_name should be specified in the params while saving in tif format')
          if ( "sampling_resolution" not in params.keys() ):
             raise ValueError('sampling_resolution should be specified in the params while saving in tif format')

        validate_params(params)
        data_name = params["data_name"]
        DEFAULT_PROJ = 4326
        projection = DEFAULT_PROJ
        if "projection" in params.keys():
            projection = int (params["projection"])
    
        # Get image dimensions
        nlines = params["sampling_resolution"][0]
        ncols =  params["sampling_resolution"][1]

        #define image extent based on mesh bounds
        extent = [self.bounds.get_long_min(), self.bounds.get_lat_min(), self.bounds.get_long_max() , self.bounds.get_lat_max()]

        logging.info ("Generating the tif image ...")
        samples = generate_samples()
        # create raster band and populate with sampled data of image_size (sampling_resolution)
        # get GDAL driver GeoTiff
        driver = gdal.GetDriverByName('GTiff')
        # reading the samples value
        data= []
        data = np.reshape(np.asarray ([get_sample_value(sample) for sample in samples] , dtype=np.float32) , (nlines, ncols))
        # create a temp grid
        grid_data = driver.Create('grid_data', ncols, nlines, 1, gdal.GDT_Float32)
        # setup geo-transform
        grid_data.SetGeoTransform(get_geo_transform(extent, nlines, ncols))
        # Write data 
        srs = osr.SpatialReference() 
        srs.ImportFromEPSG(DEFAULT_PROJ)
        grid_data.SetProjection( srs.ExportToWkt())
        grid_data.GetRasterBand(1).WriteArray(data)
         
        #check if color table is provided
        if "color_table" in params.keys():
            grid_data.GetRasterBand(1).SetRasterColorTable( params["color_table"] )
 
        # Save the file
        driver.CreateCopy(path, grid_data, 0)
        if projection!=DEFAULT_PROJ:  
            dest = osr.SpatialReference() 
            dest.ImportFromEPSG(projection)
            # transform to target proj and save
            gdal.Warp(str(path),  str(path) ,dstSRS=dest.ExportToWkt())
        
        logging.info(f'Generated GeoTIFF: {path}')
   
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

    def save(self, path, format="JSON", format_params=None):
        """
            Saves this object to a location in local storage. 

            Args:
                path (String): The file location the mesh will be saved to.
                format (String) (optional): The format the mesh will be saved in.
                    If not format is given, default is JSON.
                    Supported formats are:
                        JSON
                        GEOJSON
                format_params (dict) (optional): a dict that contains format related parameters (ex. sampling_resolution/data_name for the tif format)
        """
        

        logging.info(f"- saving the environment mesh to {path}")
        if format.upper() == "TIF":
                self.to_tif(format_params , path)
        else:
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
