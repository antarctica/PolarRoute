import json
import logging
import geopandas as gpd
import pandas as pd
from shapely import wkt
import numpy as np
import subprocess
import sys
import os
import tempfile

from polar_route.mesh_generation.jgrid_aggregated_cellbox import JGridAggregatedCellBox
from polar_route.mesh_generation.boundary import Boundary
from polar_route.mesh_generation.aggregated_cellbox import AggregatedCellBox
from polar_route.mesh_generation.neighbour_graph import NeighbourGraph


from polar_route.mesh_validation.sampler import Sampler
import collections.abc
import math
from pathlib import Path

class EnvironmentMesh:
    """
        a class that defines the environmental mesh structure and contains each cellbox aggregate information

        Attributes:
            bounds (Boundary): the boundaries of this mesh 
            agg_cellboxes (AggregatedCellBox[]): a list of aggregated cellboxes
            neighbour_graph(NeighbourGraph): an object contains each cellbox neighbours information 
            config (dict): conatins the initial config used to build this mesh\n
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
                EnvironmentMesh: object that contains all the json file mesh information.\n
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
              config (dict): conatins the initial config used to build this mesh.\n
        """

        self.bounds = bounds
        self.agg_cellboxes = agg_cellboxes
        self.neighbour_graph = neighbour_graph
        self.config = config

    def to_json(self):
        """
            Returns this Mesh converted to a JSON object.\n

            Returns:
                json: a string representation of the CellGird parseable as a JSON object. The JSON object is of the form -\n

                    {\n
                        "config": the config used to initialize the Mesh,\n
                        "cellboxes": a list of CellBoxes contained within the Mesh,\n
                        "neighbour_graph": a graph representing the adjacency of CellBoxes within the Mesh.\n
                    }\n
        """
        output = dict()
        output['config'] = self.config
        output["cellboxes"] = self.cellboxes_to_json()
        output['neighbour_graph'] = self.neighbour_graph.get_graph()


        return json.loads(json.dumps(output, indent=4))
    

    def to_geojson(self, params_file = None):
        """
            Returns the cellboxes of this mesh converted to a geoJSON format.\n

            Returns:
                geojson: The cellboxes of this mesh in a geoJSON format

            NOTE:
                geoJSON format does not contain all the data included in the standard 
                .to_json() format. geoJSON meshes do not contain the configs used to 
                build them, or the neighbour-graph which details how each of the 
                cellboxes are connected together.\n
        """
        geojson = ""
        mesh_json = self.to_json()

        if (params_file != None):
            with open(params_file) as f:
                data = f.read()
                format_params = json.loads(data)
            data_name = format_params['data_name']
            logging.info("exporting layer : " + str(data_name))

        # Formatting mesh to geoJSON
        mesh_df = pd.DataFrame(mesh_json['cellboxes'])

        for column in mesh_df.columns:

            if column in ['id', 'geometry']:
                continue
            # remove unnecessary columns
            if (params_file != None) and column not in [str(data_name)]:
                mesh_df = mesh_df.drop(column, axis=1)
            # remove unnecessary columns
            elif column in ['cx', 'cy', 'dcx', 'dcy']:
                mesh_df = mesh_df.drop(column, axis=1)
            # convert lists to mean
            elif mesh_df[column].dtype == list:
                mesh_df[column] = [np.mean(x) for x in mesh_df[column]]
            # convert bools to ints
            elif mesh_df[column].dtype == bool:
                mesh_df[column] = mesh_df[column].astype(int)
                mesh_df[column] = mesh_df[column].replace(0, np.nan)
            
        # Remove infs and replace with nan
        mesh_df = mesh_df.replace([np.inf, -np.inf], np.nan)

        mesh_df['geometry'] = mesh_df['geometry'].apply(wkt.loads)
        mesh_gdf = gpd.GeoDataFrame(
            mesh_df, crs="EPSG:4326", geometry="geometry")
        geojson = json.loads(mesh_gdf.to_json())

        return geojson

    def to_tif(self, params_file,  path):
        """
                generates a representation of the mesh in geotif image format.\n

                Args:
                    params_file(string) (optional): a path to a file that contains a dict of the folowing export parameters (If not given, default values are used for image export). The file should be of the following format -\n
                            \n
                            {\n
                                "data_name": "elevation",\n
                                "sampling_resolution": [\n
                                    150,\n
                                    150\n
                                 ],\n
                                "projection": "3031",\n
                            }\n
                            Where data_name (string) is the name of the mesh data that will be included in the tif image (ex. SIC, elevation), if it is a vector data (e.g. fuel) then the vector mean is calculated for each pixel,
                            sampling_resolution ([int]) is a 2d array that represents the sampling resolution the geotiff will be generated at (how many pixels in the final image),
                            projection (int) is an int representing the ESPG sampling projection used to create the geotiff image  (default is 4326),
                            and colour_conf (string) contains the path to color config file, which is a text-based file containing the association between data_name values and colors. It contains 4 columns per line: the data_name value and the corresponding red, green, blue value between 0 and 255, an example format where values range from 0 to 100 is -\n
                                    \n
                                    0 240 250 160  \n
                                    30 230 220 170  \n
                                    60 220 220 220 \n
                                    100 250 250 250  \n
                    path (string): the path to save the generated tif image.\n
        """
        def generate_samples():
            """
                generates uniform lat, long samples covering the image resolution space.\n

                Returns:
                    samples([[lat,long],..]): an array of samples, each item in the array is a 2d array that contains each sample lat and long.\n

            """
            mesh_height = self.bounds.get_lat_max() - self.bounds.get_lat_min()
            mesh_width = self.bounds.get_long_max() - self.bounds.get_long_min()
            pixel_height = mesh_height / ncols
            pixel_width = mesh_width / nlines
            samples = []
            # has to move in this direction as we start rendering from the upper left pixel
            for lat in np.arange(self.bounds.get_lat_max(), self.bounds.get_lat_min(), -1*pixel_height):
                for long in np.arange(self.bounds.get_long_min(), self.bounds.get_long_max(), pixel_width):
                    pixel_lat = lat - 0.5 * pixel_height   # centeralize the pixel lat value
                    pixel_long = long + 0.5*pixel_width   # centeralize the pixel long value
                    samples = np.append(samples, pixel_lat)
                    samples = np.append(samples, pixel_long)
            # shape the samples in 2d array (each entry in the array holds sample lat and long
            samples = np.reshape(samples, (nlines * ncols, 2))
            return samples

        def get_sample_value(sample):
            """
                finds the aggregated cellbox that contains the sample lat and long and returns the value within.\n

                Args:
                    sample ([lat,long]): an array conatins the sample latitude and longtitude
                Returns:
                    the aggregated value of 'data_name'(specified in to_tif params) 

             """
            lat = sample[0]
            long = sample[1]
            value = np.nan
            for agg_cellbox in self.agg_cellboxes:
                if agg_cellbox.contains_point(lat, long):
                    # get the agg_value
                    value = agg_cellbox.agg_data[data_name]
                    if isinstance(value, collections.abc.Sequence): # if it is a vector then take the mean
                        value = np.mean (value)
                        if value == float('inf') : # repalce inf with nan
                            value = np.nan
                    # break to make sure we avoid getting multiple values (for lat and long on the bounds of multiple cellboxes)
                    break
            return value

        def get_geo_transform(extent, nlines, ncols):
            """
                transforms from the image coordinate space (row, column) to the georeferenced coordinate space. \n
                Returns:    
                  GT : array consists of 6 items representing how GDAL would place the top left pixel on the generated Geotiff:\n
                  GT[0] x-coordinate of the upper-left corner of the upper-left pixel.\n
                  GT[1] w-e pixel resolution / pixel width.\n
                  GT[2] row rotation (typically zero).\n
                  GT[3] y-coordinate of the upper-left corner of the upper-left pixel.\n
                  GT[4] column rotation (typically zero).\n
                  GT[5] n-s pixel resolution / pixel height (negative value for a north-up image).\n

            """
            resx = (extent[2] - extent[0]) / ncols
            resy = (extent[3] - extent[1]) / nlines
            return [extent[0], resx, 0, extent[3], 0, -resy]

        def load_params(params_file):
            """
                  loads the parameters of the tif export and override the default values.\n

                  Args:
                      params_file (string): a path to a file containing a dict of the export params 
                  
                  Returns:
                      params (dict): a dict object that contains the loaded parameters

            """
            params = {"data_name": "SIC", "sampling_resolution": [
                100, 100], "projection": "4326"}  # the default values
            if (params_file != None):
                with open(params_file) as f:
                    data = f.read()
                    input_params = json.loads(data)
                if (input_params != None):
                    if ("projection" in input_params.keys()):
                        params["projection"] = input_params["projection"]
                    if ("data_name" in input_params.keys()):
                        params["data_name"] = input_params["data_name"]
                    if ("sampling_resolution" in input_params.keys()):
                        params["sampling_resolution"] = input_params["sampling_resolution"]
                    if ("colour_conf" in input_params.keys()):
                        params["colour_conf"] = input_params["colour_conf"]
            return params

        def transform_proj(path, params, default_proj):
            """
                  method that transforms the generated tif into another projection

                  Args:
                        path(string): the path of the generated tif
                        params (dict): a dict that contains the export parametrs
                        default_proj (stirng): a string represents the default projection (EPSG:4326).\n

            """
            if params["projection"] != str(default_proj):
                dest = osr.SpatialReference()
                dest.ImportFromEPSG(int(params["projection"]))
                # transform to target proj and save
                gdal.Warp(str(path),  str(path), dstSRS=dest.ExportToWkt())
        
        def set_colour(data, input_file, params):
            """
                  method that changes the color of the generated tif instead of using the default greyscale.
                  It defines a scale of RGB colors based on the range of data values(an example file is in unit_tests/resources/colour_conf.txt).\n

                  Args:
                        data ([float]): an array conatins the values of the 'data_name'
                        input_path(string): the path of the generated grey tif
                        params (dict): a dict that contains the export parametrs.\n

            """
            fp, color_file = tempfile.mkstemp(suffix='.txt')
            data = data[~np.isnan(data)]  # get rid of the nans before calculating range
            _max = np.nanmax(data)
            _min = np.nanmin(data)
            _range = _max-_min
            inf_color = '255 0 0' # red color for inf value
            colors = ['255 255 255', '173 216 230', '0 0 128', '0 0 139']  # default color
            with open(color_file, 'w') as f:
                for i, c in enumerate(colors[:-1]):
                    f.write(str(int(_min + (i + 1)*_range/len(colors))) +
                            ' ' + c + '\n')
                f.write(str(int(_max - _range/len(colors))) +
                        ' ' + colors[-1] + '\n')
                f.write (str(np.nan) +
                        ' ' + inf_color + '\n') # render nans in red
            os.close(fp)
            if "colour_conf" in params.keys():
                color_file = params["colour_conf"]
            cmd = "gdaldem color-relief " + input_file \
                + ' ' + color_file + ' ' + input_file
            subprocess.check_call(cmd, shell=True)
            # remove additional files created while generating the tif
            file_path = os.path.abspath(input_file)
            dir_path = os.path.split(file_path)[0]
            additional_files = ["grid_data" , "grid_data.aux.xml"]
            for file in additional_files: 
                file_path = Path(dir_path +"/"+ file)
                if os.path.isfile (file_path):
                     os.remove(file_path)

        # Only import if we need GDAL, to avoid having it as a requirement
        from osgeo import gdal, ogr, osr
        
        params = {}
        params = load_params(params_file)
        data_name = params["data_name"]
        DEFAULT_PROJ = 4326

        # Get image dimensions
        nlines = params["sampling_resolution"][0]
        ncols = params["sampling_resolution"][1]

        # define image extent based on mesh bounds
        extent = [self.bounds.get_long_min(), self.bounds.get_lat_min(
        ), self.bounds.get_long_max(), self.bounds.get_lat_max()]

        logging.info("Generating the tif image ...")
        samples = generate_samples()
        # create raster band and populate with sampled data of image_size (sampling_resolution)
        driver = gdal.GetDriverByName('GTiff')
        # reading the samples value
        data = []
        data = np.reshape(np.asarray([get_sample_value(
            sample) for sample in samples], dtype=np.float32), (nlines, ncols))
        # create a temp grid
        grid_data = driver.Create(
            'grid_data', ncols, nlines, 1, gdal.GDT_Float32)
        # setup geo-transform
        grid_data.SetGeoTransform(get_geo_transform(extent, nlines, ncols))
        # Write data
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(DEFAULT_PROJ)
        grid_data.SetProjection(srs.ExportToWkt())
        grid_data.GetRasterBand(1).WriteArray(data)
    

        # Save the file
        driver.CreateCopy(path, grid_data, 0)
        transform_proj(path, params, DEFAULT_PROJ)
        set_colour(data, path, params)
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
            Saves this object to a location in local storage in a specific format. 

            Args:
                path (String): The file location the mesh will be saved to.
                format (String) (optional): The format the mesh will be saved in.
                    If not format is given, default is JSON.
                    Supported formats are\n
                        - JSON \n
                        - GEOJSON
        """

        logging.info(f"Saving mesh in {format} format to {path}")
        if format.upper() == "TIF":
            self.to_tif(format_params, path)

        elif format.upper() == "JSON":
            with open(path, 'w') as path:
                json.dump(self.to_json(), path, indent=4)
           
        elif format.upper() == "GEOJSON":
            with open(path, 'w') as path:
                json.dump(self.to_geojson(format_params), path, indent=4)

        else:
            logging.warning(f"Cannot save mesh in a {format} format")

        if isinstance(self.agg_cellboxes[0], JGridAggregatedCellBox):
            dump_path = path.replace(".json", ".dump")
            with open(dump_path, 'w') as dump_f:
                self.dump_mesh(dump_f)


    def dump_mesh(self, file):
        """
            creates a string representation of this Mesh which
            is then saved to a file location specified by parameter
            'file' for use in j_grid regression testing,

        """
        mesh_dump_str = ""
        for cell_box in self.agg_cellboxes:
            if isinstance(cell_box, JGridAggregatedCellBox):
                mesh_dump_str += cell_box.mesh_dump()

        file.write(mesh_dump_str)
        file.close()
