

from polar_route.mesh_validation.sampler import Sampler


from polar_route.mesh_generation.mesh_builder import MeshBuilder
from polar_route.mesh_generation.mesh import Mesh
import numpy as np
import json
import math
import logging
from polar_route.mesh_generation.boundary import Boundary
from sklearn.metrics import mean_squared_error
class MeshValidator:

    """
    a class that validates a constructed mesh against its actual sourced geo-spatial data. Validation takes place by comparing the aggregated data value of mesh's cellbox against the actual data contained within cellbox's bounds.
    Attributes:
        conf (dict): conatins the initial config used to build the mesh under vlaidation
        validation_length (float): the legnth of the validation square used for each sample, all the data_points contained within this square wil be validated. Higher values would incur higher processing cost
        mesh (Mesh): object that represents the constructed mesh (a representation before aggregating cellbox data, used to have access to the mesh data source to validate against the actual data)
        env_mesh (EnvironmentMesh): objects that represents the constructed env mesh (a representation after aggregating the mesh cellboox data)

    """

    def __init__ (self , mesh_config_file):
        """

            Args:
              mesh_config_file (String): the path to the config file used to build the mesh under validation

        """
        self.conf = None
        self.validation_length = .1 
        with open (mesh_config_file , "r") as config_file:
            self.conf = json.load(config_file)['config']

        mesh_builder = MeshBuilder (self.conf)
        self.env_mesh =  mesh_builder.build_environmental_mesh()
        self.mesh = mesh_builder.mesh

    
       

    def validate_mesh (self , number_of_samples=10):
        """

          samples the mesh's lat and long space and compares the actual data within the sampled's range to the mesh agg_value then calculates the RMSE.

            Args:
              number_of_samples (int): the number of samples used to validate the mesh
            Returns:
                distance (float): the RMSE between the actaul data value and the mesh's agg_value.

        """
        
        # read the mesh bounds then generate samples of lat and long within bounds
        SAMPLE_DIM = 2  # each sample contains lat and long
    
        bounds = self.mesh.get_bounds()
        samples = Sampler(SAMPLE_DIM , number_of_samples).generate_samples([bounds.lat_range , bounds.long_range])
        # compare the sampled lat and long values in data_file to the values obtained by mesh ( agg_values returned by  get_value)
        actual_value = np.array([])
        mesh_value = np.array([])
        for sample in samples:
           actual_value =  np.append (actual_value ,self.get_value_from_data (sample))
           mesh_value =  np.append ( mesh_value ,self.get_values_from_mesh(sample))  
        # calculate the RMSE over the samples.
        distance = math.sqrt (mean_squared_error(actual_value,mesh_value))
        return distance
    

    def get_value_from_data (self , sample):
        """
            gets the actual data within the provided sample lat and long
            Args:
              sample (float[]): a decimal array contains the sampled lat and long values
            Returns:
                a numpy array that contains all the data within the sampled lat and long range
        """
        values =[]
        #calculate the sampling range based on the validation length
        lat_end, long_end = self.get_range_end(sample)
        lat_range = [sample[0] , lat_end]
        long_range = [sample[1] , long_end ]
        time_range = self.mesh.get_bounds().get_time_range()
        for source in self.mesh.cellboxes[0].get_data_source():
            data_loader = source.get_data_loader() 
            dp = data_loader.get_datapoints( Boundary (lat_range , long_range , time_range))
            values = np.append (values , dp)
        logging.info("values from data are: {}".format(' '.join(map(str, values))))
        return values

    def get_range_end(self, sample):
        """
            calculates the range end of the provided sample lat and long, claculation is based on the specified validation_length
            Args:
              sample (float[]): a decimal array contains the sampled lat and long values
            Returns:
                float[]: lat and long range end
        """
        lat_end = sample[0] + self.validation_length
        long_end = sample[1] + self.validation_length
        # make sure we are not exceeding the mesh bounds
        if lat_end > self.mesh.get_bounds().get_lat_max():
            lat_end = self.mesh.get_bounds().get_lat_max()
        if long_end > self.mesh.get_bounds().get_long_max():
            long_end = self.mesh.get_bounds().get_long_max()
        return lat_end,long_end


    def get_values_from_mesh (self , sample):
            
            """
                finds the mesh's cellboxes that contains the sample's lat and long then returns the aggregated values within.
                Args:
                sample (float[]): a decimal array contains the sampled lat and long values
                Returns:
                    a numpy array that contains the mesh's data within the sampled lat and long range
            """
            #TODO make sure to handle the vector data
            values = []
            #calculate the sampling range based on the validation length
            lat_end, long_end = self.get_range_end(sample)
            lat_range = [sample[0] , lat_end]
            long_range = [sample[1] , long_end ]
            time_range = self.mesh.get_bounds().get_time_range()

        
            for source in self.mesh.cellboxes[0].get_data_source():
                data_loader = source.get_data_loader()
                dp = data_loader.get_datapoints( Boundary (lat_range , long_range , time_range), return_coords = True)
                for index, point in dp.iterrows():
                    lat = point ['lat']
                    long = point ['long']
                    for agg_cellbox in self.env_mesh.agg_cellboxes:
                        if agg_cellbox.contains_point(lat , long):
                             values = np.append ( values , agg_cellbox.agg_data [data_loader.data_name] )#get the agg_value 
                             break  # break to make sure we avoid getting multiple values (for lat and long on bounds of multiple cellboxes)
            logging.info("values from mesh are: {}".format(' '.join(map(str, values))))
         
            return values
    

        
        



