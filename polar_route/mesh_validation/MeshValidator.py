

from polar_route.mesh_validation.Sampler import Sampler

import xarray as xr
from polar_route.MeshBuilder import MeshBuilder
from polar_route.mesh import Mesh
import numpy as np
import json
from polar_route.Boundary import Boundary
class MeshValidator:

    def __init__ (self , mesh_config_file):
        self.conf = None
        self.data = {}
        self.validation_length = 2 # the legnth of the validation square used for each sample, all the data_points contained within this square wil be validated. Higher values would incur higher processing cost

        with open (mesh_config_file , "r") as config_file:
            self.conf = json.load(config_file)['config']

        mesh_builder = MeshBuilder (self.conf)
        self.mesh = mesh_builder.mesh
    
       

    def validate_mesh (self , number_of_samples=10):
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
        print (actual_value)
        print (mesh_value)
            
        # calculate the RMSE over the samples.
        MSE = np.square(np.subtract(actual_value,mesh_value)).mean()
        return MSE
    

    def get_value_from_data (self , sample):
        values =[]
        lat_range = [sample[0] , sample[0] + self.validation_length]
        long_range = [sample[1] , sample[1] + self.validation_length]
        time_range = self.mesh.get_bounds().get_time_range()
        for source in self.mesh.cellboxes[0].get_data_source():
            data_loader = source.get_data_loader() 
            print (sample)
            dp = data_loader.get_datapoints( Boundary (lat_range , long_range , time_range))
            print (">>> value >> " , dp)
            values = np.append (values , dp)
        return values


    def get_values_from_mesh (self , sample):
            #TODO make sure to handle the vector data
            values = []
            lat_range = [sample[0] , sample[0] + self.validation_length]
            long_range = [sample[1] , sample[1] + self.validation_length]
            time_range = self.mesh.get_bounds().get_time_range()

        
            for source in self.mesh.cellboxes[0].get_data_source():
                data_loader = source.get_data_loader()
                dp = data_loader.get_datapoints( Boundary (lat_range , long_range , time_range), with_coord = True)
                for point in dp :
                    lat = data_loader.data.loc[point.index].lat
                    long = data_loader.data.loc[point.index].long
                    for cellbox in self.mesh.cellboxes:
                        if cellbox.contains_point(lat , long):
                             values = np.append ( values , data_loader.get_value (cellbox.bounds)[data_loader.data_name] )#get the agg_value 
                             break  # break to make sure we avoid getting multiple values (for lat and long on bounds of multiple cellboxes)


            return values
    

        
        



