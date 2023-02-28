

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
       
        with open (mesh_config_file , "r") as config_file:
            self.conf = json.load(config_file)['config']

        mesh_builder = MeshBuilder (self.conf)
        self.mesh = mesh_builder.mesh
        self.load_data()
       

    def validate_mesh (self , number_of_samples=10):
        # read the mesh bounds then generate samples of lat and long within bounds
        SAMPLE_DIM = 2  # each sample contains lat and long
    
        bounds = self.mesh.get_bounds()
        samples = Sampler(SAMPLE_DIM , number_of_samples).generate_samples([bounds.lat_range , bounds.long_range])
        # compare the sampled lat and long values in data_file to the values obtained by mesh ( agg_values returned by  get_value)
        actual_value = np.array([])
        mesh_value = np.array([])
        for sample in samples:
            np.append (actual_value ,self.get_value_from_data (sample))
            np.append ( mesh_value ,self.get_values_from_mesh(sample))
            
        # calculate the RMSE over the samples.
        MSE = np.square(np.subtract(actual_value,mesh_value)).mean()
        return MSE
    

    def get_value_from_data (self , sample):
        values =[]
        for source in self.mesh.cellboxes[0].get_data_source():
            data_loader = source.get_data_loader() 
            data_name = data_loader.data_name
        # select sample lat and long
            value = self.data[data_name].sel(lat=sample[0])
            value = value.sel(long=sample[1])
            np.append (values , value)
        return values


    def get_values_from_mesh (self , sample):
            values = []
            for cellbox in self.mesh:
                 if cellbox.contains_point(sample[0] , sample[1]):
                    for source in cellbox.get_data_source():
                      data_loader = source.get_data_loader()
                      np.append ( values , data_loader.get_value (cellbox.bounds)[data_loader.get_data_name()] )#get the agg_value 

            return values
    
    def load_data (self):
        for source in self.mesh.cellboxes[0].get_data_source():
            data_loader = source.get_data_loader()
            data_file = data_loader.file

             # Open Dataset
            data = xr.open_dataset(data_file)
            data = data.rename({'lon':'long'})
          
            #TODO check if we can merge datasets better
            # Limit to initial boundary
            data = data.sel(lat=slice(self.mesh.get_bounds().get_lat_min(),self.mesh.get_bounds().get_lat_max()))
            self.data[data_loader.data_name] = data.sel(long=slice(self.mesh.get_bounds().get_long_min(),self.mesh.get_bounds().get_long_max()))
        
        



