from polar_route.dataloaders.vector.abstract_vector import VectorDataLoader
from polar_route.utils import gaussian_random_field

import logging
import pandas as pd
import numpy as np

class VectorGRFDataLoader(VectorDataLoader):
    def import_data(self, bounds):
        '''
        Creates data in the form of a Gaussian Random Field
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            pd.DataFrame: 
                Scalar dataset within limits of bounds. Dataset has coordinates
                'lat', 'long', and variable name 'data'. \n
                NOTE - In __init__, variable name will be set to data_name if 
                defined in config
                
        '''

        def grf_to_vector(magnitudes, directions, min_val, max_val):
            # Scale to max/min
            magnitudes = magnitudes * (max_val - min_val) + min_val
            
            dy = np.cos(directions) * magnitudes
            dx = np.sin(directions) * magnitudes
            
            return dx, dy
        
        # Set seed for generation. If not specified, will be 'random'
        np.random.seed(self.seed)
        
        # Create a GRF of magnitudes and angles
        magnitudes = gaussian_random_field(self.size, self.alpha)
        directions = gaussian_random_field(self.size, self.alpha)
        directions = np.radians(360*directions)
        
        vec_x, vec_y = grf_to_vector(magnitudes, directions, self.min, self.max)
        
    
        # Set up domain of field
        lat_array = np.linspace(bounds.get_lat_min(), 
                                bounds.get_lat_max(), 
                                self.size)
        long_array = np.linspace(bounds.get_long_min(), 
                                 bounds.get_long_max(), 
                                 self.size)
        latv, longv = np.meshgrid(lat_array, long_array, indexing='ij')

        # Create an entry for each row in final dataframe
        rows = [
            {'lat': latv[i,j], 'long': longv[i,j], 
             self.vec_x: vec_x[i,j], self.vec_y: vec_y[i,j]}
            for i in range(self.size) 
            for j in range(self.size)
            ]
        # Cast to dataframe
        data = pd.DataFrame(rows).set_index(['lat','long'])
        # Set to xarray dataset
        data = data.to_xarray()
        
        return data