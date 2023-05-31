from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader
from polar_route.utils import gaussian_random_field

import logging
import pandas as pd
import numpy as np

class ScalarGRFDataLoader(ScalarDataLoader):
    
    def add_default_params(self, params):
        '''
        Set default values for abstract GRF dataloaders, starting by
        including defaults for scalar dataloaders.
        
        Args:
            params (dict): 
                Dictionary containing attributes that are required for the
                shape being loaded. Must include 'shape'.
            
        Returns:
            (dict): 
                Dictionary of attributes the dataloader will require, 
                completed with default values if not provided in config.
        '''
        params = super().add_default_params(params)
        
        # Params that all GRF dataloaders need
        if 'seed' not in params:
            params['seed'] = None
        if 'size' not in params:
            params['size'] = 512
        if 'alpha' not in params:
            params['alpha'] = 3
        
        # Set defaults for binary GRF
        if params['binary'] == True:
            if params['data_name'] is None:
                params['data_name'] = 'binary_data'
            if 'min' not in params:
                params['min'] = 0
            if 'max' not in params:
                params['max'] = 1
            # If threshold not set, make it average of min/max val
            if 'threshold' not in params:
                params['threshold'] = 0.5
        # Set defaults for smooth scalar GRF
        else:
            if params['data_name'] is None:
                params['data_name'] = 'scalar_data'
            if 'min' not in params:
                params['min'] = -10
            if 'max' not in params:
                params['max'] = 10
            # If threshold not set, make it min/max vals
            if 'threshold' not in params:
                params['threshold'] = [0,1]
            if 'multiplier' not in params:
                params['multiplier'] = 1
            if 'offset' not in params:
                params['offset'] = 0
                
        return params
        
    
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
    
        def grf_to_binary(grf, threshold):
            '''
            Creates a mask out of a GRF if params specify a binary output
            '''
            # Cast all above threshold to 1, below to 0
            grf[grf >= threshold] = 1.0
            grf[grf <  threshold] = 0.0
            # Cast 0/1 to False/True
            grf = (grf == True)
            return grf
        
        def grf_to_scalar(grf, threshold, min_val, max_val):
            # Bound GRF to threshold limits
            grf[grf <  threshold[0]] = threshold[0]
            grf[grf >= threshold[1]] = threshold[1]
            # Renormalise the GRF
            grf = grf - np.min(grf)
            grf = grf/(np.max(grf)-np.min(grf))
            # Scale to max/min
            grf = grf * (max_val - min_val) + min_val
            
            return grf
        
        # Set seed for generation. If not specified, will be 'random'
        np.random.seed(self.seed)
        
        # Create a GRF
        grf = gaussian_random_field(self.size, self.alpha)
        
        # Set it to a binary mask if chosen in config
        if self.binary == True:
            grf = grf_to_binary(grf, self.threshold)
        # Set it to scalar GRF field if chosen in config
        else:
            grf = grf_to_scalar(grf, self.threshold, self.min, self.max)
            # Scale with multiplier and offset
            grf = self.multiplier * grf + self.offset
    
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
            {'lat': latv[i,j], 'long': longv[i,j], 'data': grf[i,j]}
            for i in range(self.size) 
            for j in range(self.size)
            ]
        # Cast to dataframe
        data = pd.DataFrame(rows).set_index(['lat','long'])
        # Set to xarray dataset
        data = data.to_xarray()
        
        return data