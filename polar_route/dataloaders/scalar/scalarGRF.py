from polar_route.dataloaders.scalar.abstractScalar import ScalarDataLoader
from polar_route.utils import gaussian_random_field

import logging
import pandas as pd
import numpy as np

class ScalarGRFDataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        '''
        Generates a dataset using a gaussian random field
       Args:
            bounds (Boundary): 
                Initial boundary to limit the dataset to
            params (dict):
                Dictionary of {key: value} pairs. Keys are attributes 
                this dataloader requires to function
        '''
        logging.info("Initalising Scalar CSV dataloader")
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            logging.debug(f"self.{key}={val} (dtype={type(val)}) from params")
            setattr(self, key, val)
        
        # Import data
        self.data = self.import_data(bounds)
        
        # Get data name from column name if not set in params
        if self.data_name is None:
            logging.debug('- Setting self.data_name from column name')
            self.data_name = self.get_data_col_name()
        # or if set in params, set col name to data name
        else:
            logging.debug(f'- Setting data column name to {self.data_name}')
            self.data = self.set_data_col_name(self.data_name)
        
    def import_data(self, bounds):
        '''
        Creates data in the form of a Gaussian Random Field
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            pd.DataFrame: 
                Scalar dataset within limits of bounds. Dataset has coordinates
                'lat', 'long', and variable name 'data'. 
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
        data = pd.DataFrame(rows)
        
        return data