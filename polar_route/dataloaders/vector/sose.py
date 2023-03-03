from polar_route.dataloaders.vector.abstractVector import VectorDataLoader

import logging

import xarray as xr

class SOSEDataLoader(VectorDataLoader):

    def __init__(self, bounds, params):
        '''
        Initialises Baltic currents dataset. Does no post-processing
        
        Args:
            bounds (Boundary): 
                Initial boundary to limit the dataset to
            params (dict):
                Dictionary of {key: value} pairs. Keys are attributes 
                this dataloader requires to function
        '''
        logging.info("Initalising SOSE currents dataloader")
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
        Reads in data from a SOSE Currents NetCDF file. 
        Renames coordinates to 'lat' and 'long', and renames variable to 
        'uC, vC'
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            xr.Dataset: 
                SOSE currents dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'uC', 'vC'
        '''

        # Open dataset and cast to pandas df
        logging.info(f"- Opening file {self.file}")
        data = xr.open_dataset(self.file)

        df = data.to_dataframe().reset_index()
        
        # Change long coordinate to be in [-180,180) domain rather than [0,360)
        df['long'] = df['lon'].apply(lambda x: x-360 if x>180 else x)
        # Extract relevant columns
        df = df[['lat','long','uC','vC']]
        logging.info('- Limiting to initial bounds')
        # Limit to  values between lat/long boundaries
        df = df[df['long'].between(bounds.get_long_min(), bounds.get_long_max())]
        df = df[df['lat'].between(bounds.get_lat_min(), bounds.get_lat_max())]

        return df
