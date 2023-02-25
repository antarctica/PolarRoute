from .AbstractVector import VectorDataLoader

import logging

import xarray as xr

class SOSEDataLoader(VectorDataLoader):

    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        self.data = self.import_data(bounds)
        self.data_name = self.get_data_col_name()
        
    def import_data(self, bounds):
        '''
        Load SOSE netCDF
        '''
        logging.debug("Importing SOSE data...")
        # Import raw data

        # Open dataset and cast to pandas df
        logging.debug(f"- Opening file {self.file}")
        data = xr.open_dataset(self.file)

        df = data.to_dataframe().reset_index()
        
        # Change long coordinate to be in [-180,180) domain rather than [0,360)
        df['long'] = df['lon'].apply(lambda x: x-360 if x>180 else x)
        # Extract relevant columns
        df = df[['lat','long','uC','vC']]
        # Limit to  values between lat/long boundaries
        df = df[df['long'].between(bounds.get_long_min(), bounds.get_long_max())]
        df = df[df['lat'].between(bounds.get_lat_min(), bounds.get_lat_max())]

        return df
