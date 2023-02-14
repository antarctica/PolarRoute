from .AbstractVector import VectorDataLoader

import logging

import xarray as xr

class NorthSeaCurrentDataLoader(VectorDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Set up data
        self.data = self.import_data(bounds)
        self.data_name = self.col_names_to_str() # = 'SIC'
        
    def import_data(self, bounds):
        logging.debug("Importing North Sea Current data...")
        logging.debug(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        # Change column names
        data = data.rename({'lon': 'long',
                            'times': 'time',
                            'U': 'uC',
                            'V': 'vC'})
        # Limit to just these coords and variables
        data = data[['uC','vC']]
        
        # data = data.assign_coords(time=data.times.)
        
        # Limit to initial boundary
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),
                                   bounds.get_long_max()))
        data = data.sel(time=slice(bounds.get_time_min(), 
                                   bounds.get_time_max()))
        
        return data
