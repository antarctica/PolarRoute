from .AbstractScalar import ScalarDataLoader

import logging

import xarray as xr


class MODISDataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Set up data
        self.data = self.import_data(bounds)
        self.data_name = self.get_data_col_name() # = 'SIC'
        
    def import_data(self, bounds):
        logging.debug("Importing MODIS Sea Ice data...")
        logging.debug(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        # Change column name
        data = data.rename({'iceArea': 'SIC'})

        # Set areas obscured by cloud to NaN values
        data = data.where(data.cloud != 1, drop=True)
        
        # Limit to initial boundary
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),
                                   bounds.get_long_max()))
        data = data.sel(time=slice(bounds.get_time_min(), 
                                   bounds.get_time_max()))
        
        return data