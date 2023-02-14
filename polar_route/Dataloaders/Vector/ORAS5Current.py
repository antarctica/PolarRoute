from .AbstractVector import VectorDataLoader

import logging

import xarray as xr
import numpy as np

#TODO Read in 2 files, combine to one object
class ORAS5CurrentDataLoader(VectorDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Set up data
        self.data = self.import_data(bounds)
        self.data_name = self.col_names_to_str() # = 'SIC'
        
    def import_data(self, bounds, binsize=0.25):
        logging.debug("Importing ORAS5 Current data...")
        
        # Open Dataset
        logging.debug(f"- Opening zonal velocity file {self.file_u}")
        data_u = xr.open_dataset(self.file_u)
        
        
        logging.debug(f"- Opening meridional velocity file {self.file_v}")
        data_v = xr.open_dataset(self.data_v)
        
        # Ensure time frame of both zonal/meridional components match
        assert (data_u.time_counter.values == data_v.time_counter.values), \
            'Timestamp of input files do not match!'
            
        # Set domain of new coordinates to bin within
        lat_min = np.floor(min(data_u.nav_lat.min(), data_v.nav_lat.min()))
        lat_max = np.ceil(max(data_u.nav_lat.max(), data_v.nav_lat.max()))
        lat_range = np.arange(lat_min, lat_max, binsize)
        lon_min = np.floor(min(data_u.nav_lon.min(), data_v.nav_lon.min()))
        lon_max = np.ceil(max(data_u.nav_lon.max(), data_v.nav_lon.max()))
        lon_range = np.arange(lon_min, lon_max, binsize)
        time = data_u.time_counter.values
        
        
        
        # TODO
        

        data = xr.open_dataset(self.file)
        
        # Change column names
        data = data.rename({'nav_lon': 'long',
                            'nav_lat': 'lat',
                            'uo': 'uC',
                            'vo': 'vC'})
        # Limit to just these coords and variables
        data = data[['lat','long','uC','vC']]
        
        # Limit to initial boundary
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),
                                   bounds.get_long_max()))
        data = data.sel(time=slice(bounds.get_time_max(), 
                                   bounds.get_time_max()))
        
        return data
