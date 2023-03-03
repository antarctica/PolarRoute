from polar_route.Dataloaders.Vector.AbstractVector import VectorDataLoader

import logging

import xarray as xr
import numpy as np

#TODO Read in 2 files, combine to one object
class ORAS5CurrentDataLoader(VectorDataLoader):
    def __init__(self, bounds, params):
        '''
        Initialises ORAS5 currents dataset. Does no post-processing.
        NOTE - This dataloader uses a NetCDF file that has been pre-processed
        to include both zonal and meridional velocities! Additional work 
        needed to make it work with data straight from the website
        
        Args:
            bounds (Boundary): 
                Initial boundary to limit the dataset to
            params (dict):
                Dictionary of {key: value} pairs. Keys are attributes 
                this dataloader requires to function
        '''
        
        logging.info("Initalising ORAS5 currents dataloader")
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
        Reads in data from a ORAS5 Depth NetCDF files. 
        Renames coordinates to 'lat' and 'long', and renames variable to 
        'uC, vC'
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            xr.Dataset: 
                ORAS5 currents dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'uC', 'vC'
        '''
        logging.info(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        
        # Change column names
        data = data.rename({'nav_lon': 'long',
                            'nav_lat': 'lat',
                            'uo': 'uC',
                            'vo': 'vC'})
        # Limit to just these coords and variables
        data = data[['lat','long','uC','vC']]
        
        # Limit to initial boundary
        logging.info('- Limiting to initial bounds')
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),
                                   bounds.get_long_max()))
        data = data.sel(time=slice(bounds.get_time_max(), 
                                   bounds.get_time_max()))
        
        return data