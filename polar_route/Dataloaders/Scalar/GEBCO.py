from polar_route.Dataloaders.Scalar.AbstractScalar import ScalarDataLoader

import logging
import xarray as xr

class GEBCODataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        '''
        Initialises GEBCO dataset, 
        downsamples (if downsample_factors specified in params)
        and sets variable name (if data_name specified in params)
        
        Args:
            bounds (Boundary): 
                Initial boundary to limit the dataset to
            params (dict):
                Dictionary of {key: value} pairs. Keys are attributes 
                this dataloader requires to function
        '''
        logging.info("Initalising GEBCO dataloader")
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            logging.debug(f"self.{key}={val} (dtype={type(val)}) from params")
            setattr(self, key, val)
        
        # Import data
        self.data = self.import_data(bounds)
        # Downsample if specified in params
        self.data = self.downsample()
        
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
        Reads in data from a GEBCO NetCDF file. Renames coordinates to
        'lat' and 'long'.
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            xr.Dataset: 
                GEBCO dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'elevation'         
        '''
        # Open Dataset
        logging.info(f"- Opening file {self.file}")
        data = xr.open_dataset(self.file)
        data = data.rename({'lon':'long'})
        
        # Limit to initial boundary
        logging.info('- Limiting to initial bounds')
        data = data.sel(lat=slice(bounds.get_lat_min(),bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),bounds.get_long_max()))
        return data
