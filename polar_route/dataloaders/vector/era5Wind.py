from polar_route.dataloaders.vector.abstractVector import VectorDataLoader

import logging

import xarray as xr

from datetime import datetime

class ERA5WindDataLoader(VectorDataLoader):
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
        logging.info("Initalising ERA5 Wind dataloader")
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
        Reads in data from a ERA5 NetCDF file. 
        Renames coordinates to 'lat' and 'long'
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            xr.Dataset: 
                ERA5 wind dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'elevation'
        '''
        logging.info(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        # Change column names
        data = data.rename({'latitude': 'lat',
                            'longitude': 'long'})

        # TODO Remove this temp fix
        data = data.assign(time= data['time'] + pd.Timedelta(days=365*2))

        # Set min time to start of month to ensure we include data as only have
        # monthly cadence. Assuming time is in str format
        time_min = datetime.strptime(bounds.get_time_min(), '%Y-%m-%d')
        time_min = datetime.strftime(time_min, '%Y-%m-01')

        # Reverse order of lat as array goes from max to min
        data = data.reindex(lat=data.lat[::-1])

        # Limit to initial boundary
        logging.info('- Limiting to initial bounds')
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),
                                   bounds.get_long_max()))
        data = data.sel(time=slice(time_min, 
                                   bounds.get_time_max()))
        
        return data
