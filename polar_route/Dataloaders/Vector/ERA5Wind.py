from .AbstractVector import VectorDataLoader

import logging

import xarray as xr

from datetime import datetime

class ERA5WindDataLoader(VectorDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Set up data
        self.data = self.import_data(bounds)
        self.data_name = self.col_names_to_str() # = 'SIC'
        
    def import_data(self, bounds):
        logging.debug("Importing ERA5 Wind data...")
        logging.debug(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        # Change column names
        data = data.rename({'latitude': 'lat',
                            'longitude': 'long'})

        # TODO Ask if we need this line or not? Seems really weird
        data = data.assign(time= data['time'] + pd.Timedelta(days=365*2))

        # Set min time to start of month to ensure we include data as only have
        # monthly cadence. Assuming time is in str format
        time_min = datetime.strptime(bounds.get_time_min(), '%Y-%m-%d')
        time_min = datetime.strftime(time_min, '%Y-%m-01')

        # Reverse order of lat as array goes from max to min
        data = data.reindex(lat=data.lat[::-1])

        # Limit to initial boundary
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),
                                   bounds.get_long_max()))
        data = data.sel(time=slice(time_min, 
                                   bounds.get_time_max()))
        
        return data
