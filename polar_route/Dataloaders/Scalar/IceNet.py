from polar_route.Dataloaders.Scalar.AbstractScalar import ScalarDataLoader

import logging

from datetime import datetime, timedelta

import xarray as xr
from pandas import to_timedelta
from numpy import datetime64


class IceNetDataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        self.data = self.import_data(bounds)
        self.data = self.set_data_col_name('SIC')
        self.data_name = self.get_data_col_name() # = 'SIC'
        
        
    def import_data(self, bounds):
        
        max_time = datetime.strptime(bounds.get_time_max(), '%Y-%m-%d')
        min_time = datetime.strptime(bounds.get_time_min(), '%Y-%m-%d')
        time_range = max_time - min_time
        
        assert time_range < timedelta(days=6), f'Time boundary too large! Forcast only runs for max of 6 days'
        
        logging.debug("Importing IceNet data...")
        logging.debug(f"- Opening file {self.file}")
        # Open Dataset
        ds = xr.open_dataset(self.file)
        ds = ds.rename({'lon':'long',
                        'sic_mean': 'SIC'})        
        
        # Start time set as previous day
        start_time  = datetime64(min_time - timedelta(days=1))
        
        # Extract forecast from boundary start date - 1 day (so first day in dataset is first forecast day)
        ds = ds.sel(time=start_time).sel(leadtime=range(1, time_range.days + 1))
        # TODO Handle days that don't exist
        
        
        # Limit to coordinate range
        df = ds.to_dataframe().reset_index()
        # Turn SIC into a percentage
        df.SIC = df.SIC.apply(lambda x: x*100)
        
        df.time = df.time + to_timedelta(df.leadtime, unit='d')
        # Remove unwanted columns
        df = df.drop(columns=['yc','xc','leadtime', 'Lambert_Azimuthal_Grid'])
        
        mask = (df['lat']  >  bounds.get_lat_min())  & \
               (df['lat']  <= bounds.get_lat_max())  & \
               (df['long'] >  bounds.get_long_min()) & \
               (df['long'] <= bounds.get_long_max())
        
        return df.loc[mask]
