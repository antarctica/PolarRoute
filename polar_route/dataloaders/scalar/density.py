from polar_route.dataloaders.scalar.abstractScalar import ScalarDataLoader
from polar_route.utils import date_range


from datetime import datetime
import logging


import numpy as np
import xarray as xr

class DensityDataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        '''
        Initialises density dataset. Initialises from values in a lookup table
        This will eventually be deprecated and replaced with a 
        'Lookup Table Dataloader'
        
       Args:
            bounds (Boundary): 
                Initial boundary to limit the dataset to
            params (dict):
                Dictionary of {key: value} pairs. Keys are attributes 
                this dataloader requires to function
        '''
        logging.info("Initalising Sea Ice Density dataloader")
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
        Creates a simulated dataset of sea ice density based on 
        scientific literature.
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            density_xr (xarray): 
                Sea Ice Density dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'density'
        '''
        # Look up table parameters hardcoded
        seasons = {
            1: 'su', 2: 'su', 12: 'su',
            3: 'a', 4: 'a', 5: 'a', 
            6: 'w', 7: 'w', 8: 'w', 
            9: 'sp', 10: 'sp', 11: 'sp',
            }
        densities = {
            'su': 875.0, 
            'sp': 900.0, 
            'a': 900.0, 
            'w': 920.0
            }

        def ice_density(d):
            '''
            Retrieve ice density from a datetime object
            '''
            month = d.month
            season = seasons[month]
            den = densities[season]
            return den

        # Set boundary coordinates and dates
        logging.info("- Setting boundaries for simulated dataset")
        start_date = datetime.strptime(bounds.get_time_min(), "%Y-%m-%d").date()
        end_date = datetime.strptime(bounds.get_time_max(), "%Y-%m-%d").date()

        lats = [lat for lat in np.arange(bounds.get_lat_min(), bounds.get_lat_max(), 0.05)]
        lons = [lon for lon in np.arange(bounds.get_long_min(), bounds.get_long_max(), 0.05)]
        dates = [single_date for single_date in date_range(start_date, end_date)]

        # Generate a density dataset from hardcoded values
        logging.info("- Generating dataset from boundaries")
        density_data = xr.DataArray(
            data=[[[ice_density(dt)
                    for _ in lons]
                for _ in lats]
                for dt in dates],
            coords=dict(
                lat=lats,
                long=lons,
                time=[dt.strftime("%Y-%m-%d") for dt in dates],
            ),
            dims=("time", "lat", "long"),
            name="density",
        )
        # Cast as dataframe
        density_df = density_data.\
            to_dataframe().\
            reset_index().\
            set_index(['lat', 'long', 'time'])
        
        density_xr = density_df.to_xarray()

        del density_df

        return density_xr