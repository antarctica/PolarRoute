from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader
from polar_route.utils import date_range

from datetime import datetime
import logging

import numpy as np
import xarray as xr

class DensityDataLoader(ScalarDataLoader):
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
        # Cast to xr.Dataset for faster processing
        density_xr = density_df.to_xarray()
        # Clear memory of now unused dataframe
        del density_df
        # No need to trim data, as was defined by bounds

        return density_xr