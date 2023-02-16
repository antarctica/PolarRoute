from .AbstractScalar import ScalarDataLoader
from polar_route.utils import date_range


from datetime import datetime, timedelta
import logging


import numpy as np
import xarray as xr

class DensityDataLoader(ScalarDataLoader):
    
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        self.data = self.import_data(bounds)
        # self.data_name = self.get_data_col_name()
        self.data_name = 'density'
        
    def import_data(self, bounds):
        '''
        Placeholder until lookup-table dataloader class implemented
        '''
        seasons = {1: 'su', 2: 'su', 3: 'a', 4: 'a', 5: 'a', 6: 'w', 7: 'w', 8: 'w', 9: 'sp', 10: 'sp', 11: 'sp',
                12: 'su'}
        densities = {'su': 875.0, 'sp': 900.0, 'a': 900.0, 'w': 920.0}

        def ice_density(d):
            month = d.month
            season = seasons[month]
            den = densities[season]
            return den

        start_date = datetime.strptime(bounds.get_time_min(), "%Y-%m-%d").date()
        end_date = datetime.strptime(bounds.get_time_max(), "%Y-%m-%d").date()

        lats = [lat for lat in np.arange(bounds.get_lat_min(), bounds.get_lat_max(), 0.05)]
        lons = [lon for lon in np.arange(bounds.get_long_min(), bounds.get_long_max(), 0.05)]
        dates = [single_date for single_date in date_range(start_date, end_date)]

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

        density_df = density_data.\
            to_dataframe().\
            reset_index().\
            set_index(['lat', 'long', 'time']).reset_index()

        logging.debug("returned {} datapoints".format(len(density_df.index)))
        return density_df