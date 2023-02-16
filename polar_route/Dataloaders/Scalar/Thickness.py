from .AbstractScalar import ScalarDataLoader
from polar_route.utils import date_range

from datetime import datetime, timedelta
import logging
import numpy as np
import xarray as xr

class ThicknessDataLoader(ScalarDataLoader):
    
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        self.data = self.import_data(bounds)
        # self.data_name = self.get_data_col_name()
        self.data_name = 'thickness'
                    
    def import_data(self, bounds):
        '''
        Placeholder until lookup-table dataloader class implemented
        '''
        thicknesses = {'Ross': {'w': 0.72, 'sp': 0.67, 'su': 1.32, 'a': 0.82, 'y': 1.07},
                    'Bellinghausen': {'w': 0.65, 'sp': 0.79, 'su': 2.14, 'a': 0.79, 'y': 0.90},
                    'Weddell E': {'w': 0.54, 'sp': 0.89, 'su': 0.87, 'a': 0.44, 'y': 0.73},
                    'Weddell W': {'w': 1.33, 'sp': 1.33, 'su': 1.20, 'a': 1.38, 'y': 1.33},
                    'Indian': {'w': 0.59, 'sp': 0.78, 'su': 1.05, 'a': 0.45, 'y': 0.68},
                    'West Pacific': {'w': 0.72, 'sp': 0.68, 'su': 1.17, 'a': 0.75, 'y': 0.79},
                    'None': {'w': 0.72, 'sp': 0.67, 'su': 1.32, 'a': 0.82, 'y': 1.07}}
        seasons = {1: 'su', 2: 'su', 3: 'a', 4: 'a', 5: 'a', 6: 'w', 7: 'w', 8: 'w', 9: 'sp', 10: 'sp', 11: 'sp',
                12: 'su'}

        def ice_thickness(d, long):
            """
                Returns ice thickness. Data taken from Table 3 in: doi:10.1029/2007JC004254
            """
            # The table has missing data points for Bellinghausen Autumn and Weddell W Winter, may require further thought
            month = d.month
            season = seasons[month]
            sea = None

            if -130 <= long < -60:
                sea = 'Bellinghausen'
            elif -60 <= long < -45:
                sea = 'Weddell W'
            elif -45 <= long < 20:
                sea = 'Weddell E'
            elif 20 <= long < 90:
                sea = 'Indian'
            elif 90 <= long < 160:
                sea = 'West Pacific'
            elif (160 <= long < 180) or (-180 <= long < -130):
                sea = 'Ross'
            else:
                sea = 'None'

            return thicknesses[sea][season]

        start_date = datetime.strptime(bounds.get_time_min(), "%Y-%m-%d").date()
        end_date = datetime.strptime(bounds.get_time_max(), "%Y-%m-%d").date()

        lats = [lat for lat in np.arange(bounds.get_lat_min(), bounds.get_lat_max(), 0.05)]
        lons = [lon for lon in np.arange(bounds.get_long_min(), bounds.get_long_max(), 0.05)]
        dates = [single_date for single_date in date_range(start_date, end_date)]

        thick_data = xr.DataArray(
            data=[[[ice_thickness(dt, lng)
                    for lng in lons]
                for _ in lats]
                for dt in dates],
            coords=dict(
                lat=lats,
                long=lons,
                time=[dt.strftime("%Y-%m-%d") for dt in dates],
            ),
            dims=("time", "lat", "long"),
            name="thickness",
        )

        thick_df = thick_data.\
            to_dataframe().\
            reset_index().\
            set_index(['lat', 'long', 'time']).reset_index()

        logging.debug("returning {} datapoints".format(len(thick_df.index)))
        return thick_df
