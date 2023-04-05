from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader
from polar_route.utils import date_range

from datetime import datetime
import logging
import numpy as np
import xarray as xr

class ThicknessDataLoader(ScalarDataLoader):
    def import_data(self, bounds):
        '''
        Creates a simulated dataset of sea ice thickness based on 
        scientific literature.
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            thickness_xr (xarray): 
                Sea Ice Density dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'thickness'
        '''
        # Look up table parameters hardcoded
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
                Returns ice thickness. Data taken from 
                Table 3 in: doi:10.1029/2007JC004254
                Values don't extend to the northern hemisphere, 
                but are used for the time being as a placeholder
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

        logging.info("- Setting boundaries for simulated dataset")
        start_date = datetime.strptime(bounds.get_time_min(), "%Y-%m-%d").date()
        end_date = datetime.strptime(bounds.get_time_max(), "%Y-%m-%d").date()

        lats = [lat for lat in np.arange(bounds.get_lat_min(), bounds.get_lat_max(), 0.05)]
        lons = [lon for lon in np.arange(bounds.get_long_min(), bounds.get_long_max(), 0.05)]
        dates = [single_date for single_date in date_range(start_date, end_date)]

        logging.info("- Generating dataset from boundaries")
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
            set_index(['lat', 'long', 'time'])
        
        thickness_xr = thick_df.to_xarray()

        logging.debug("returning {} datapoints".format(len(thick_df.index)))

        del thick_df
        # No need to trim data, as was defined by bounds

        return thickness_xr
