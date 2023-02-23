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
        
        # Convert temporal boundary to datetime objects for comparison
        max_time = datetime.strptime(bounds.get_time_max(), '%Y-%m-%d')
        min_time = datetime.strptime(bounds.get_time_min(), '%Y-%m-%d')
        time_range = max_time - min_time
        
        logging.info("Importing IceNet data...")
        logging.info(f"- Opening file {self.file}")
        
        # Open Dataset
        ds = xr.open_dataset(self.file)
        # Cast coordinates/variables to those understood by mesh
        ds = ds.rename({'lon':'long',
                        'sic_mean': 'SIC'})
        
        # Max number of days in future IceNet can predict
        max_leadtime = int(ds.leadtime.max())
        
        # Ensure that temporal boundary is possible before extracting
        assert time_range < timedelta(days=max_leadtime),\
            f'Time boundary too large! Forecast only runs for max of {max_leadtime} days'
        
        # For the days in forecast range of IceNet dataset
        for days_ago in range(1, max_leadtime+1):
            # Set the date from which the forecast is taken
            start_time  = datetime64(min_time - timedelta(days=days_ago))
            try:
                # See if day exists, raises error if date not in dataset
                ds = ds.sel(time=start_time)
                break
            except:
                # Error thrown, date not in dataset. Try previous day
                logging.debug(f' - Unable to select start day of {start_time} for IceNet, trying previous day')
                continue
        else:
            # If ran through entire dataset with no valid dates
            raise EOFError('No valid start date found in IceNet data!')
        
        assert (time_range.days < max_leadtime - days_ago),\
            f'''Not enough leadtime to support date range specified!
            End ({max_time}) - Start({min_time}) = {time_range.days} days
            Leadtime ({max_leadtime}) days - Prediction({days_ago}) days ago = {max_leadtime-days_ago} days
            '''
        # Choose predictions from earliest date before start_date
        ds = ds.sel(leadtime=range(days_ago, time_range.days + days_ago))
                
        # Set to pd.DataFrame so can limit by lat/long
        df = ds.to_dataframe().reset_index()
        # Set time column to be dates of predictions
        # rather than date on which prediction made
        df.time = df.time + to_timedelta(df.leadtime, unit='d')
        # Remove unwanted columns
        df = df.drop(columns=['yc','xc','leadtime', 'Lambert_Azimuthal_Grid'])
        # Remove rows outside of spatial boundary
        mask = (df['lat']  >  bounds.get_lat_min())  & \
               (df['lat']  <= bounds.get_lat_max())  & \
               (df['long'] >  bounds.get_long_min()) & \
               (df['long'] <= bounds.get_long_max())
        # Turn SIC into a percentage
        df.SIC = df.SIC.apply(lambda x: x*100)
        
        # Return extracted data
        return df.loc[mask]
