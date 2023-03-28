from polar_route.dataloaders.scalar.abstractScalar import ScalarDataLoader

import logging

from datetime import datetime, timedelta

import xarray as xr
from pandas import to_timedelta
from numpy import datetime64


class IceNetDataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        '''
        Initialises IceNet 2 dataset. Does no post-processing
        
       Args:
            bounds (Boundary): 
                Initial boundary to limit the dataset to
            params (dict):
                Dictionary of {key: value} pairs. Keys are attributes 
                this dataloader requires to function
        '''
        logging.info("Initalising IceNet dataloader")
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
        Reads in data from a IceNet 2 NetCDF file. 
        Renames coordinates to 'lat' and 'long', and renames variable to 
        'SIC'
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            pd.DataFrame: 
                IceNet dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'SIC'
        '''
        # Convert temporal boundary to datetime objects for comparison
        max_time = datetime.strptime(bounds.get_time_max(), '%Y-%m-%d')
        min_time = datetime.strptime(bounds.get_time_min(), '%Y-%m-%d')
        time_range = max_time - min_time
        
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
        
        logging.info(f"- Searching for closest date prior to {bounds.get_time_min()}")
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
        
        # TODO fix logging bug.
        #logging.info(f"- Found date {datetime.strftime('%Y-%m-%d')}")

        # Choose predictions from earliest date before start_date
        ds = ds.sel(leadtime=range(days_ago, time_range.days + days_ago))
        # Set to pd.DataFrame so can limit by lat/long
        df = ds.to_dataframe().reset_index()
        # Set time column to be dates of predictions
        # rather than date on which prediction made
        df.time = df.time + to_timedelta(df.leadtime, unit='d')
        # Remove unwanted columns
        df = df.drop(columns=['yc','xc','leadtime', 'Lambert_Azimuthal_Grid', 'sic_stddev', 'forecast_date'])
        
        logging.info('- Limiting to initial bounds')
        # Remove rows outside of spatial boundary
        mask = (df['lat']  >  bounds.get_lat_min())  & \
               (df['lat']  <= bounds.get_lat_max())  & \
               (df['long'] >  bounds.get_long_min()) & \
               (df['long'] <= bounds.get_long_max())
        # Turn SIC into a percentage
        df.SIC = df.SIC.apply(lambda x: x*100)
        
        # Return extracted data
        return df.loc[mask]
