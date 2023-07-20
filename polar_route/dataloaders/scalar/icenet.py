from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader

import logging

from datetime import datetime, timedelta

import xarray as xr
from pandas import to_timedelta
from numpy import datetime64


class IceNetDataLoader(ScalarDataLoader):
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
        # Retrieve list of dates from filenames
        # assumes file names are in the format 
        # <hemisphere>_daily_forecast.<YYYY-MM-DD>.nc
        file_dates = {datetime.strptime(file.split('.')[1], '%Y-%m-%d'): file 
                      for file in self.files}

        # Find closest date prior to min_time
        closest_date = min(file_dates, 
                           key=lambda x: (x>min_time, abs(x-min_time)))
        
        # Open Dataset
        ds = xr.open_dataset(file_dates[closest_date])
        # Cast coordinates/variables to those understood by mesh
        ds = ds.rename({'lon':'long',
                        'sic_mean': 'SIC'})
        
        # Max number of days in future IceNet can predict
        max_leadtime = int(ds.leadtime.max())
        
        # Ensure that temporal boundary is possible before extracting
        assert time_range < timedelta(days=max_leadtime),\
            f'Time boundary too large! Forecast only runs for max of {max_leadtime} days'
        
        assert closest_date + timedelta(days=max_leadtime) > max_time,\
            'Time boundary runs beyond max forecast date!'
        
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
                logging.debug(f'\tUnable to select start day of {start_time} for IceNet, trying previous day')
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
        # Trim to initial datapoints
        df = self.trim_datapoints(bounds, data=df)
        
        # Turn SIC into a percentage
        df.SIC = df.SIC.apply(lambda x: x*100)
        
        # Return extracted data
        return df
