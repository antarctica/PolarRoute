from .AbstractScalar import ScalarDataLoader
from polar_route.utils import date_range


from datetime import datetime, timedelta



import numpy as np
import xarray as xr

class DensityDataLoader(ScalarDataLoader):
    
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        self.data = self.import_data(bounds)
        self.data_name = self.get_data_col_name()
        
    def import_data(self, bounds):
        '''
        Placeholder until lookup-table dataloader class implemented
        '''
        densities = {'summer': 875.0, 
                     'autumn': 900.0, 
                     'winter': 920.0,
                     'spring': 900.0}

        seasons = {
            12: 'summer', 1:  'summer', 2:  'summer', 
            3:  'autumn', 4:  'autumn', 5:  'autumn', 
            6:  'winter', 7:  'winter', 8:  'winter', 
            9:  'spring', 10: 'spring', 11: 'spring'}
        
        def get_density(d):
            month = d.month
            season = seasons[month]
            return densities[season]
        
        
        lats = [lat for lat in np.arange(bounds.get_lat_min(), 
                                         bounds.get_lat_max(), 0.05)]
        lons = [lon for lon in np.arange(bounds.get_long_min(), 
                                         bounds.get_long_max(), 0.05)]

        start_date = datetime.strptime(bounds.get_time_min(), "%Y-%m-%d")
        end_date = datetime.strptime(bounds.get_time_max(), "%Y-%m-%d")
        # delta = end_date - start_date
        #TODO Add 1 to range(delta.days), standard code missed this
        dates = [single_date for single_date in date_range(start_date, end_date)]
        
        density_data = xr.DataArray(
            data=[[[get_density(d) for _ in lons] for _ in lats] for d in dates],
            coords=dict(
                lat=lats,
                long=lons,
                time=[d.strftime('%Y-%m-%d') for d in dates]),
            dims=('time','lat','long'),
            name='density')
        
        return density_data.to_dataframe().reset_index().set_index(['lat', 'long', 'time']).reset_index()