from .AbstractScalar import ScalarDataLoader
from polar_route.utils import date_range

from datetime import datetime, timedelta

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
        thicknesses = {
            'Ross':          {'winter': 0.72, 'spring': 0.67, 'summer': 1.32, 
                              'autumn': 0.82, 'year': 1.07},
            'Bellinghausen': {'winter': 0.65, 'spring': 0.79, 'summer': 2.14, 
                              'autumn': 0.79, 'year': 0.90},
            'Weddell E':     {'winter': 0.54, 'spring': 0.89, 'summer': 0.87, 
                              'autumn': 0.44, 'year': 0.73},
            'Weddell W':     {'winter': 1.33, 'spring': 1.33, 'summer': 1.20, 
                              'autumn': 1.38, 'year': 1.33},
            'Indian':        {'winter': 0.59, 'spring': 0.78, 'summer': 1.05, 
                              'autumn': 0.45, 'year': 0.68},
            'West Pacific':  {'winter': 0.72, 'spring': 0.68, 'summer': 1.17, 
                              'autumn': 0.75, 'year': 0.79},
            'None':          {'winter': 0.72, 'spring': 0.67, 'summer': 1.32, 
                              'autumn': 0.82, 'year': 1.07}}
        seasons = {
            12: 'summer', 1:  'summer', 2:  'summer', 
            3:  'autumn', 4:  'autumn', 5:  'autumn', 
            6:  'winter', 7:  'winter', 8:  'winter', 
            9:  'spring', 10: 'spring', 11: 'spring'}
        
        def get_thickness(d, lon):
            month = d.month
            season = seasons[month]
            
            if   -130 <= lon <  -60: sea = 'Bellinghausen'
            elif  -60 <= lon <  -45: sea = 'Weddell W'
            elif  -45 <= lon <   20: sea = 'Weddell E'
            elif   20 <= lon <   90: sea = 'Indian'
            elif   90 <= lon <  160: sea = 'West Pacific'
            elif  160 <= lon <  180: sea = 'Ross'
            elif -180 <= lon < -130: sea = 'Ross'
            else: sea = 'None'
            
            return thicknesses[sea][season]
        
        
        lats = [lat for lat in np.arange(bounds.get_lat_min(), 
                                         bounds.get_lat_max(), 0.05)]
        lons = [lon for lon in np.arange(bounds.get_long_min(), 
                                         bounds.get_long_max(), 0.05)]
        
        start_date = datetime.strptime(bounds.get_time_min(), "%Y-%m-%d")
        end_date = datetime.strptime(bounds.get_time_max(), "%Y-%m-%d")
        # delta = end_date - start_date
        #TODO Add 1 to range(delta.days), standard code missed this
        # dates = [start_date + timedelta(days=i) for i in range(delta.days)]
        dates = [single_date for single_date in date_range(start_date, end_date)]
        
        thickness_data = xr.DataArray(
            data=[[[get_thickness(d, lon) for lon in lons] for _ in lats] for d in dates],
            coords=dict(
                lat=lats,
                long=lons,
                time=[d.strftime('%Y-%m-%d') for d in dates]),
            dims=('time','lat','long'),
            name='thickness')

        return thickness_data.to_dataframe().reset_index().set_index(['lat', 'long', 'time']).reset_index()
