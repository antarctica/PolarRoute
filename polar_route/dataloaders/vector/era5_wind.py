from polar_route.dataloaders.vector.abstract_vector import VectorDataLoader

import logging

import xarray as xr

from datetime import datetime

class ERA5WindDataLoader(VectorDataLoader):
    def import_data(self, bounds):
        '''
        Reads in data from a ERA5 NetCDF file. 
        Renames coordinates to 'lat' and 'long'
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            xr.Dataset: 
                ERA5 wind dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'elevation'
        '''
        # Open Dataset
        if len(self.files) == 1:    data = xr.open_dataset(self.files[0])
        else:                       data = xr.open_mfdataset(self.files)
        # Change column names
        data = data.rename({'latitude': 'lat',
                            'longitude': 'long'})

        # Set min time to start of month to ensure we include data as only have
        # monthly cadence. Assuming time is in str format
        time_min = datetime.strptime(bounds.get_time_min(), '%Y-%m-%d')
        time_min = datetime.strftime(time_min, '%Y-%m-01')

        # Reverse order of lat as array goes from max to min
        data = data.reindex(lat=data.lat[::-1])

        # Trim to initial datapoints
        data = self.trim_datapoints(bounds, data=data)
        
        return data
