import numpy as np

from polar_route.dataloaders.vector.abstract_vector import VectorDataLoader

import logging

import xarray as xr

from datetime import datetime

class ERA5WaveDirectionLoader(VectorDataLoader):
    def import_data(self, bounds):
        """
        Reads in wave direction data from a ERA5 NetCDF file.
        Renames coordinates to 'lat' and 'long' and calculates unit vector
        from mean wave direction variable 'mwd'.

        Args:
            bounds (Boundary): Initial boundary to limit the dataset to

        Returns:
            xr.Dataset:
                ERA5 wave dataset within limits of bounds.
                Dataset has coordinates 'lat', 'long', and variables 'uW', 'vW'
        """
        # Open Dataset
        if len(self.files) == 1:    data = xr.open_dataset(self.files[0])
        else:                       data = xr.open_mfdataset(self.files)
        # Change column names
        data = data.rename({'latitude': 'lat',
                            'longitude': 'long'})

        # Change domain of dataset from [0:360) to [-180:180)
        data = data.assign_coords(long=((data.long + 180) % 360) - 180)
        # Sort the 'long' axis so that sel() will work
        data = data.sortby('long')

        # Convert direction in degrees to u and v components
        data['mwd'] = np.radians(data['mwd'])
        data['uW'] = -1 * np.sin(data['mwd'])
        data['vW'] = -1 * np.cos(data['mwd'])

        # Limit to just uW and vW
        data = data[['uW', 'vW']]

        # Set min time to start of month to ensure we include data as we only have a
        # monthly cadence. Assuming time is in str format
        time_min = datetime.strptime(bounds.get_time_min(), '%Y-%m-%d')
        time_min = datetime.strftime(time_min, '%Y-%m-01')

        # Reverse order of lat as array goes from max to min
        data = data.reindex(lat=data.lat[::-1])

        # Trim to initial datapoints
        data = self.trim_datapoints(bounds, data=data)
        
        return data
