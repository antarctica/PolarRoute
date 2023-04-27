from polar_route.dataloaders.vector.abstract_vector import VectorDataLoader

import logging
 
import xarray as xr

class BalticCurrentDataLoader(VectorDataLoader):
    def import_data(self, bounds):
        '''
        Reads in data from a BSOSE Depth NetCDF file. 
        Renames coordinates to 'lat' and 'long', and renames variable to 
        'uC, vC'
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            xr.Dataset: 
                Baltic currents dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'uC', 'vC'
        '''
        # Open Dataset
        if len(self.files) == 1:    data = xr.open_dataset(self.files[0])
        else:                       data = xr.open_mfdataset(self.files)
        # Change column names
        data = data.rename({'latitude': 'lat',
                            'longitude': 'long',
                            'uo': 'uC',
                            'vo': 'vC'})
        
        # Trim to initial datapoints
        data = self.trim_datapoints(bounds, data=data)
        
        return data
