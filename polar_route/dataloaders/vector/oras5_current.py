from polar_route.dataloaders.vector.abstract_vector import VectorDataLoader

import logging

import xarray as xr
import numpy as np

#TODO Read in 2 files, combine to one object
class ORAS5CurrentDataLoader(VectorDataLoader):
    def import_data(self, bounds):
        '''
        Reads in data from a ORAS5 Depth NetCDF files. 
        Renames coordinates to 'lat' and 'long', and renames variable to 
        'uC, vC'
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            xr.Dataset: 
                ORAS5 currents dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'uC', 'vC'
        '''
        # Open Dataset
        if len(self.files) == 1:    data = xr.open_dataset(self.files[0])
        else:                       data = xr.open_mfdataset(self.files)
        
        # Change column names
        data = data.rename({'nav_lon': 'long',
                            'nav_lat': 'lat',
                            'uo': 'uC',
                            'vo': 'vC'})
        # Limit to just these coords and variables
        data = data[['lat','long','uC','vC']]
        
        # Limit to initial boundary
        data = self.trim_datapoints(bounds, data=data)
        
        return data