from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader

import logging
import xarray as xr

class GEBCODataLoader(ScalarDataLoader):
    def import_data(self, bounds):
        '''
        Reads in data from GEBCO NetCDF files. Renames coordinates to
        'lat' and 'long'.
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            xr.Dataset: 
                GEBCO dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'elevation'         
        '''
        # Import data from files defined in config
        if len(self.files) == 1:    data = xr.open_dataset(self.files[0])
        else:                       data = xr.open_mfdataset(self.files)
        # Rename columns to standard format
        data = data.rename({'lon':'long'})
        # Trim to initial datapoints
        data = self.trim_datapoints(bounds, data=data)
        
        return data
