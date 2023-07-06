from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader

import logging
import xarray as xr

class VisualIceDataLoader(ScalarDataLoader):
    def import_data(self, bounds):
        '''
        Reads in data from Visual_Ice NetCDF files. Renames coordinates to
        'lat' and 'long'.
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            xr.Dataset: 
                visual_ice dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'SIC'         
        '''
        # Import data from files defined in config
        if len(self.files) == 1:    visual_ice = xr.open_dataset(self.files[0])
        else:                       visual_ice = xr.open_mfdataset(self.files)
        # Rename columns to standard format
        # visual_ice = visual_ice.rename({'x':'lat', 'y':'long'})
        visual_ice = visual_ice.drop_vars('polar_stereographic')
        visual_ice = visual_ice.rename({'Band1':'SIC'})
       
        
        return visual_ice