from polar_route.Dataloaders.Scalar.AbstractScalar import ScalarDataLoader

import logging

import xarray as xr


class GEBCODataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        self.data = self.import_data(bounds)
        self.data = self.downsample()
        
        self.data_name = self.get_data_col_name() # = 'elevation'
        
    def import_data(self, bounds):
        logging.debug("Importing GEBCO data...")
        logging.debug(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        data = data.rename({'lon':'long'})
        
        # Limit to initial boundary
        data = data.sel(lat=slice(bounds.get_lat_min(),bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),bounds.get_long_max()))
        return data
