from .AbstractScalar import ScalarDataLoader

import logging

import xarray as xr

class BSOSESeaIceDataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Set up data
        self.data = self.import_data(bounds)
        self.data_name = self.get_data_col_name() # = 'SIC'
        
    def import_data(self, bounds):
        logging.debug("Importing BSOSE Sea Ice data...")
        logging.debug(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        # Change column names
        data = data.rename({'SIarea': 'SIC',
                            'YC': 'lat',
                            'XC': 'long'})
        
        # Limit to initial boundary
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        # bound%360 because dataset is from [0:360), and bounds in [-180:180]
        data = data.sel(long=slice(bounds.get_long_min()%360,
                                   bounds.get_long_max()%360))
        data = data.sel(time=slice(bounds.get_time_min(), 
                                   bounds.get_time_max()))
        
        # Change domain of dataset from [0:360) to [-180:180)
        # NOTE: Must do this AFTER sel because otherwise KeyError
        data = data.assign_coords(long=((data.long + 180) % 360) - 180)
        if hasattr(self, 'units'):
            # Convert to percentage form if requested in params
            if self.units == 'percentage':
                data = data.assign(SIC= data['SIC'] * 100)
            elif self.units == 'fraction':
                pass # BSOSE data already in fraction form
            else:
                raise ValueError(f"Parameter 'units' not understood."\
                                  "Expected 'percentage' or 'fraction',"\
                                  "but recieved {self.units}")
        return data
