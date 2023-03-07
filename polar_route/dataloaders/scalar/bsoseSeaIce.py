from polar_route.dataloaders.scalar.abstractScalar import ScalarDataLoader

import logging

import xarray as xr

class BSOSESeaIceDataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        '''
        Initialises BSOSE Sea Ice  dataset. Does no post-processing
        
       Args:
            bounds (Boundary): 
                Initial boundary to limit the dataset to
            params (dict):
                Dictionary of {key: value} pairs. Keys are attributes 
                this dataloader requires to function
        '''
        logging.info("Initalising BSOSE Sea Ice dataloader")
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            logging.debug(f"self.{key}={val} (dtype={type(val)}) from params")
            setattr(self, key, val)
        
        # Import data
        self.data = self.import_data(bounds)
        
        # Get data name from column name if not set in params
        if self.data_name is None:
            logging.debug('- Setting self.data_name from column name')
            self.data_name = self.get_data_col_name()
        # or if set in params, set col name to data name
        else:
            logging.debug(f'- Setting data column name to {self.data_name}')
            self.data = self.set_data_col_name(self.data_name)
        
    def import_data(self, bounds):
        '''
        Reads in data from a BSOSE Sea Ice NetCDF file. 
        Renames coordinates to 'lat' and 'long', and renames variable to 
        'SIC'
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            xr.Dataset: 
                BSOSE Sea Ice dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'SIC'
                
        Raises:
            ValueError: 
                If units specified in config, 
                and value not 'fraction' or 'percentage
        '''
        logging.info(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        # Change column names
        data = data.rename({'SIarea': 'SIC',
                            'YC': 'lat',
                            'XC': 'long'})
        
        # Limit to initial boundary
        logging.info('- Limiting to initial bounds')
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        # bounds%360 because dataset is from [0:360), and bounds in [-180:180]
        data = data.sel(long=slice(bounds.get_long_min()%360,
                                   bounds.get_long_max()%360))
        data = data.sel(time=slice(bounds.get_time_min(), 
                                   bounds.get_time_max()))
        
        # Change domain of dataset from [0:360) to [-180:180)
        # NOTE: Must do this AFTER sel because otherwise KeyError
        data = data.assign_coords(long=((data.long + 180) % 360) - 180)
        if hasattr(self, 'units'):
            logging.info(f'- Changing units of data to {self.units}')
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
