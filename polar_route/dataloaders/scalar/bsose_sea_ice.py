from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader

import logging

import xarray as xr

class BSOSESeaIceDataLoader(ScalarDataLoader):
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
        if len(self.files) == 1:    data = xr.open_dataset(self.files[0])
        else:                       data = xr.open_mfdataset(self.files)
        # Change column names
        data = data.rename({'SIarea': 'SIC',
                            'YC': 'lat',
                            'XC': 'long'})
        
        # Change domain of dataset from [0:360) to [-180:180)
        data = data.assign_coords(long=((data.long + 180) % 360) - 180)
        # Sort the 'long' axis so that sel() will work
        data = data.sortby('long')
        # Trim to initial datapoints
        data = self.trim_datapoints(bounds, data=data)

        if hasattr(self, 'units'):
            logging.info(f'- Changing units of data to {self.units}')
            # Convert to percentage form if requested in params
            if self.units == 'percentage':
                data = data.assign(SIC= data['SIC'] * 100)
            elif self.units == 'fraction':
                pass # BSOSE data already in fraction form
            else:
                raise ValueError("Parameter 'units' not understood."\
                                 "Expected 'percentage' or 'fraction',"\
                                f"but recieved {self.units}")
        return data
