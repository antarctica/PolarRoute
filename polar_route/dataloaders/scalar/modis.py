from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader

import logging
import xarray as xr


class MODISDataLoader(ScalarDataLoader):        
    def import_data(self, bounds):
        '''
        Reads in data from a MODIS NetCDF file. 
        Renames variable to 'SIC'
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            xr.Dataset: 
                MODIS dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'SIC'
        '''
        logging.info(f"- Opening file {self.file}")
        # Open Dataset
        if len(self.files) == 1:    data = xr.open_dataset(self.files[0])
        else:                       data = xr.open_mfdataset(self.files)
        # Change column name
        data = data.rename({'iceArea': 'SIC'})

        # Set areas obscured by cloud to NaN values
        data = data.where(data.cloud != 1, drop=True)
        # Trim to initial datapoints
        data = self.trim_datapoints(bounds, data=data)
        
        return data
