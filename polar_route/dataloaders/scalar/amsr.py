from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader
from datetime import datetime
import logging

import xarray as xr
class AMSRDataLoader(ScalarDataLoader):
    
    def add_default_params(self, params):
        '''
        Translates 'hemisphere' parameter into values of in_proj and out_proj 
        that pyProj can understand. Also defines x_col and y_col for AMSR data
        for reprojection to function
        
        Args:
            params (dict): 
                Dictionary holding keys and values that will be turned into 
                object attributes
        
        Returns:
            dict:
                Params dictionary with addition of translated key/value pairs
        '''
        # Set default parameters same as all other scalar dataloaders
        params = super().add_default_params(params)
        # Translate 'hemisphere' into initial projection
        hemisphere = params['hemisphere'].lower()
        if  hemisphere == 'north':
            params['in_proj'] = 'EPSG:3411'
        elif hemisphere == 'south':
            params['in_proj'] = 'EPSG:3412'
        else:
            raise ValueError(
                "Hemisphere defined in config is not 'north' or 'south'"
                )
        # Set initial projection column names
        params['x_col'] = 'x'
        params['y_col'] = 'y'
        
        return params
            
    def import_data(self, bounds):
        '''
        Reads in data from a AMSR NetCDF file, or folder of files. 
        Drops unnecessary column 'polar_stereographic', and renames variable
        'z' to 'SIC'
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            xr.Dataset: 
                AMSR SIC dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'SIC'
        '''
        def retrieve_date(filename):
            '''
            Get date from filename in format:
                asi-AMSR2-s6250-<year><month><day>-v.5.4.nc
            '''
            date = filename.split('-')[-2]
            date = f'{date[:4]}-{date[4:6]}-{date[6:]}'
            return date
        
        def retrieve_data(filename, date):
            '''
            Read in data as xr.Dataset, create time coordinate
            '''
            data = xr.open_dataset(filename)
            # Add date to data
            data = data.assign_coords(time=date)
            return data

        data_array = []
        relevant_files = []
        # For each file found from config
        for file in self.files:
            # If date within boundary
            date = retrieve_date(file)
            # If file data from within time boundary, append to list
            # Doing now to avoid ingesting too much data initially
            if datetime.strptime(bounds.get_time_min(), '%Y-%m-%d') <= \
                datetime.strptime(date, '%Y-%m-%d') <= \
                datetime.strptime(bounds.get_time_max(), '%Y-%m-%d'):
                data_array.append(retrieve_data(file, date))
                relevant_files += [file]
        # Concat all valid files
        if len(data_array) == 0:
            logging.error('\tNo files found for date range '+\
                         f'[ {bounds.get_time_min()} : {bounds.get_time_max()} ]')
            raise FileNotFoundError('No AMSR files found within specified time range!')
        data = xr.concat(data_array,'time')

        # Remove unnecessary column, rename data column
        data = data.drop_vars('polar_stereographic')
        data = data.rename({'z': 'SIC'})
        
        # Limit self.files to only those actually used
        self.files = relevant_files
        
        # TODO Limit data range before reprojection
        
        return data
