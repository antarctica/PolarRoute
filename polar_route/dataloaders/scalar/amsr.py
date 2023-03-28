from polar_route.dataloaders.scalar.abstractScalar import ScalarDataLoader
from datetime import datetime

import logging
import glob
import xarray as xr
class AMSRDataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        '''
        Initialises AMSR dataset. Reprojects data.
        
       Args:
            bounds (Boundary): 
                Initial boundary to limit the dataset to
            params (dict):
                Dictionary of {key: value} pairs. Keys are attributes 
                this dataloader requires to function
        '''
        logging.info("Initalising AMSR Sea Ice dataloader")
        
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Read in and manipulate data to standard form
        self.data = self.import_data(bounds)
        self.data = self.downsample()
        
        # Cast to df to reproject
        self.data = self.data.to_dataframe().reset_index().dropna()
        
        # Set to lower case for case insensitivity
        self.hemisphere = self.hemisphere.lower()
        # Reproject to mercator
        if self.hemisphere == 'north': 
            self.data = self.reproject('EPSG:3411', 'EPSG:4326', x_col='x', y_col='y')
        elif self.hemisphere == 'south':
            self.data = self.reproject('EPSG:3412', 'EPSG:4326', x_col='x', y_col='y')
        else:
            raise ValueError('No hemisphere defined in params!')
        
        # Limit dataset to just values within bounds
        self.data = self.data.loc[self.get_datapoints(bounds).index]
        
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

        # If single NetCDF File specified
        if hasattr(self, 'file'):
            # Ensure .nc file passed in params
            assert(self.file[-3:] == '.nc')
            logging.debug(f"- Opening file {self.file}")
            # Extract data, append date to coords
            date = retrieve_date(self.file)
            data = retrieve_data(self.file, date)
        # If folder specified
        elif hasattr(self, 'folder'):
            # Open folder and read in files
            logging.debug(f"- Searching folder {self.folder}")
            data_array = []
            # For each .nc file in folder
            for file in sorted(glob.glob(f'{self.folder}*.nc')):
                logging.debug(f"- Opening file {file}")
                # If date within boundary
                date = retrieve_date(file)
                # If file data from within time boundary, append to list
                # Doing now to avoid ingesting too much data initially
                if datetime.strptime(bounds.get_time_min(), '%Y-%m-%d') <= \
                   datetime.strptime(date, '%Y-%m-%d') <= \
                   datetime.strptime(bounds.get_time_max(), '%Y-%m-%d'):
                    data_array.append(retrieve_data(file, date))
            # Concat all valid files
            data = xr.concat(data_array,'time')
        # Otherwise need a file or folder to read from
        else:
            raise ValueError('File or folder not specified in params!')

        # Remove unnecessary column
        data = data.drop_vars('polar_stereographic')
        data = data.rename({'z': 'SIC'})
        return data
