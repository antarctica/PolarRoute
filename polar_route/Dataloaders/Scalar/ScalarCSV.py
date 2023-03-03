from polar_route.Dataloaders.Scalar.AbstractScalar import ScalarDataLoader

import logging
import pandas as pd

class ScalarCSVDataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        '''
        Initialises dataset from CSV file. Does no post-processing
        
       Args:
            bounds (Boundary): 
                Initial boundary to limit the dataset to
            params (dict):
                Dictionary of {key: value} pairs. Keys are attributes 
                this dataloader requires to function
        '''
        logging.info("Initalising Scalar CSV dataloader")
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
        Reads in data from a CSV file. 
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            data (pd.DataFrame): 
                Scalar dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', potentially 'time',
                and variable defined by column heading in csv file
        '''

        logging.info(f"- Opening file {self.file}")
        # Load in a csv file
        data = pd.read_csv(self.file)

        logging.info('- Limiting to initial bounds')
        # Limit to within boundaries
        data = data[data['long'].between(bounds.get_long_min(), 
                                         bounds.get_long_max())]
        data = data[data['lat'].between(bounds.get_lat_min(), 
                                        bounds.get_lat_max())]
        if 'time' in data.columns:
            data = data[data['time'].between(bounds.get_time_min(), 
                                            bounds.get_time_max())]
        
        return data
