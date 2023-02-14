from .AbstractVector import VectorDataLoader

import logging

import pandas as pd

class VectorCSVDataLoader(VectorDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Set up data
        self.data = self.import_data(bounds)
        self.data_name = self.get_data_col_name() # 'dummy_data'
        
    def import_data(self, bounds):
        logging.debug("Importing dummy data...")
        logging.debug(f"- Opening file {self.file}")
        # Load in a csv file
        data = pd.read_csv(self.file)

        # Limit to within boundaries
        data = data[data['long'].between(bounds.get_long_min(), 
                                         bounds.get_long_max())]
        data = data[data['lat'].between(bounds.get_lat_min(), 
                                        bounds.get_lat_max())]
        data = data[data['time'].between(bounds.get_time_min(), 
                                         bounds.get_time_max())]
        
        return data
