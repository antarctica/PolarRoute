from polar_route.Dataloaders.LookupTable.AbstractLUT import LookupTableDataLoader

from polar_route.Boundary import Boundary

import json


class ThicknessDataLoader(LookupTableDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
            
        self.data = self.import_data(bounds)
        
    def import_data(self, bounds):
        
        # Import data from JSON
        with open(f'{self.file}', 'r') as fp:
            lut_json = json.load(fp)
        
        data = {}
        # For each entry in JSON
        for key in lut_json:
            #Extract boundary
            lut_bounds = Boundary([lut_json[key]['lat_min'], lut_json[key]['lat_max']],
                              [lut_json[key]['long_min'], lut_json[key]['long_max']],
                              [lut_json[key]['time_min'], lut_json[key]['time_max']],)
            # If boundary in global bounds
            if self.bounds_in_boundary(lut_bounds, bounds):
                # Add new entry in standardised format
                data[key] = {
                    'boundary': lut_bounds,
                    'value': lut_json[key]['value']
                }
        # Return standardised data
        return data
    

if __name__ == '__main__':
    bounds = Boundary(
        [20, 40],
        [-70,-50],
        ['2020-02-01', '2020-03-31']
    )
    params = {
        'data_name': 'thickness',
        'file': '/home/habbot/Documents/Work/PolarRoute/datastore/thickness/thickness_lut.json'
    }
    tdl = ThicknessDataLoader(bounds, params)
    value = tdl.get_value(bounds)
    
    print(value)