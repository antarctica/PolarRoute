from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader

import logging
import pandas as pd
import h5py as h5

class MISRDataLoader(ScalarDataLoader):
        
    def import_data(self, bounds):
        '''
        Reads in data from a MISR h5 file.
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            pd.DataFrame: 
                MISR dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'SIR'         
        '''
        # Open Dataset
        logging.info(f"- Opening file {self.file}")
        fp = h5.File(self.file, 'r')
        
        year = self.file.split('April')[1].strip(' Roughness.h5')
        # Extract only required columns
        lats = fp['GeoLocation']['Latitude']
        lons = fp['GeoLocation']['Longitude']
        vals =  fp['Roughness']['Roughness_2D_svm']
        
        # Set columns
        data_dict = {
            'lat': lats[:].flatten(),
            'long': lons[:].flatten(),
            'SIR': vals[:].flatten()
        }
        data = pd.DataFrame.from_dict(data_dict).dropna()
        data['time'] = f'{year}-04-01'
        
        # Limit to initial boundary
        logging.info('- Limiting to initial bounds')
        self.data = data
        self.data = self.get_datapoints(bounds, return_coords=True)
        
        return self.data
