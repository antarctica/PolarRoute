from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader

import dask as dd

class ScalarCSVDataLoader(ScalarDataLoader):
    def import_data(self, bounds):
        '''
        Reads in data from a CSV file. 
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            pd.DataFrame: 
                Scalar dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', potentially 'time',
                and variable defined by column heading in csv file
        '''
        # Read in data
        data = dd.read_csv(self.files)
        # Trim to initial datapoints
        data = self.trim_datapoints(bounds, data=data)
        
        return data
