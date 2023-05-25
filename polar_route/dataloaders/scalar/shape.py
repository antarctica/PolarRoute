from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader

import logging
import pandas as pd
import numpy as np

class ShapeDataLoader(ScalarDataLoader):
    
    def add_default_params(self, params):
        '''
        Set default values for abstract shape dataloaders, starting by
        including defaults for scalar dataloaders.
        
        Args:
            params (dict): 
                Dictionary containing attributes that are required for the
                shape being loaded. Must include 'shape'.
            
        Returns:
            (dict): 
                Dictionary of attributes the dataloader will require, 
                completed with default values if not provided in config.
        '''
        # Add default scalar params
        params = super().add_default_params(params)
        
        # Number of datapoints to populate per axis
        if 'nx' not in params:
            params['nx'] = 101
        if 'ny' not in params:
            params['ny'] = 101

        # Define default circle parameters
        if params['dataloader_name'] == 'circle':
            if 'radius' not in params:
                params['radius'] = 1
            if 'centre' not in params:
                params['centre'] = (None, None)
        # Define default square parameters
        elif params['dataloader_name'] == 'square':
            if 'side_length' not in params:
                params['side_length'] = 1
            if 'centre' not in params:
                params['centre'] = (None, None)
        # Define default gradient params
        elif params['dataloader_name'] == 'gradient':
            if 'vertical' not in params:
                params['vertical'] = True
        # Define default checkerboard params
        elif params['dataloader_name'] == 'checkerboard':
            if 'gridsize' not in params:
                params['gridsize'] = (1,1)
            
        return params
        
    
    def import_data(self, bounds):
        '''
        Generates data in the form of an abstract shape, such as circle,
        checkerboard, or gradient. This method acts like a factory in that
        it simply selects the correct shape method to enact
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            data_xr (xarray): 
                xarray with coordinates within bounds, and values between
                [0:1]. xarray has dimensions 'lat', 'long', 'time', and
                'dummy_data' (by default)
        '''
        # TODO Move self.lat/long = np.linspace here after reg tests pass

        # Generate abstract data set
        if self.dataloader_name == 'circle':
            data = self.gen_circle(bounds)
        elif self.dataloader_name == 'checkerboard':
            data = self.gen_checkerboard(bounds)
        elif self.dataloader_name == 'gradient':
            data = self.gen_gradient(bounds)
        else:
            raise ValueError(
                f'Unknown abstract shape type: {self.dataloader_name}'
                )

        data_xr = data.set_index(['lat', 'long']).to_xarray()
        # No need to trim data, as was defined by bounds

        return data_xr
    
    def gen_circle(self, bounds):
        """
            Generates a circle within bounds of lat/long min/max.
            Circle centre and radius can be defined in the config, as well as
            resolution of simulated datapoints
            Args:
                bounds (Boundary): Limits of lat/long to generate within
        """
        logging.info("\tSetting up boundary of dataset")
        # Generate rows
        self.lat  = np.linspace(bounds.get_lat_min(), bounds.get_lat_max(), self.ny)    
        # Generate cols
        self.long = np.linspace(bounds.get_long_min(),bounds.get_long_max(),self.nx)        

        # Set centre as centre of data_grid if none specified
        c_y = self.lat[int(self.ny/2)]  if not self.centre[0] else self.centre[0]
        c_x = self.long[int(self.nx/2)] if not self.centre[1] else self.centre[1]
        
        # Create vectors for row and col idx's
        y = np.vstack(np.linspace(bounds.get_lat_min(), 
                                  bounds.get_lat_max(), 
                                  self.ny))
        x = np.linspace(bounds.get_long_min(), bounds.get_long_max(), self.nx)

        logging.info("\tCreating mask of circle")
        # Create a 2D-array with distance from defined centre
        dist_from_centre = np.sqrt((x-c_x)**2 + (y-c_y)**2)
        # Turn this into a mask of values within radius
        mask = dist_from_centre <= self.radius
        # Set up empty dataframe to populate with dummy data
        dummy_df = pd.DataFrame(columns = ['lat', 'long', 'dummy_data'])
        logging.info("\tGenerating dataset")
        # For each combination of lat/long
        for i in range(self.ny):
            for j in range(self.nx):
                # Create a new row, adding mask value
                row = pd.DataFrame(data={'lat':self.lat[i], 
                                         'long':self.long[j], 
                                         'dummy_data':mask[i][j]}, index=[0])
                dummy_df = pd.concat([dummy_df, row], ignore_index=True)
                
        # Change boolean values to int
        dummy_df = dummy_df.replace(False, 0)
        dummy_df = dummy_df.replace(True, 1)

        return dummy_df

    def gen_gradient(self, bounds):
        """
            Generates a gradient within bounds of lat/long min/max.
            Gradient direction can be defined in the config, as well as
            resolution of simulated datapoints
            Args:
                bounds (Boundary): Limits of lat/long to generate within
        """
        logging.info("\tSetting up boundary of dataset")
        # Generate rows
        self.lat  = np.linspace(bounds.get_lat_min(), 
                                bounds.get_lat_max(), 
                                self.ny)    
        # Generate cols
        self.long = np.linspace(bounds.get_long_min(),
                                bounds.get_long_max(),
                                self.nx)
        
        logging.info("\tCreating gradient of values")
        #Create 1D gradient
        if self.vertical:   gradient = np.linspace(0,1,self.ny)
        else:               gradient = np.linspace(0,1,self.nx)
            
        dummy_df = pd.DataFrame(columns = ['lat', 'long', 'dummy_data'])
        logging.info("- Generating dataset")
        # For each combination of lat/long
        for i in range(self.ny):
            for j in range(self.nx):
                # Change dummy data depending on which axis to gradient
                datum = gradient[i] if self.vertical else gradient[j]
                # Create a new row, adding datum value
                row = pd.DataFrame(data={'lat':self.lat[i], 
                                         'long':self.long[j], 
                                         'dummy_data':datum}, index=[0])
                dummy_df = pd.concat([dummy_df, row], ignore_index=True)  
        
        return dummy_df

    def gen_checkerboard(self, bounds):
        """
            Generates a checkerboard pattern within bounds of lat/long min/max
            Square size can be defined in the config, as well as resolution of 
            simulated datapoints
            Args:
                bounds (Boundary): Limits of lat/long to generate within
        """
        logging.info("\tSetting up boundary of dataset")
        # Generate rows
        self.lat  = np.linspace(bounds.get_lat_min(), 
                                bounds.get_lat_max(), 
                                self.ny, endpoint=False)    
        # Generate cols
        self.long = np.linspace(bounds.get_long_min(),
                                bounds.get_long_max(),
                                self.nx, endpoint=False)

        logging.info("- Creating series of 0's and 1's for lat/long")
        # Create checkerboard pattern
        # Create horizontal stripes of 0's and 1's, stripe size defined by gridsize
        horizontal = np.floor((self.lat - bounds.get_lat_min()) \
                              / self.gridsize[1]) % 2
        # Create vertical stripes of 0's and 1's, stripe size defined by gridsize
        vertical   = np.floor((self.long - bounds.get_long_min())\
                              / self.gridsize[0]) % 2   
        dummy_df = pd.DataFrame(columns = ['lat', 'long', 'dummy_data'])
        logging.info("- Generating dataset")
        # For each combination of lat/long
        for i in range(self.ny):
            for j in range(self.nx):
                # Horizontal XOR Vertical should create boxes
                datum = (horizontal[i] + vertical[j]) % 2
                # Create a new row, adding datum value
                row = pd.DataFrame(data={'lat':self.lat[i], 
                                         'long':self.long[j], 
                                         'dummy_data':datum}, index=[0])
                dummy_df = pd.concat([dummy_df, row], ignore_index=True)
        
        return dummy_df    

    # TODO Add square
