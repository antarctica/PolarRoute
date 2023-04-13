from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader

import logging
import pandas as pd
import numpy as np

class ShapeDataLoader(ScalarDataLoader):
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
        # Choose appropriate shape to generate
        if self.shape == 'circle':
            data = self.gen_circle(bounds)
        elif self.shape == 'checkerboard':
            data = self.gen_checkerboard(bounds)
        elif self.shape == 'gradient':
            data = self.gen_gradient(bounds)
    
        # Fill dummy time values
        data['time'] = bounds.get_time_min()

        data_xr = data.set_index(['lat', 'long', 'time']).to_xarray()
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
