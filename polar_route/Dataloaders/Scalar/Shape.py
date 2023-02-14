from .AbstractScalar import ScalarDataLoader#

import pandas as pd
import numpy as np

class ShapeDataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Set up data
        self.data = self.import_data(bounds)
        self.data = self.set_data_col_name("dummy_data")
        # self.data = self.set_data_col_name(self.shape)
        # self.data_name = self.get_data_col_name()
        self.data_name = "dummy_data"
    
    def import_data(self, bounds):
        # TODO Move self.lat/long = np.linspace here after reg tests pass
        # Choose appropriate shape to generate
        if self.shape == 'circle':
            data = self._gen_circle(bounds)
        elif self.shape == 'checkerboard':
            data = self._gen_checkerboard(bounds)
        elif self.shape == 'gradient':
            data = self._gen_gradient(bounds)
    
        # Fill dummy time values
        data['time'] = bounds.get_time_min()
    
        return data
    
    def _gen_circle(self, bounds):
        """
            Generates a circle within bounds of lat/long min/max.

            Args:
                bounds (Boundary): Limits of lat/long to generate within
        """
        # Generate rows
        self.lat  = np.linspace(bounds.get_lat_min(), bounds.get_lat_max(), self.ny)    
        # Generate cols
        self.long = np.linspace(bounds.get_long_min(),bounds.get_long_max(),self.nx)        

        # Set centre as centre of data_grid if none specified
        c_y = self.lat[int(self.ny/2)]  if not self.centre[0] else self.centre[0]
        c_x = self.long[int(self.nx/2)] if not self.centre[1] else self.centre[1]
        
        # Create vectors for row and col idx's
        y = np.vstack(np.linspace(bounds.get_lat_min(), bounds.get_lat_max(), self.ny))
        x = np.linspace(bounds.get_long_min(), bounds.get_long_max(), self.nx)
    
        # Create a 2D-array with distance from defined centre
        dist_from_centre = np.sqrt((x-c_x)**2 + (y-c_y)**2)
        # Turn this into a mask of values within radius
        mask = dist_from_centre <= self.radius
        # Set up empty dataframe to populate with dummy data
        dummy_df = pd.DataFrame(columns = ['lat', 'long', 'dummy_data'])
        # For each combination of lat/long
        for i in range(self.ny):
            for j in range(self.nx):
                # Create a new row, adding mask value
                row = pd.DataFrame(data={'lat':self.lat[i], 'long':self.long[j], 'dummy_data':mask[i][j]}, index=[0])
                dummy_df = pd.concat([dummy_df, row], ignore_index=True)
                
        # Change boolean values to int
        dummy_df = dummy_df.replace(False, 0)
        dummy_df = dummy_df.replace(True, 1)

        return dummy_df

    def _gen_gradient(self, bounds):
        """
            Generates a gradient across the map
            
            Args:
                bounds (Boundary): Limits of lat/long to generate within
        """
        # Generate rows
        self.lat  = np.linspace(bounds.get_lat_min(), bounds.get_lat_max(), self.ny)    
        # Generate cols
        self.long = np.linspace(bounds.get_long_min(),bounds.get_long_max(),self.nx)
        
        #Create 1D gradient
        if self.vertical:   gradient = np.linspace(0,1,self.ny)
        else:               gradient = np.linspace(0,1,self.nx)
            
        dummy_df = pd.DataFrame(columns = ['lat', 'long', 'dummy_data'])
        # For each combination of lat/long
        for i in range(self.ny):
            for j in range(self.nx):
                # Change dummy data depending on which axis to gradient
                datum = gradient[i] if self.vertical else gradient[j]
                # Create a new row, adding datum value
                row = pd.DataFrame(data={'lat':self.lat[i], 'long':self.long[j], 'dummy_data':datum}, index=[0])
                dummy_df = pd.concat([dummy_df, row], ignore_index=True)  
        
        return dummy_df

    def _gen_checkerboard(self, bounds):
        """
            Generates a checkerboard pattern across map
            
            Args:
                bounds (Boundary): Limits of lat/long to generate within
        """
        # Generate rows
        self.lat  = np.linspace(bounds.get_lat_min(), bounds.get_lat_max(), self.ny, endpoint=False)    
        # Generate cols
        self.long = np.linspace(bounds.get_long_min(),bounds.get_long_max(),self.nx, endpoint=False)

        # Create checkerboard pattern
        # Create horizontal stripes of 0's and 1's, stripe size defined by gridsize
        horizontal = np.floor((self.lat - bounds.get_lat_min()) \
                              / self.gridsize[1]) % 2
        # Create vertical stripes of 0's and 1's, stripe size defined by gridsize
        vertical   = np.floor((self.long - bounds.get_long_min())\
                              / self.gridsize[0]) % 2   
        
        dummy_df = pd.DataFrame(columns = ['lat', 'long', 'dummy_data'])
        # For each combination of lat/long
        for i in range(self.ny):
            for j in range(self.nx):
                # Horizontal XOR Vertical should create boxes
                datum = (horizontal[i] + vertical[j]) % 2
                # Create a new row, adding datum value
                row = pd.DataFrame(data={'lat':self.lat[i], 'long':self.long[j], 'dummy_data':datum}, index=[0])
                dummy_df = pd.concat([dummy_df, row], ignore_index=True)  
        
        return dummy_df    

    # TODO finish this
    def _gen_square(self, bounds):
        """
            Generates a square within bounds
            
            Args:
                bounds (Boundary): Limits of lat/long to generate within
        """
        # Set centre as centre of data_grid if none specified
        c_y = self.lat[int(self.ny/2)]  if not self.centre[0] else self.centre[0]
        c_x = self.long[int(self.nx/2)] if not self.centre[1] else self.centre[1]
        
        # Find indexes of central point
        c_idx_y = np.abs(self.lat - c_y).argmin()
        c_idx_x = np.abs(self.long - c_x).argmin()
        
        
        #
        mask = np.zeros((self.ny, self.nx))
        
        y_mask = (c_y - self.lat <= self.side_length/2) & \
                 (self.lat - c_y <= self.side_length/2)
        x_mask = (c_x - self.lat <= self.side_length/2) & \
                 (self.lat - c_x <= self.side_length/2)
        
        mask[y_mask][x_mask] = 1

        # Set up empty dataframe to populate with dummy data
        dummy_df = pd.DataFrame(columns = ['lat', 'long', 'dummy_data'])
        # For each combination of lat/long
        for i in range(self.ny):
            for j in range(self.nx):
                # Create a new row, adding mask value
                row = pd.DataFrame(data={'lat':self.lat[i], 'long':self.long[j], 'dummy_data':mask[i][j]}, index=[0])
                dummy_df = pd.concat([dummy_df, row], ignore_index=True)
                
        # Change boolean values to int
        dummy_df = dummy_df.replace(False, 0)
        dummy_df = dummy_df.replace(True, 1)

        return dummy_df

