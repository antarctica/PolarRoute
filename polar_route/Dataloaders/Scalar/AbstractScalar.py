from pyproj import Transformer, CRS

from polar_route.Dataloaders.DataLoaderInterface import DataLoaderInterface
from abc import abstractmethod

import xarray as xr
import pandas as pd
import numpy as np
import logging

class ScalarDataLoader(DataLoaderInterface):
    '''
    Abstract class for all scalar Datasets. Has the following methods
    '''
    @abstractmethod
    def __init__(self, bounds, params, min_dp):
        '''
        User defined. This is where large-scale operations are performed, 
        such as importing data, downsampling, reprojecting, and renaming 
        variables
        
        Args:
            bounds (Boundary): 
                Initial mesh boundary to limit scope of data ingest
            params (dict): 
                Values needed by dataloader to initialise. Unique to each
                dataloader
            min_dp (int): 
                Minimum datapoints required to get homogeneity condition
                
        Attributes:
            self.data (pd.DataFrame or xr.Dataset): 
                Data stored by dataloader to use when called upon by the mesh.
                Must be saved in mercator projection (EPSG:4326), with 
                coordinates names 'lat', 'long', and 'time' (if applicable).
            self.data_name (str): 
                Name of scalar variable. Must be the column name if self.data
                is pd.DataFrame. Must be variable if self.data is xr.Dataset
        '''
        pass

    @abstractmethod
    def import_data(self, bounds):
        '''
        User defined method for importing data from files, or even generating 
        data from scratch
                
        Returns:
            data (xr.Dataset or pd.DataFrame):
                if xr.Dataset, 
                    - Must have coordinates 'lat' and 'long'
                    - Must have single data variable
                    - Useful methods include 
                if pd.DataFrame, 
                    - Must have columns 'lat' and 'long'
                    - Must have single data column
                    
                Downsampling and reprojecting happen in __init__() method
        '''
        pass

    def get_dp_from_coord(self, long=None, lat=None, return_coords=False):
        '''
        Extracts datapoint from self.data with lat and long specified in kwargs.
        self.data can be pd.DataFrame or xr.Dataset. Will return multiple values
        if one set of coordinates have multiple entries (e.g. time series data)
        
        Args:
            long (float): Longitude coordinate to search for
            lat (float) : Latitude coordinate to search for
            
        Returns:
            pd.Series:  Column of data values with chosen lat/long. Could be many 
                datapoints because either bad data or multiple time steps 
        '''
        def get_dp_from_coord_df(data, name, long, lat, return_coords):
            '''
            Extracts data from a pd.DataFrame
            '''
            # Mask off any positions not within spatial bounds
            mask = (data['lat']  == lat)  & \
                   (data['long'] == long) 
                   
            # Include lat/long/time if requested
            if return_coords: columns = list(data.columns)
            else:             columns = [name]
            # Return column of data from within bounds
            return data.loc[mask][columns]
        
        def get_dp_from_coord_xr(data, name, long, lat, return_coords):
            '''
            Extracts data from a xr.Dataset
            '''
            # Select data region within spatial bounds
            data = data.sel(lat=lat, long=long)
            # Cast as a pd.DataFrame
            data = data.to_dataframe().reset_index()
            # Include lat/long/time if requested
            if return_coords: columns = list(data.columns)
            else:             columns = [name]
            # Return column of data from within bounds
            return data[columns]
        
        # Ensure that lat and long provided
        assert (lat is not None) and (long) is not None, \
            'Must provide lat and long to this method!'
            
        # Choose which method to retrieve data based on input type
        if hasattr(self, 'data_name'): data_name = self.data_name
        else:                          data_name = self.get_data_col_name()
        
        logging.debug(f'- Retrieving datapoint from a {type(self.data)}')
        if type(self.data) == type(pd.DataFrame()):
            return get_dp_from_coord_df(self.data, data_name, long, lat, return_coords)
        elif type(self.data) == type(xr.Dataset()):
            return get_dp_from_coord_xr(self.data, data_name, long, lat, return_coords)
        
        

    def get_datapoints(self, bounds, return_coords=False):
        '''
        Extracts datapoints from self.data within boundary defined by 'bounds'.
        self.data can be pd.DataFrame or xr.Dataset
        
        Args:
            bounds (Boundary): Limits of lat/long/time to select data from
            return_coords (boolean): 
                Flag to determine if coordinates are provided for each 
                datapoint found. Default is False.
                            
            
        Returns:
            data (pd.Series): Column of data values within selected region 
        '''
        def get_datapoints_from_df(data, name, bounds, return_coords):
            '''
            Extracts data from a pd.DataFrame
            '''
            # Mask off any positions not within spatial bounds
            mask = (data['lat']  > bounds.get_lat_min())  & \
                   (data['lat']  <= bounds.get_lat_max())  & \
                   (data['long'] > bounds.get_long_min()) & \
                   (data['long'] <= bounds.get_long_max())
            # Mask with time if time column exists
            if 'time' in data.columns:
                mask &= (data['time'] >= bounds.get_time_min()) & \
                        (data['time'] <= bounds.get_time_max())
                        
            # Include lat/long/time if requested
            if return_coords: columns = list(data.columns)
            else:             columns = [name]
            
            # Return column of data from within bounds
            return data.loc[mask][columns]
        
        def get_datapoints_from_xr(data, name, bounds, return_coords):
            '''
            Extracts data from a xr.Dataset
            '''
            # Select data region within spatial bounds
            # TODO make sure boundaries act same as df
            data = data.sel(lat=slice(bounds.get_lat_min(),  bounds.get_lat_max() ))
            data = data.sel(long=slice(bounds.get_long_min(), bounds.get_long_max()))
            # Select data region within temporal bounds if time exists as a coordinate
            if 'time' in data.coords.keys():
                data = data.sel(time=slice(bounds.get_time_min(),  bounds.get_time_max()))
            # Cast as a pd.DataFrame
            data = data.to_dataframe().reset_index().dropna()
            
            # Include lat/long/time if requested
            if return_coords: columns = list(data.columns)
            else:             columns = [name]
            
            # Return column of data from within bounds
            return data[columns]
            
        # Choose which method to retrieve data based on input type
        if hasattr(self, 'data_name'): data_name = self.data_name
        else:                          data_name = self.get_data_col_name()
        
        logging.debug(f'- Retrieving datapoint from a {type(self.data)}')
        if type(self.data) == type(pd.DataFrame()):
            return get_datapoints_from_df(self.data, data_name, bounds, return_coords)
        elif type(self.data) == type(xr.Dataset()):
            return get_datapoints_from_xr(self.data, data_name, bounds, return_coords)

    def get_value(self, bounds, agg_type=None, skipna=True):
        '''
        Retrieve aggregated value from within bounds
        
        Args:
            aggregation_type (str): Method of aggregation of datapoints within
                bounds. Can be upper or lower case. 
                Accepts 'MIN', 'MAX', 'MEAN', 'MEDIAN', 'STD', 'CLEAR'
            bounds (Boundary): Boundary object with limits of lat/long
            skipna (bool): Defines whether to propogate NaN's or not
                Default = True (ignore's NaN's)

        Returns:
            aggregate_value (float): Aggregated value within bounds following
                aggregation_type
        '''
        # Set to params if no specific aggregate type specified
        if agg_type is None:
            agg_type = self.aggregate_type
        # Limit data series to just the data, excluding coords/index
        dps = self.get_datapoints(bounds)[self.data_name]
        dps = dps.dropna().sort_values()
        logging.debug(f"- {len(dps)} datapoints found within bounds")
        # If no data
        if len(dps) == 0:
            return {self.data_name: np.nan}
        # Return float of aggregated value
        elif agg_type == 'MIN':
            return {self.data_name :float(dps.min(skipna=skipna))}
        elif agg_type == 'MAX':
            return {self.data_name :float(dps.max(skipna=skipna))}
        elif agg_type == 'MEAN':
            return {self.data_name :float(dps.mean(skipna=skipna))}
        elif agg_type == 'MEDIAN':
            return {self.data_name :float(dps.median(skipna=skipna))}
        elif agg_type == 'STD':
            return {self.data_name :float(dps.std(skipna=skipna))}
        elif agg_type =='COUNT':
            return {self.data_name: len(dps)}
        # If aggregation_type not available
        else:
            raise ValueError(f'Unknown aggregation type {agg_type}')

    def get_hom_condition(self, bounds, splitting_conds):
        '''
        Retrieve homogeneity condition
        
        Args: 
            bounds (Boundary): Boundary object with limits of datarange to analyse
            splitting_conds (dict):
                ['threshold'] (float):  The threshold at which data points of 
                    type 'value' within this CellBox are checked to be either 
                    above or below
                ['upper_bound'] (float): The lowerbound of acceptable percentage 
                    of data_points of type value within this CellBox that are 
                    above 'threshold'
                ['lower_bound'] (float): the upperbound of acceptable percentage 
                    of data_points of type value within this CellBox that are 
                    above 'threshold'

        Returns:
            hom_condition (string): The homogeniety condtion of this CellBox by 
                given parameters hom_condition is of the form -
                CLR = the proportion of data points within this cellbox over a 
                given threshold is lower than the lowerbound
                HOM = the proportion of data points within this cellbox over a
                given threshold is higher than the upperbound
                MIN = the cellbox contains less than a minimum number of 
                data points
                HET = the proportion of data points within this cellbox over a
                given threshold if between the upper and lower bound
        '''
        # Retrieve datapoints to analyse
        dps = self.get_datapoints(bounds)[self.data_name]
        logging.debug(f"- {len(dps)} datapoints found within bounds")
        # If not enough datapoints
        if len(dps) < self.min_dp: return 'MIN'
        # Otherwise, extract the homogeneity condition

        # Calculate fraction over threshold
        num_over_threshold = dps[dps > splitting_conds['threshold']]
        frac_over_threshold = num_over_threshold.shape[0]/dps.shape[0]
        
        # Return homogeneity condition
        if   frac_over_threshold <= splitting_conds['lower_bound']: return 'CLR'
        elif frac_over_threshold >= splitting_conds['upper_bound']: return 'HOM'
        else: return 'HET'
        
    def reproject(self, in_proj='EPSG:4326', out_proj='EPSG:4326', 
                        x_col='lat', y_col='long'):
        '''
        Reprojects data using pyProj.Transformer
        self.data can be pd.DataFrame or xr.Dataset
        
        Args:
            in_proj (str): 
                Projection that the imported dataset is in
                Must be allowed by PyProj.CRS (Coordinate Reference System)
            out_proj (str): 
                Projection required for final data output
                Must be allowed by PyProj.CRS (Coordinate Reference System)
                Shouldn't change from default value (EPSG:4326)
            x_col (str): Name of coordinate column 1
            y_col (str): Name of coordinate column 2
                x_col and y_col will be cast into lat and long by the 
                reprojection 
            
        Returns:
            data (pd.DataFrame): Reprojected data with 'lat', 'long' columns 
                replacing 'x_col' and 'y_col'
        '''
        def reproject_df(data, in_proj, out_proj, x_col, y_col):
            '''
            Reprojects a pandas dataframe
            '''
            # Do the reprojection
            x, y = Transformer\
                    .from_crs(CRS(in_proj), CRS(out_proj), always_xy=True)\
                    .transform(data[x_col].to_numpy(), data[y_col].to_numpy())
            # Replace columns with reprojected columns called 'lat'/'long'
            data = data.drop(x_col, axis=1)
            data = data.drop(y_col, axis=1)
            data['lat']  = y
            data['long'] = x
            
            return data
            
        def reproject_xr(data, in_proj, out_proj, x_col, y_col):
            '''
            Reprojects a xarray dataset
            '''
            # Cast to dataframe, then reproject using reproject_df
            # Cannot reproject directly as memory usage skyrockets
            df = data.to_dataframe().reset_index().dropna()
            return reproject_df(df, in_proj, out_proj, x_col, y_col)
        
        # If no reprojection to do
        if in_proj == out_proj:
            logging.debug("- self.reproject() called but don't need to")
            return self.data
        else:
            logging.info(f"- Reprojecting data from {in_proj} to {out_proj}")
        # Choose appropriate method of reprojection based on data type
        if type(self.data) == type(pd.DataFrame()):
            return reproject_df(self.data, in_proj, out_proj, x_col, y_col)
        elif type(self.data) == type(xr.Dataset()):
            return reproject_xr(self.data, in_proj, out_proj, x_col, y_col)
    
    def downsample(self, agg_type=None):
        '''
        Downsamples imported data to be more easily manipulated. Data size 
        should be reduced by a factor of m*n, where (m,n) are the 
        downsample_factors defined in the params.        
        self.data can be pd.DataFrame or xr.Dataset
        
        Args:
            agg_type (str): 
                Method of aggregation to bin data by to downsample. Default is
                same method used for homogeneity condition.            

        Returns:
            data (xr.Dataset or pd.DataFrame): 
                Downsampled data
                
        Todo:
            Fix downsampling to work with aggregation type as intended. Right now 
            it just takes every m'th and n'th datapoint in each coord axis
        '''
        def downsample_xr(data, ds, agg):
            '''
            Downsample xarray dataset
            '''
            # TODO Replace with coarsen when refactor passes regression tests
            # Better method of downsampling
            # data = data.coarsen(lat=ds[1]).max()
            # data = data.coarsen(long=ds[0]).max()
            # return data
            # Old method of downsampling
            return downsample_df(data.to_dataframe().reset_index(), ds, agg)
        
        def downsample_df(data, ds, agg):
            '''
            Downsample pandas dataframe
            '''
            # TODO Replace with aggregate type method of downsampling when
            # refactor passes regression tests
            
            # Old method of downsampling just takes every nth column and row
            # Retrieve each unique coordinate and downsample
            ds_lats = data.lat.unique()[::ds[1]]
            ds_lons = data.long.unique()[::ds[0]]
            # Cut down dataset to only those values with downsampled coords
            data = data[data.lat.isin(ds_lats)]
            data = data[data.long.isin(ds_lons)]
            
            return data
        
        # Set to params if no specific aggregate type specified
        if agg_type is None:
            agg_type = self.aggregate_type
        
        # If no downsampling
        if self.downsample_factors == (1,1):
            logging.debug("- self.downsample() called but don't have to")
            return self.data
        else:
            logging.info(f"- Downsampling data by {self.downsample_factors}")
        # Otherwise, downsample appropriately
        if type(self.data) == type(pd.DataFrame()):
            return downsample_df(self.data, self.downsample_factors, agg_type)
        elif type(self.data) == type(xr.Dataset()):
            return downsample_xr(self.data, self.downsample_factors, agg_type)
        
    def get_data_col_name(self):
        '''
        Retrieve name of data column (for pd.DataFrame), or variable 
        (for xr.Dataset). Used for when data_name not defined in params.

        Returns:
            data_name (str): Name of data column
            
        Raises:
            ValueError: 
                If multiple possible data columns found, can't retrieve data 
                name
        '''
        def get_data_name_from_df(data):
            '''
            Filters out standard columns to extract only data column's name
            '''
            # Store name of data column for future reference
            columns = data.columns
            # Filter out lat, long, time columns leaves us with data column name
            filtered_cols = filter(lambda col: \
                                    col not in ['lat','long','time'], columns)
            name = list(filtered_cols)
            if len(name) != 1:
                raise ValueError(
                f'More than 1 data column detected, cannot retrieve data \
                    name! Found columns: {",".join(name)}'
                                 )
            return name[0]
        
        def get_data_name_from_xr(data):
            '''
            Extracts variable name directly from xr.Dataset metadata
            '''
            # Extract data variables from xr.Dataset
            name = list(data.keys())
            # Ensure there's only 1 data column to read name from
            if len(name) != 1:
                raise ValueError(
                f'More than 1 data column detected, cannot retrieve data \
                    name! Found columns: {",".join(name)}'
                                 )
            return name[0]
        
        logging.info(f"- Retrieving data name from {type(self.data)}")
        # Choose method of extraction based on data type
        if type(self.data) == type(pd.DataFrame()):
            return get_data_name_from_df(self.data)
        elif type(self.data) == type(xr.Dataset()):
            return get_data_name_from_xr(self.data)

    def set_data_col_name(self, new_name):
        '''
        Sets name of data column/data variable
        
        Args:
            name (str): Name to replace currently stored name with

        Returns:
            data (xr.Dataset or pd.DataFrame): 
                Data with variable name changed
        '''
        def set_name_df(data, old_name, new_name):
            '''
            Renames data column in pandas dataframe
            '''
            # Rename data column to new name
            return data.rename(columns={old_name: new_name})
        def set_name_xr(data, old_name, new_name):
            '''
            Renames data variable in xarray dataset
            '''
            # Rename data variable to new name
            return data.rename({old_name: new_name})
        
        old_name = self.get_data_col_name()
        logging.info(f"- Changing data name from {old_name} to {new_name}")
        # Change data name depending on data type
        if type(self.data) == type(pd.DataFrame()):
            return set_name_df(self.data, old_name, new_name)
        elif type(self.data) == type(xr.Dataset()):
            return set_name_xr(self.data, old_name, new_name)
