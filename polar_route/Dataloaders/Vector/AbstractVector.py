from polar_route.Dataloaders.DataLoaderInterface import DataLoaderInterface
from abc import abstractmethod

from pyproj import Transformer, CRS

import numpy as np
import xarray as xr
import pandas as pd

class VectorDataLoader(DataLoaderInterface):
    '''
    Abstract class for all vector Datasets

    Methods:
        __init__:
            User defined. This is where large-scale operations are performed, 
            e.g.
                - params are set as object attributes
                - self.data set
                - self.data downsampled
                - self.data reprojected
                - self.data_name set
                
        import_data:
            User defined. Must return either pd.DataFrame or xarray.Dataset
            with lat, long, (time), data
        get_datapoints:
            Returns pd.DataFrame of datapoints within Boundary object
        get_value:
            Returns aggregated value of data within Boundary. 
            Returns as dict {data_name: data_value}
        get_hom_condition:
            Retrieves homogeneity condition of dataset within boundary. This
            defines how cellboxes are split in the mesh.
        reproject:
            Reprojects dataset from imported projection to desired projection
            Default desired projection is mercator
        downsample:
            Bins data to be more sparse if input dataset too large to handle
            Will use aggregation method defined in 'params' to bin data
        get_data_col_name:
            Returns label of data from column name in 'self.data'
        set_data_col_name:
            Changes column/data variable name in self.data to a user defined
            string. 
        col_names_to_str:
            Creates a string combining all data column names, comma seperated 
    '''
    @abstractmethod
    def __init__(self, bounds, params, min_dp):
        '''
        Args:
            bounds (Boundary): 
                Initial mesh boundary to limit scope of data ingest
            params (dict): 
                Values needed by dataloader to initialise. Unique to each
                dataloader
            min_dp (int): 
                Minimum datapoints required to get homogeneity condition
                
        User defines:
            self.data = self.import_data(bounds)
            self.data = self.downsample() [OPTIONAL]
            self.data = self.reproject(   [OPTIONAL]
                                in_proj= Current projection e.g. 'EPSG:3412',
                                out_proj= Desired projection e.g. 'EPSG:4326',
                                x_col= Name of position column 1 e.g. 'x',
                                y_col= Name of position column 2 e.g. 'y'
                                )
            self.data_name = self.get_data_col_name() [OPTIONAL]
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
    
    def get_dp_from_coord(self, long=None, lat=None):
        '''
        Extracts datapoint from self.data with lat and long specified in kwargs.
        self.data can be pd.DataFrame or xr.Dataset
        
        Args:
            long (float): Longitude coordinate to search for
            lat (float) : Latitude coordinate to search for
            
        Returns:
            data (pd.Series): Column of data values with chosen lat/long
                              Could be many datapoints because either bad data
                              or multiple time steps 
        '''
        def get_dp_from_coord_df(data, names, long, lat):
            '''
            Extracts data from a pd.DataFrame
            '''
            # Mask off any positions not within spatial bounds
            mask = (data['lat']  == lat)  & \
                   (data['long'] == long) 

            # Return column of data from within bounds
            return data.loc[mask][names.split(',')]
        
        def get_dp_from_coord_xr(data, names, long, lat):
            '''
            Extracts data from a xr.Dataset
            '''
            # Select data region within spatial bounds
            data = data.sel(lat=lat, long=long)
            # Cast as a pd.DataFrame
            data = data.to_dataframe().reset_index()
            # Return column of data from within bounds
            return data[names.split(',')]
        
        # Ensure that lat and long provided
        assert (lat is not None) and (long) is not None, \
            'Must provide lat and long to this method!'
            
        # Choose which method to retrieve data based on input type
        if hasattr(self, 'data_name'): data_name = self.data_name
        else:                          data_name = self.get_data_col_name()
        
        if type(self.data) == type(pd.DataFrame()):
            return get_dp_from_coord_df(self.data, data_name, long, lat)
        elif type(self.data) == type(xr.Dataset()):
            return get_dp_from_coord_xr(self.data, data_name, long, lat)
        

    def get_datapoints(self, bounds, return_coords=False):
        '''
        Extracts datapoints from self.data within boundary defined by 'bounds'.
        self.data can be pd.DataFrame or xr.Dataset
        
        Args:
            bounds (Boundary): Limits of lat/long/time to select data from
            
        Returns:
            data (pd.Series): Column of data values within selected region 
        '''
        def get_datapoints_from_df(data, names, bounds, return_coords):
            '''
            Extracts data from a pd.DataFrame
            '''
            # Mask off any positions not within spatial bounds
            # TODO Change <= to < after regression tests pass
            mask = (data['lat']  >= bounds.get_lat_min())  & \
                   (data['lat']  <= bounds.get_lat_max())  & \
                   (data['long'] > bounds.get_long_min()) & \
                   (data['long'] <= bounds.get_long_max())
            # Mask with time if time column exists
            if 'time' in data.columns:
                mask &= (data['time'] >= bounds.get_time_min()) & \
                        (data['time'] <= bounds.get_time_max())
            # Return column of data from within bounds
            # TODO add dropna() when merged, standard didn't have it 
            # Extract lat/long/time if requested
            if return_coords:   
                columns = ['lat', 'long']
                columns += names.split(',')
                if 'time' in data.columns:
                    columns += ['time']
            else:
                columns = names.split(',')
            # Return column of data from within bounds
            return data.loc[mask][columns]
        
        def get_datapoints_from_xr(data, names, bounds, return_coords):
            '''
            Extracts data from a xr.Dataset
            '''
            # Select data region within spatial bounds
            data = data.sel(lat=slice(bounds.get_lat_min(),  bounds.get_lat_max() ))
            data = data.sel(long=slice(bounds.get_long_min(), bounds.get_long_max()))
            # Select data region within temporal bounds if time exists as a coordinate
            if 'time' in data.coords.keys():
                data = data.sel(time=slice(bounds.get_time_min(),  bounds.get_time_max()))
            # Cast as a pd.DataFrame
            # TODO add dropna() when merged, standard didn't have it
            data = data.to_dataframe().reset_index()#.dropna()
            # Extract lat/long/time if requested
            if return_coords:   
                columns = ['lat', 'long']
                columns += names.split(',')
                if 'time' in data.columns:
                    columns += ['time']
            else:               
                columns = names.split(',')
            # Return column of data from within bounds
            return data.loc[columns]

            
        # Choose which method to retrieve data based on input type
        if hasattr(self, 'data_name'): data_name = self.data_name
        else:                          data_name = self.get_data_col_name()
        
        # Choose which method to retrieve data based on input type
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
                Accepts 'MIN', 'MAX', 'MEAN', 'STD'
                Errors on 'MEDIAN' since nonsensical for 2D vectors
            bounds (Boundary): Boundary object with limits of lat/long
            skipna (bool): Defines whether to propogate NaN's or not
                Default = True (ignore's NaN's)

        Returns:
            aggregate_value (float): Aggregated value within bounds following
                aggregation_type
        '''
        def extract_vals(row, col_vars):
            '''
            Extracts column variable values from a row, returns them in a 
            dictionary {variable: value}
            '''
            # Initialise empty dictionary
            values = {}
            # For each variable
            for col in col_vars:
                # If there isn't a row (i.e. no data), value is NaN
                if row == None: 
                    values[col] = np.nan
                # Otherwise, extract value from row
                else:           
                    values[col] = row[col]
            return values
        # Set to params if no specific aggregate type specified
        if agg_type is None:
            agg_type = self.aggregate_type
        # Get list of variables that aren't coords
        col_vars = self.get_data_col_name().split(',')
        # Remove lat, long and time column if they exist
        dps = self.get_datapoints(bounds)[col_vars]
        # Create a magnitude column 
        dps['mag'] = np.sqrt(np.square(dps).sum(axis=1))

        # If no data
        if len(dps) == 0:
            row = {col: np.nan for col in col_vars}
        # Return float of aggregated value
        elif agg_type == 'MIN': # Find min mag vector
            row = dps[dps.mag == dps.mag.min(skipna=skipna)]
        elif agg_type == 'MAX': # Find max mag vector
           row = dps[dps.mag == dps.mag.max(skipna=skipna)]
        elif agg_type == 'MEAN': # Average each vector axis
            # TODO below is a workaround to make this work like standard code. Needs a fix
            row = {col: dps[col].mean(skipna=skipna) for col in col_vars}
        elif agg_type == 'STD': # Std Dev each vector axis
            # TODO Needs a fix like above statement
            row = {col: dps[col].std(skipna=skipna) for col in col_vars}
        elif agg_type == 'COUNT':
            row = {col: len(dps[col].dropna()) for col in col_vars}
        # Median of vectors does not make sense
        elif agg_type == 'MEDIAN':
            raise ArithmeticError('Cannot find median of multi-dimensional variable!')
        # If aggregation_type not available
        else:
            raise ValueError(f'Unknown aggregation type {self.aggregate_type}')
        
        # Extract variable values from single row (or dict) and return
        return extract_vals(row, col_vars)

    # TODO get_hom_condition()
    def get_hom_condition(self, bounds, splitting_conds, agg_type):
        pass
    # Using Curl / Divergence / Vorticity
    # Reynolds number?
        
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
            df = data.to_dataframe().reset_index()
            return reproject_df(df, in_proj, out_proj, x_col, y_col)
        
        # If no reprojection to do
        if in_proj == out_proj:
            return self.data
        # Choose appropriate method of reprojection based on data type
        elif type(self.data) == type(pd.DataFrame()):
            return reproject_df(self.data, in_proj, out_proj, x_col, y_col)
        elif type(self.data) == type(xr.Dataset()):
            return reproject_xr(self.data, in_proj, out_proj, x_col, y_col)
    
    def downsample(self, agg_type=None):
        '''
        Downsamples imported data to be more easily manipulated
        self.data can be pd.DataFrame or xr.Dataset

        Returns:
            data (xr.Dataset or pd.DataFrame): 
                Dataset downsampled by factors defined in params, 
                and binned via aggregation method defined in params
        '''
        def downsample_xr(data, ds, agg):
            '''
            Downsample xarray dataset
            '''
            # TODO Replace with coarsen when refactor passes regression tests
            # Better method of downsampling
            # data = data.coarsen(lat=self.ds[1]).max()
            # data = data.coarsen(lon=self.ds[0]).max()
            
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
            return self.data
        # Otherwise, downsample appropriately
        elif type(self.data) == type(pd.DataFrame()):
            return downsample_df(self.data, self.downsample_factors, agg_type)
        elif type(self.data) == type(xr.Dataset()):
            return downsample_xr(self.data, self.downsample_factors, agg_type)
        
    def get_data_col_name(self):
        '''
        Retrieve name of data column

        Returns:
            data_names [str]: List of strings of the column names
        '''
        def get_data_names_from_df(data):
            '''
            Filters out standard columns to extract only data column's name
            '''
            # Store name of data column for future reference
            columns = data.columns
            # Filter out lat, long, time columns leaves us with data column name
            filtered_cols = filter(lambda col: \
                                    col not in ['lat','long','time'], columns)
            data_names = list(filtered_cols)
            # Turn into comma seperated string and return
            return ','.join(data_names)
        
        def get_data_names_from_xr(data):
            '''
            Extracts variable name directly from xr.Dataset metadata
            '''
            # Extract data variables from xr.Dataset
            data_names = list(data.keys())
            # Turn into comma seperated string and return
            return ','.join(data_names)
        
        # Choose method of extraction based on data type
        if type(self.data) == type(pd.DataFrame()):
            return get_data_names_from_df(self.data)
        elif type(self.data) == type(xr.Dataset()):
            return get_data_names_from_xr(self.data)

    def set_data_col_name(self, name_dict):
        '''
        Sets name of data column/data variable
        
        Args:
            name_dict {str:str}: 
                Dictionary mapping old variable names to new variable names

        Returns:
            data (xr.Dataset or pd.DataFrame): 
                Data with variable name changed
        '''
        def set_names_df(data, name_dict):
            '''
            Renames data columns in pandas dataframe
            '''
            # Rename data column to new name
            return data.rename(columns=name_dict)
        def set_names_xr(data, name_dict):
            '''
            Renames data variables in xarray dataset
            '''
            # Rename data variable to new name
            return data.rename(name_dict)
        
        # Change data name depending on data type
        if type(self.data) == type(pd.DataFrame()):
            return set_names_df(self.data, name_dict)
        elif type(self.data) == type(xr.Dataset()):
            return set_names_xr(self.data, name_dict)
