
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pyproj import Transformer, CRS

import xarray as xr
import pandas as pd
import numpy as np

import logging
import glob

from polar_route.Boundary import Boundary

class DataLoaderFactory:
    '''
    Produces initialised DataLoader objects
    '''    
    def get_dataloader(self, name, bounds, params, min_dp=5):
        '''
        Creates appropriate dataloader object based on name
        
        Args:
            name (str): 
                Name of data source/type. Must be one of following - 
                'GEBCO','AMSR','SOSE','thickness','density',
                'GRFScalar','GRFVector','GRFMask'
            bounds (Boundary): 
                Boundary object with initial mesh space&time limits
            params (dict): 
                Dictionary of parameters required by each dataloader
            min_dp (int):  
                Minimum datapoints required to get homogeneity condition

        Returns:
            data_loader (Scalar/Vector/LUT DataLoader): 
                DataLoader object of correct type, with required params set 
        '''
        # Cast name to lowercase to make case insensitive
        name = name.lower()
        # Add default values if they don't exist
        params = self.set_default_params(name, params, min_dp)
        
        dataloader_requirements = {
            # Scalar
            'dummy_scalar':(DummyScalarDataLoader, ['file']),
            'amsr':        (AMSRDataLoader, ['file', 'hemisphere']),
            'amsr_folder': (AMSRDataLoader, ['folder', 'hemisphere']),
            'bsose_sic':   (BSOSESeaIceDataLoader, ['file']),
            'bsose_depth': (BSOSEDepthDataLoader, ['file']),
            'baltic_sic':  (BalticSeaIceDataLoader, ['file']),
            'gebco':       (GEBCODataLoader, ['file']),
            'modis':       (MODISDataLoader, ['file']),
            # Scalar - Abstract shapes
            'circle':       (AbstractShapeDataLoader, ['shape', 'nx', 'ny', 'radius', 'centre']),
            'square':       (AbstractShapeDataLoader, ['shape', 'nx', 'ny', 'side_length', 'centre']),
            'gradient':     (AbstractShapeDataLoader, ['shape', 'nx', 'ny', 'vertical']),
            'checkerboard': (AbstractShapeDataLoader, ['shape', 'nx', 'ny', 'gridsize']),
            # Vector
            'dummy_vector':     (DummyVectorDataLoader, ['file']),
            'baltic_currents':  (BalticCurrentsDataLoader, ['file']),
            'era5_wind':        (ERA5WindDataLoader, ['file']),
            'northsea_currents':(NorthSeaCurrentsDataLoader, ['file']),
            'oras5_currents':   (ORAS5CurrentDataLoader, ['file_u', 'file_v']),
            'sose':             (SOSEDataLoader, ['file']),
            # Lookup Table
            # TODO actually make these LUT
            'thickness': (ThicknessDataLoader, []),
            'density':   (DensityDataLoader, [])
        }
        # If name is recognised as a dataloader
        if name in dataloader_requirements:
            # Set data loader and params required for it to work
            data_loader = dataloader_requirements[name][0]
            required_params = dataloader_requirements[name][1]
        else: 
            raise ValueError(f'{name} not in known list of DataLoaders')

        # Assert dataloader will get all required params to work
        assert all(key in params for key in required_params), \
            f'Dataloader {name} is missing some parameters! Requires {required_params}. Has {list(params.keys())}'

        # Create instance of dataloader
        return data_loader(bounds, params)
    
    def set_default_params(self, name, params, min_dp):
        '''
        Set default values for all dataloaders
        '''
        
        if 'downsample_factors' not in params:
            params['downsample_factors'] = (1,1)

        if 'data_name' not in params:
            params['data_name'] = name

        if 'aggregate_type' not in params: 
            params['aggregate_type']  = 'MEAN'
            
        if 'min_dp' not in params:
            params['min_dp'] = min_dp
            
        # Set defaults for abstract data generators
        if name in ['circle', 'checkerboard', 'gradient']:
            params = self.set_default_abstract_shape_params(name, params)
                
        return params
    
    def set_default_abstract_shape_params(self, name, params):
        '''
        Set default values for abstract shape dataloaders
        '''
        # Number of datapoints to populate per axis
        if 'nx' not in params:
            params['nx'] = 101
        if 'ny' not in params:
            params['ny'] = 101
            
        # Shape of abstract dataset
        if 'shape' not in params:
            params['shape'] = name
            
        # Define default circle parameters
        if name == 'circle':
            if 'radius' not in params:
                params['radius'] = 1
            if 'centre' not in params:
                params['centre'] = (None, None)
        # Define default square parameters
        elif name == 'square':
            if 'side_length' not in params:
                params['side_length'] = 1
            if 'centre' not in params:
                params['centre'] = (None, None)
        # Define default gradient params
        elif name == 'gradient':
            if 'vertical' not in params:
                params['vertical'] = True
        # Define default checkerboard params
        elif name == 'checkerboard':
            if 'gridsize' not in params:
                params['gridsize'] = (1,1)   
        
        
        return params

# ---------- ABSTRACTED DL OBJECTS ---------- #

class ScalarDataLoader(ABC):
    '''
    Abstract class for all scalar Datasets

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

    def get_datapoints(self, bounds):
        '''
        Extracts datapoints from self.data within boundary defined by 'bounds'.
        self.data can be pd.DataFrame or xr.Dataset
        
        Args:
            bounds (Boundary): Limits of lat/long/time to select data from
            
        Returns:
            data (pd.Series): Column of data values within selected region 
        '''
        def get_datapoints_from_df(data, name, bounds):
            '''
            Extracts data from a pd.DataFrame
            '''
            # Mask off any positions not within spatial bounds
            # TODO Change <= to < after regression tests pass
            mask = (data['lat']  > bounds.get_lat_min())  & \
                   (data['lat']  <= bounds.get_lat_max())  & \
                   (data['long'] > bounds.get_long_min()) & \
                   (data['long'] <= bounds.get_long_max())
            # Mask with time if time column exists
            if 'time' in data.columns:
                mask &= (data['time'] >= bounds.get_time_min()) & \
                        (data['time'] <= bounds.get_time_max())
            # Return column of data from within bounds
            return data.loc[mask][name]
        
        def get_datapoints_from_xr(data, name, bounds):
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
            data = data.to_dataframe().reset_index().dropna()
            # Return column of data from within bounds
            return data[name]
            
        # Choose which method to retrieve data based on input type
        if hasattr(self, 'data_name'): data_name = self.data_name
        else:                          data_name = self.get_data_col_name()
        
        if type(self.data) == type(pd.DataFrame()):
            return get_datapoints_from_df(self.data, data_name, bounds)
        elif type(self.data) == type(xr.Dataset()):
            return get_datapoints_from_xr(self.data, data_name, bounds)

    def get_value(self, bounds, skipna=True):
        '''
        Retrieve aggregated value from within bounds
        
        Args:
            aggregation_type (str): Method of aggregation of datapoints within
                bounds. Can be upper or lower case. 
                Accepts 'MIN', 'MAX', 'MEAN', 'MEDIAN', 'STD'
            bounds (Boundary): Boundary object with limits of lat/long
            skipna (bool): Defines whether to propogate NaN's or not
                Default = True (ignore's NaN's)

        Returns:
            aggregate_value (float): Aggregated value within bounds following
                aggregation_type
        '''

        # Remove lat, long and time column if they exist
        dps = self.get_datapoints(bounds).dropna().sort_values()
        # If no data
        if len(dps) == 0:
            return {self.data_name: np.nan}
        # Return float of aggregated value
        elif self.aggregate_type == 'MIN':
            return {self.data_name :float(dps.min(skipna=skipna))}
        elif self.aggregate_type == 'MAX':
            return {self.data_name :float(dps.max(skipna=skipna))}
        elif self.aggregate_type == 'MEAN':
            return {self.data_name :float(dps.mean(skipna=skipna))}
        elif self.aggregate_type == 'MEDIAN':
            return {self.data_name :float(dps.median(skipna=skipna))}
        elif self.aggregate_type == 'STD':
            return {self.data_name :float(dps.std(skipna=skipna))}
        # If aggregation_type not available
        else:
            raise ValueError(f'Unknown aggregation type {self.aggregate_type}')

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
            CLR = the proportion of data points within this cellbox over a given
                threshold is lower than the lowerbound
            HOM = the proportion of data points within this cellbox over a given
                threshold is higher than the upperbound
            MIN = the cellbox contains less than a minimum number of data points

            HET = the proportion of data points within this cellbox over a given
                threshold if between the upper and lower bound
        '''
        # Retrieve datapoints to analyse
        dps = self.get_datapoints(bounds)
        
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
            return self.data
        # Choose appropriate method of reprojection based on data type
        elif type(self.data) == type(pd.DataFrame()):
            return reproject_df(self.data, in_proj, out_proj, x_col, y_col)
        elif type(self.data) == type(xr.Dataset()):
            return reproject_xr(self.data, in_proj, out_proj, x_col, y_col)
    
    def downsample(self):
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
        
        # If no downsampling
        if self.downsample_factors == (1,1):
            return self.data
        # Otherwise, downsample appropriately
        elif type(self.data) == type(pd.DataFrame()):
            return downsample_df(self.data, self.downsample_factors, self.aggregate_type)
        elif type(self.data) == type(xr.Dataset()):
            return downsample_xr(self.data, self.downsample_factors, self.aggregate_type)
        
    def get_data_col_name(self):
        '''
        Retrieve name of data column

        Returns:
            data_name (str): Name of data column
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
            data_name = list(filtered_cols)[0]
            return data_name
        
        def get_data_name_from_xr(data):
            '''
            Extracts variable name directly from xr.Dataset metadata
            '''
            # Extract data variables from xr.Dataset
            name = list(data.keys())
            # Ensure there's only 1 data column to read name from
            assert len(name) == 1, \
                'More than 1 data column detected, cannot retrieve data name!'
            return name[0]
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
        
        # Change data name depending on data type
        if type(self.data) == type(pd.DataFrame()):
            return set_name_df(self.data, self.get_data_col_name(), new_name)
        elif type(self.data) == type(xr.Dataset()):
            return set_name_xr(self.data, self.get_data_col_name(), new_name)

class VectorDataLoader(ABC):
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

    def get_datapoints(self, bounds):
        '''
        Extracts datapoints from self.data within boundary defined by 'bounds'.
        self.data can be pd.DataFrame or xr.Dataset
        
        Args:
            bounds (Boundary): Limits of lat/long/time to select data from
            
        Returns:
            data (pd.Series): Column of data values within selected region 
        '''
        def get_datapoints_from_df(data, names, bounds):
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
            return data.loc[mask][names]
        
        def get_datapoints_from_xr(data, names, bounds):
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
            data = data.to_dataframe().reset_index()
            # Return column of data from within bounds
            return data[names]
            
        # Choose which method to retrieve data based on input type
        if type(self.data) == type(pd.DataFrame()):
            return get_datapoints_from_df(self.data, self.get_data_col_names(), bounds)
        elif type(self.data) == type(xr.Dataset()):
            return get_datapoints_from_xr(self.data, self.get_data_col_names(), bounds)

    def get_value(self, bounds, skipna=True):
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

        # Remove lat, long and time column if they exist
        dps = self.get_datapoints(bounds)
        # Get list of variables that aren't coords
        col_vars = self.get_data_col_names()
        # Create a magnitude column 
        dps['mag'] = np.sqrt(np.square(dps).sum(axis=1))

        # If no data
        if len(dps) == 0:
            row = None
        # Return float of aggregated value
        elif self.aggregate_type == 'MIN': # Find min mag vector
            row = dps[dps.mag == dps.mag.min(skipna=skipna)]
        elif self.aggregate_type == 'MAX': # Find max mag vector
           row = dps[dps.mag == dps.mag.max(skipna=skipna)]
        elif self.aggregate_type == 'MEAN': # Average each vector axis
            # TODO below is a workaround to make this work like standard code. Needs a fix
            row = {col: dps[col].mean(skipna=skipna) for col in col_vars}
        elif self.aggregate_type == 'STD': # Std Dev each vector axis
            # TODO Needs a fix like above statement
            row = {col: dps[col].std(skipna=skipna) for col in col_vars}
            
        # Median of vectors does not make sense
        elif self.aggregate_type == 'MEDIAN':
            raise ArithmeticError('Cannot find median of multi-dimensional variable!')
        # If aggregation_type not available
        else:
            raise ValueError(f'Unknown aggregation type {self.aggregate_type}')
        
        # Extract variable values from single row (or dict) and return
        return extract_vals(row, col_vars)

    # TODO get_hom_condition()
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
    
    def downsample(self):
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
        
        # If no downsampling
        if self.downsample_factors == (1,1):
            return self.data
        # Otherwise, downsample appropriately
        elif type(self.data) == type(pd.DataFrame()):
            return downsample_df(self.data, self.downsample_factors, self.aggregate_type)
        elif type(self.data) == type(xr.Dataset()):
            return downsample_xr(self.data, self.downsample_factors, self.aggregate_type)
        
    def get_data_col_names(self):
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
            return data_names
        
        def get_data_names_from_xr(data):
            '''
            Extracts variable name directly from xr.Dataset metadata
            '''
            # Extract data variables from xr.Dataset
            data_names = list(data.keys())
            return data_names
        
        # Choose method of extraction based on data type
        if type(self.data) == type(pd.DataFrame()):
            return get_data_names_from_df(self.data)
        elif type(self.data) == type(xr.Dataset()):
            return get_data_names_from_xr(self.data)

    def set_data_col_names(self, name_dict):
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

    def col_names_to_str(self):
        '''
        Turns list of col names into comma seperated string
        '''
        # Get col names
        col_names = self.get_data_col_names()
        # Join all cols into single string, seperated by commas
        cols_as_str = ','.join(col_names)
        return cols_as_str
    
class LookupTableDataLoader(ABC):
    
    # User defines Boundary objects
    # Have to ensure they cover entire domain of lat/long/time
    # Dict of boundary to value
    
    # When get_value called, input bounds
    # Figure which boundaries 'bounds' overlaps
    # calculate how much area/volume bounds takes within each Boundary
    # add all together to get value
    @abstractmethod
    def __init__(self, bounds, params):
        pass
    
    @abstractmethod
    def setup_lookup_table(self):
        # Returns dict of
        # {Boundary: value}
        pass

    def get_value(self, bounds):
        pass        

    def coord_to_boundary(self, coords):
        # If coord lat < boundary_lat_min, drop
        # If coords lat > boundary_lat_max, drop
        # Same for long
        # Same for time
        # Return list of boundaries that haven't been dropped
        pass
    
# ---------- SCALAR DATA LOADERS ---------- #
class AbstractShapeDataLoader(ScalarDataLoader):
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



# class AMSRDataLoader(ScalarDataLoader):
#     def __init__(self, bounds, params):
#         # Creates a class attribute for all keys in params
#         for key, val in params.items():
#             setattr(self, key, val)
          
#         self.data = self.import_data(bounds)
#         # self.data = self.downsample()
#         # self.data = self.set_data_col_name('z', 'SIC')
        
#         self.data = self.data.to_dataframe().reset_index().dropna()
#         # Set to lower case for case insensitivity
#         self.hemisphere = self.hemisphere.lower()
#         # Reproject to mercator
#         if self.hemisphere == 'north': 
#             self.data = self.reproject('EPSG:3411', 'EPSG:4326', x_col='x', y_col='y')
#         elif self.hemisphere == 'south':
#             self.data = self.reproject('EPSG:3412', 'EPSG:4326', x_col='x', y_col='y')
#         else:
#             raise ValueError('No hemisphere defined in params!')
        
#         self.data['SIC'] = self.data['z']
#         # Limit dataset to just values within bounds
#         self.data = self.data.loc[self.get_datapoints(bounds).index]
        
#         print('')
        
#     def import_data(self, bounds):
#         '''
#         Load AMSR netCDF, either as single '.nc' file, 
#         or from a folder of '.nc' files
#         '''
#         def retrieve_date(filename):
#             '''
#             Get date from filename in format:
#                 asi-AMSR2-s6250-<year><month><day>-v.5.4.nc
#             '''
#             date = filename.split('-')[-2]
#             date = f'{date[:4]}-{date[4:6]}-{date[6:]}'
#             return date
        
#         def retrieve_data(filename, date):
#             '''
#             Read in data as xr.Dataset, create time coordinate
#             '''
#             data = xr.open_dataset(filename)
#             # Add date to data
#             data = data.assign_coords(time=date)
#             return data
        
#         logging.debug("Importing AMSR data...")

#         # If single NetCDF File specified
#         if hasattr(self, 'file'):
#             # Ensure .nc file passed in params
#             assert(self.file[-3:] == '.nc')
#             logging.debug(f"- Opening file {self.file}")
#             # Extract data, append date to coords
#             date = retrieve_date(self.file)
#             data = retrieve_data(self.file, date)
#         # If folder specified
#         elif hasattr(self, 'folder'):
#             # Open folder and read in files
#             logging.debug(f"- Searching folder {self.folder}")
#             data_array = []
#             # For each .nc file in folder
#             for file in sorted(glob.glob(f'{self.folder}*.nc')):
#                 logging.debug(f"- Opening file {file}")
#                 # If date within boundary
#                 date = retrieve_date(file)
#                 # If file data from within time boundary, append to list
#                 # Doing now to avoid ingesting too much data initially
#                 if datetime.strptime(bounds.get_time_min(), '%Y-%m-%d') <= \
#                    datetime.strptime(date, '%Y-%m-%d') <= \
#                    datetime.strptime(bounds.get_time_max(), '%Y-%m-%d'):
#                     data_array.append(retrieve_data(file, date))
#             # Concat all valid files
#             data = xr.concat(data_array,'time')
#         # Otherwise need a file or folder to read from
#         else:
#             raise ValueError('File or folder not specified in params!')

#         # Remove unnecessary column
#         # data = data.drop_vars('polar_stereographic')
        
#         return data

class AMSRDataLoader:
    
    '''
    Abstract class for all scalar datasets

    Args:
        file_location (str): Path to data file or folder
        min_dp (int)   : Minimum number of datapoints to require per cellbox
            before allowing HOM condition to be calculated
        ds (int, int)  : Tuple of downsampling factors in lat, long
        data_name (str): Name of data, also name of data column in self.data
        aggregate_type (str): Type of aggregation to be used when calling
            self.get_hom_condition()
    '''
    def __init__(self, bounds, params):

    # def __init__(self, file_location, bounds, min_dp=5, ds=None, data_name=None, aggregate_type="MEAN"):
        self.file_location  = params['folder']
        self.min_dp         = params['min_dp']
        self.ds             = params['downsample_factors']
        self.hemisphere     = params['hemisphere'].lower()

        # Cast string to uppercase to accept mismatched case
        self.aggregate_type = params['aggregate_type'].upper()

        self.data = self.import_data(bounds)
        # If no data name specified, retrieve from self.data
        self.data_name = 'SIC' #data_name if data_name else self.get_data_name()
        
        logging.debug(f'- Successfully extracted {self.data_name}')    

    def import_data(self, bounds):
        
        '''
        Load AMSR netCDF, either as single '.nc' file, 
        or from a folder of '.nc' files
        '''
        def retrieve_date(filename):
            '''
            Get date from filename in format:
                asi-AMSR2-s6250-<year><month><day>-v.5.4.nc
            '''
            date = filename.split('-')[-2]
            date = f'{date[:4]}-{date[4:6]}-{date[6:]}'
            return date
        
        def retrieve_data(filename, date):
            '''
            Read in data as xr.Dataset, create time coordinate
            '''
            data = xr.open_dataset(filename)
            # Add date to data
            data = data.assign_coords(time=date)
            return data
        
        logging.debug("Importing AMSR data...")
        
        if self.file_location[-3:] == '.nc':
            self.file = self.file_location
        else:
            self.folder = self.file_location


        # If single NetCDF File specified
        if hasattr(self, 'file'):
            # Ensure .nc file passed in params
            assert(self.file[-3:] == '.nc')
            logging.debug(f"- Opening file {self.file}")
            # Extract data, append date to coords
            date = retrieve_date(self.file)
            data = retrieve_data(self.file, date)
        # If folder specified
        elif hasattr(self, 'folder'):
            # Open folder and read in files
            logging.debug(f"- Searching folder {self.folder}")
            data_array = []
            # For each .nc file in folder
            for file in sorted(glob.glob(f'{self.folder}*.nc')):
                logging.debug(f"- Opening file {file}")
                # If date within boundary
                date = retrieve_date(file)
                # If file data from within time boundary, append to list
                # Doing now to avoid ingesting too much data initially
                if datetime.strptime(bounds.get_time_min(), '%Y-%m-%d') <= \
                   datetime.strptime(date, '%Y-%m-%d') <= \
                   datetime.strptime(bounds.get_time_max(), '%Y-%m-%d'):
                    data_array.append(retrieve_data(file, date))
            # Concat all valid files
            data = xr.concat(data_array,'time')
        # Otherwise need a file or folder to read from
        else:
            raise ValueError('File or folder not specified in params!')

#         # Remove unnecessary column
        # data = data.drop(labels='polar_stereographic')
        
        
        raw_df = data.to_dataframe().reset_index().dropna()
        
        # raw_df = raw_df.set_index('time').sort_index().reset_index()
        # AMSR data is in a EPSG:3412 projection and must be reprojected into
        # EPSG:4326
        # TODO Different projections per hemisphere
        in_proj = 'EPSG:3412'
        out_proj = 'EPSG:4326'
        
        
        
        logging.debug(f'- Reprojecting from {in_proj} to {out_proj}')

        reprojected_df = self.reproject(raw_df, in_proj=in_proj, out_proj=out_proj, 
                                        x_col='x', y_col='y')
        
        mask = (reprojected_df['lat']  >= bounds.get_lat_min())  & \
               (reprojected_df['lat']  <= bounds.get_lat_max())  & \
               (reprojected_df['long'] > bounds.get_long_min()) & \
               (reprojected_df['long'] <= bounds.get_long_max()) & \
               (reprojected_df['time'] >= bounds.get_time_min()) & \
               (reprojected_df['time'] <=  bounds.get_time_max())
               
        return reprojected_df.loc[mask]

    def reproject(self, data, in_proj='EPSG:4326', out_proj='EPSG:4326', 
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
            data['SIC'] = data['z']
            data['time'] = pd.to_datetime(data['time'])
            
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
            return data
        # Choose appropriate method of reprojection based on data type
        elif type(data) == type(pd.DataFrame()):
            return reproject_df(data, in_proj, out_proj, x_col, y_col)
        elif type(data) == type(xr.Dataset()):
            return reproject_xr(data, in_proj, out_proj, x_col, y_col)

    def get_datapoints(self, bounds):
        '''
        Extracts datapoints from self.data within boundary defined by 'bounds'.
        self.data can be pd.DataFrame or xr.Dataset
        
        Args:
            bounds (Boundary): Limits of lat/long/time to select data from
            
        Returns:
            data (pd.Series): Column of data values within selected region 
        '''
        def get_datapoints_from_df(data, name, bounds):
            '''
            Extracts data from a pd.DataFrame
            '''
            # Mask off any positions not within spatial bounds
            # TODO Change <= to < after regression tests pass
            mask = (data['lat']  >= bounds.get_lat_min())  & \
                   (data['lat']  <= bounds.get_lat_max())  & \
                   (data['long'] >= bounds.get_long_min()) & \
                   (data['long'] <= bounds.get_long_max())
            # Mask with time if time column exists
            if 'time' in data.columns:
                mask &= (data['time'] >= bounds.get_time_min()) & \
                        (data['time'] <= bounds.get_time_max())
            # Return column of data from within bounds
            return data.loc[mask][name]
        
        def get_datapoints_from_xr(data, name, bounds):
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
            data = data.to_dataframe().reset_index().dropna()
            # Return column of data from within bounds
            return data[name]
            
        # Choose which method to retrieve data based on input type
        if type(self.data) == type(pd.DataFrame()):
            return get_datapoints_from_df(self.data, 'SIC', bounds)
        elif type(self.data) == type(xr.Dataset()):
            return get_datapoints_from_xr(self.data, 'SIC', bounds)


    def get_data_name(self):
        '''
        Retrieve name of data column

        Returns:
            data_name (str): Name of data column
        '''
        # Store name of data column for future reference
        
        columns = self.data.columns
        # Filter out lat, long, time columns leaves us with data column name
        filtered_cols = filter(lambda col: col not in ['lat','long','time'], columns)
        data_name = list(filtered_cols)[0]
        return data_name

    def get_value(self, bounds, skipna=True):
        '''
        Retrieve aggregated value from within bounds
        
        Args:
            aggregation_type (str): Method of aggregation of datapoints within
                bounds. Can be upper or lower case. 
                Accepts 'MIN', 'MAX', 'MEAN', 'MEDIAN', 'STD'
            bounds (Boundary): Boundary object with limits of lat/long
            skipna (bool): Defines whether to propogate NaN's or not
                Default = True (ignore's NaN's)

        Returns:
            aggregate_value (float): Aggregated value within bounds following
                aggregation_type
        '''

        # Remove lat, long and time column if they exist
        dps = self.get_datapoints(bounds).dropna()
        # If no data
        if len(dps) == 0:
            return {self.data_name :np.nan}
        # Return float of aggregated value
        elif self.aggregate_type == 'MIN':
            return {self.data_name :float(dps.min(skipna=skipna))}
        elif self.aggregate_type == 'MAX':
            return {self.data_name :float(dps.max(skipna=skipna))}
        elif self.aggregate_type == 'MEAN':
            return {self.data_name :float(dps.mean(skipna=skipna))}
        elif self.aggregate_type == 'MEDIAN':
            return {self.data_name :float(dps.median(skipna=skipna))}
        elif self.aggregate_type == 'STD':
            return {self.data_name :float(dps.std(skipna=skipna))}
        # If aggregation_type not available
        else:
            raise ValueError(f'Unknown aggregation type {self.aggregate_type}')

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
            CLR = the proportion of data points within this cellbox over a given
                threshold is lower than the lowerbound
            HOM = the proportion of data points within this cellbox over a given
                threshold is higher than the upperbound
            MIN = the cellbox contains less than a minimum number of data points

            HET = the proportion of data points within this cellbox over a given
                threshold if between the upper and lower bound
        '''
        # Retrieve datapoints to analyse
        dps = self.get_datapoints(bounds)
        
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

class BSOSESeaIceDataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Set up data
        self.data = self.import_data(bounds)
        self.data_name = self.get_data_col_name() # = 'SIC'
        
    def import_data(self, bounds):
        logging.debug("Importing BSOSE Sea Ice data...")
        logging.debug(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        # Change column names
        data = data.rename({'SIarea': 'SIC',
                            'YC': 'lat',
                            'XC': 'long'})
        
        # Limit to initial boundary
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        # bound%360 because dataset is from [0:360), and bounds in [-180:180]
        data = data.sel(long=slice(bounds.get_long_min()%360,
                                   bounds.get_long_max()%360))
        data = data.sel(time=slice(bounds.get_time_min(), 
                                   bounds.get_time_max()))
        
        # Change domain of dataset from [0:360) to [-180:180)
        # NOTE: Must do this AFTER sel because otherwise KeyError
        data = data.assign_coords(long=((data.long + 180) % 360) - 180)
        if hasattr(self, 'units'):
            # Convert to percentage form if requested in params
            if self.units == 'percentage':
                data = data.assign(SIC= data['SIC'] * 100)
            elif self.units == 'fraction':
                pass # BSOSE data already in fraction form
            else:
                raise ValueError(f"Parameter 'units' not understood."\
                                  "Expected 'percentage' or 'fraction',"\
                                  "but recieved {self.units}")
        return data

class BSOSEDepthDataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Set up data
        self.data = self.import_data(bounds)
        self.data_name = self.get_data_col_name() # = 'elevation'
        
    def import_data(self, bounds):
        logging.debug("Importing BSOSE Depth data...")
        logging.debug(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        # Change column names
        data = data.rename({'Depth': 'elevation',
                            'YC': 'lat',
                            'XC': 'long'})
        # Change domain of dataset from [0:360) to [-180:180)
        data = data.assign_coords(long=((da.lon + 180) % 360) - 180)
        
        # Limit to initial boundary
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),
                                   bounds.get_long_max()))
        data = data.sel(time=slice(bounds.get_time_min(), 
                                   bounds.get_time_max()))
        
        return data

class BalticSeaIceDataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Set up data
        self.data = self.import_data(bounds)
        self.data_name = self.get_data_col_name() # = 'SIC'
        
    def import_data(self, bounds):
        logging.debug("Importing Baltic Sea Ice data...")
        logging.debug(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        # Change column names
        data = data.rename({'ice_concentration': 'SIC',
                            'lon': 'long'})
        # Limit to just SIC data
        data = data['SIC'].to_dataset()
        # Reverse order of lat as array goes from max to min
        data = data.reindex(lat=data.lat[::-1])
        # Limit to initial boundary
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),
                                   bounds.get_long_max()))
        data = data.sel(time=slice(bounds.get_time_min(), 
                                   bounds.get_time_max()))
        
        return data

class DummyScalarDataLoader(ScalarDataLoader):
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

class GEBCODataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        self.data = self.import_data(bounds)
        self.data = self.downsample()
        
        self.data_name = self.get_data_col_name() # = 'elevation'
        
    def import_data(self, bounds):
        logging.debug("Importing GEBCO data...")
        logging.debug(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        data = data.rename({'lon':'long'})
        
        # Limit to initial boundary
        data = data.sel(lat=slice(bounds.get_lat_min(),bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),bounds.get_long_max()))
        return data

class MODISDataLoader(ScalarDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Set up data
        self.data = self.import_data(bounds)
        self.data_name = self.get_data_col_name() # = 'SIC'
        
    def import_data(self, bounds):
        logging.debug("Importing MODIS Sea Ice data...")
        logging.debug(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        # Change column name
        data = data.rename({'iceArea': 'SIC'})

        # Set areas obscured by cloud to NaN values
        data = data.where(data.cloud != 1, drop=True)
        
        # Limit to initial boundary
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),
                                   bounds.get_long_max()))
        data = data.sel(time=slice(bounds.get_time_min(), 
                                   bounds.get_time_max()))
        
        return data

# TODO Convert these to LUT dataloaders
class DensityDataLoader(ScalarDataLoader):
    
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        self.data = self.import_data(bounds)
        self.data_name = self.get_data_col_name()
                    
    def import_data(self, bounds):
        '''
        Placeholder until lookup-table dataloader class implemented
        '''
        thicknesses = {
            'Ross':          {'winter': 0.72, 'spring': 0.67, 'summer': 1.32, 
                              'autumn': 0.82, 'year': 1.07},
            'Bellinghausen': {'winter': 0.65, 'spring': 0.79, 'summer': 2.14, 
                              'autumn': 0.79, 'year': 0.90},
            'Weddell E':     {'winter': 0.54, 'spring': 0.89, 'summer': 0.87, 
                              'autumn': 0.44, 'year': 0.73},
            'Weddell W':     {'winter': 1.33, 'spring': 1.33, 'summer': 1.20, 
                              'autumn': 1.38, 'year': 1.33},
            'Indian':        {'winter': 0.59, 'spring': 0.78, 'summer': 1.05, 
                              'autumn': 0.45, 'year': 0.68},
            'West Pacific':  {'winter': 0.72, 'spring': 0.68, 'summer': 1.17, 
                              'autumn': 0.75, 'year': 0.79},
            'None':          {'winter': 0.72, 'spring': 0.67, 'summer': 1.32, 
                              'autumn': 0.82, 'year': 1.07}}
        seasons = {
            12: 'summer', 1:  'summer', 2:  'summer', 
            3:  'autumn', 4:  'autumn', 5:  'autumn', 
            6:  'winter', 7:  'winter', 8:  'winter', 
            9:  'spring', 10: 'spring', 11: 'spring'}
        
        def get_thickness(d, lon):
            month = d.month
            season = seasons[month]
            
            if   -130 <= lon <  -60: sea = 'Bellinghausen'
            elif  -60 <= lon <  -45: sea = 'Weddell W'
            elif  -45 <= lon <   20: sea = 'Weddell E'
            elif   20 <= lon <   90: sea = 'Indian'
            elif   90 <= lon <  160: sea = 'West Pacific'
            elif  160 <= lon <  180: sea = 'Ross'
            elif -180 <= lon < -130: sea = 'Ross'
            else: sea = 'None'
            
            return thicknesses[sea][season]
        
        
        lats = [lat for lat in np.arange(bounds.get_lat_min(), 
                                         bounds.get_lat_max(), 0.05)]
        lons = [lon for lon in np.arange(bounds.get_long_min(), 
                                         bounds.get_long_max(), 0.05)]
        
        start_date = datetime.strptime(bounds.get_time_min(), "%Y-%m-%d")
        end_date = datetime.strptime(bounds.get_time_max(), "%Y-%m-%d")
        delta = end_date - start_date
        dates = [start_date + timedelta(days=i) for i in range(delta.days+1)]
        
        thickness_data = xr.DataArray(
            data=[[[get_thickness(d, lon) for lon in lons] for _ in lats] for d in dates],
            coords=dict(
                lat=lats,
                long=lons,
                time=[d.strftime('%Y-%m-%d') for d in dates]),
            dims=('time','lat','long'),
            name='thickness')

        return thickness_data.to_dataframe().reset_index().set_index(['lat', 'long', 'time']).reset_index()

class ThicknessDataLoader(ScalarDataLoader):
    
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        self.data = self.import_data(bounds)
        self.data_name = self.get_data_col_name()
        
    def import_data(self, bounds):
        '''
        Placeholder until lookup-table dataloader class implemented
        '''
        densities = {'summer': 875.0, 
                     'autumn': 900.0, 
                     'winter': 920.0,
                     'spring': 900.0}

        seasons = {
            12: 'summer', 1:  'summer', 2:  'summer', 
            3:  'autumn', 4:  'autumn', 5:  'autumn', 
            6:  'winter', 7:  'winter', 8:  'winter', 
            9:  'spring', 10: 'spring', 11: 'spring'}
        
        def get_density(d):
            month = d.month
            season = seasons[month]
            return densities[season]
        
        
        lats = [lat for lat in np.arange(bounds.get_lat_min(), 
                                         bounds.get_lat_max(), 0.05)]
        lons = [lon for lon in np.arange(bounds.get_long_min(), 
                                         bounds.get_long_max(), 0.05)]

        start_date = datetime.strptime(bounds.get_time_min(), "%Y-%m-%d")
        end_date = datetime.strptime(bounds.get_time_max(), "%Y-%m-%d")
        delta = end_date - start_date
        dates = [start_date + timedelta(days=i) for i in range(delta.days+1)]
        
        density_data = xr.DataArray(
            data=[[[get_density(d) for _ in lons] for _ in lats] for d in dates],
            coords=dict(
                lat=lats,
                long=lons,
                time=[d.strftime('%Y-%m-%d') for d in dates]),
            dims=('time','lat','long'),
            name='density')
        
        return density_data.to_dataframe().reset_index().set_index(['lat', 'long', 'time']).reset_index()
    

# ---------- VECTOR DATA LOADERS ---------- #
class BalticCurrentsDataLoader(VectorDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Set up data
        self.data = self.import_data(bounds)
        self.data_name = self.col_names_to_str() # = 'SIC'
        
    def import_data(self, bounds):
        logging.debug("Importing Baltic Current data...")
        logging.debug(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        # Change column names
        data = data.rename({'latitude': 'lat',
                            'longitude': 'long',
                            'uo': 'uC',
                            'vo': 'vC'})

        # Limit to initial boundary
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),
                                   bounds.get_long_max()))
        data = data.sel(time=slice(bounds.get_time_min(), 
                                   bounds.get_time_max()))
        
        return data

class DummyVectorDataLoader(VectorDataLoader):
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

class ERA5WindDataLoader(VectorDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Set up data
        self.data = self.import_data(bounds)
        self.data_name = self.col_names_to_str() # = 'SIC'
        
    def import_data(self, bounds):
        logging.debug("Importing ERA5 Wind data...")
        logging.debug(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        # Change column names
        data = data.rename({'latitude': 'lat',
                            'longitude': 'long'})

        # TODO Ask if we need this line or not? Seems really weird
        data = data.assign(time= data['time'] + pd.Timedelta(days=365*2))

        # Set min time to start of month to ensure we include data as only have
        # monthly cadence. Assuming time is in str format
        time_min = datetime.strptime(bounds.get_time_min(), '%Y-%m-%d')
        time_min = datetime.strftime(time_min, '%Y-%m-01')

        # Reverse order of lat as array goes from max to min
        data = data.reindex(lat=data.lat[::-1])

        # Limit to initial boundary
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),
                                   bounds.get_long_max()))
        data = data.sel(time=slice(time_min, 
                                   bounds.get_time_max()))
        
        return data

class NorthSeaCurrentsDataLoader(VectorDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Set up data
        self.data = self.import_data(bounds)
        self.data_name = self.col_names_to_str() # = 'SIC'
        
    def import_data(self, bounds):
        logging.debug("Importing North Sea Current data...")
        logging.debug(f"- Opening file {self.file}")
        # Open Dataset
        data = xr.open_dataset(self.file)
        # Change column names
        data = data.rename({'lon': 'long',
                            'times': 'time',
                            'U': 'uC',
                            'V': 'vC'})
        # Limit to just these coords and variables
        data = data[['uC','vC']]
        
        # data = data.assign_coords(time=data.times.)
        
        # Limit to initial boundary
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),
                                   bounds.get_long_max()))
        data = data.sel(time=slice(bounds.get_time_min(), 
                                   bounds.get_time_max()))
        
        return data

#TODO Read in 2 files, combine to one object
class ORAS5CurrentDataLoader(VectorDataLoader):
    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        # Set up data
        self.data = self.import_data(bounds)
        self.data_name = self.col_names_to_str() # = 'SIC'
        
    def import_data(self, bounds, binsize=0.25):
        logging.debug("Importing ORAS5 Current data...")
        
        # Open Dataset
        logging.debug(f"- Opening zonal velocity file {self.file_u}")
        data_u = xr.open_dataset(self.file_u)
        
        
        logging.debug(f"- Opening meridional velocity file {self.file_v}")
        data_v = xr.open_dataset(self.data_v)
        
        # Ensure time frame of both zonal/meridional components match
        assert (data_u.time_counter.values == data_v.time_counter.values), \
            'Timestamp of input files do not match!'
            
        # Set domain of new coordinates to bin within
        lat_min = np.floor(min(data_u.nav_lat.min(), data_v.nav_lat.min()))
        lat_max = np.ceil(max(data_u.nav_lat.max(), data_v.nav_lat.max()))
        lat_range = np.arange(lat_min, lat_max, binsize)
        lon_min = np.floor(min(data_u.nav_lon.min(), data_v.nav_lon.min()))
        lon_max = np.ceil(max(data_u.nav_lon.max(), data_v.nav_lon.max()))
        lon_range = np.arange(lon_min, lon_max, binsize)
        time = data_u.time_counter.values
        
        
        
        # TODO
        

        data = xr.open_dataset(self.file)
        
        # Change column names
        data = data.rename({'nav_lon': 'long',
                            'nav_lat': 'lat',
                            'uo': 'uC',
                            'vo': 'vC'})
        # Limit to just these coords and variables
        data = data[['lat','long','uC','vC']]
        
        # Limit to initial boundary
        data = data.sel(lat=slice(bounds.get_lat_min(),
                                  bounds.get_lat_max()))
        data = data.sel(long=slice(bounds.get_long_min(),
                                   bounds.get_long_max()))
        data = data.sel(time=slice(bounds.get_time_max(), 
                                   bounds.get_time_max()))
        
        return data

class SOSEDataLoader(VectorDataLoader):

    def __init__(self, bounds, params):
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
        
        self.data = self.import_data(bounds)
        self.data_name = self.col_names_to_str()
        
    def import_data(self, bounds):
        '''
        Load SOSE netCDF
        '''
        logging.debug("Importing SOSE data...")
        # Import raw data

        # Open dataset and cast to pandas df
        logging.debug(f"- Opening file {self.file}")
        data = xr.open_dataset(self.file)

        df = data.to_dataframe().reset_index()
        
        # Change long coordinate to be in [-180,180) domain rather than [0,360)
        df['long'] = df['lon'].apply(lambda x: x-360 if x>180 else x)
        # Extract relevant columns
        df = df[['lat','long','uC','vC']]
        # Limit to  values between lat/long boundaries
        df = df[df['long'].between(bounds.get_long_min(), bounds.get_long_max())]
        df = df[df['lat'].between(bounds.get_lat_min(), bounds.get_lat_max())]

        return df


if __name__ == '__main__':

    def polygon_str_to_boundaries(polygon):
        "POLYGON ((-61.25 -65, -61.25 -64.375, -60 -64.375, -60 -65, -61.25 -65))"
        new_str = polygon.replace('POLYGON ((', '').replace('))','')
        coords = new_str.split(',')
        c1 = coords[0].lstrip().split(' ')
        c2 = coords[2].lstrip().split(' ')
        
        long_min = float(c1[0])
        lat_min = float(c1[1])
        
        long_max = float(c2[0])
        lat_max = float(c2[1])
        
        return [lat_min, lat_max], [long_min, long_max]
    
    def boundary_str_to_boundaries(bounds):
        "POLYGON ((-61.25 -65, -61.25 -64.375, -60 -64.375, -60 -65, -61.25 -65))"
        new_str = bounds.replace('[', '').replace(']','').replace(' ','')
        coords = new_str.split(',')
        long_min = float(coords[0])
        lat_min = float(coords[1])
        
        long_max = float(coords[4])
        lat_max = float(coords[5])
        
        return [lat_min, lat_max], [long_min, long_max]    

    # lat_range, long_range = polygon_str_to_boundaries(
    #     "POLYGON ((-55 -61.953125, -55 -61.875, -54.84375 -61.875, -54.84375 -61.953125, -55 -61.953125))"
    #     )
    
    lat_range = [-65, -60]
    long_range = [-70, -50]
    
    factory = DataLoaderFactory()
    bounds = Boundary(lat_range, long_range, ['2013-03-01','2013-03-14'])
    bad_lat_range, bad_long_range = polygon_str_to_boundaries(
        # "POLYGON ((-59.375 -60.703125, -59.375 -60.625, -59.21875 -60.625, -59.21875 -60.703125, -59.375 -60.703125))"          # Total
        "POLYGON ((-61.5625 -63.828125, -61.5625 -63.75, -61.40625 -63.75, -61.40625 -63.828125, -61.5625 -63.828125))"
    )
    bad_cb_bounds = Boundary(bad_lat_range, bad_long_range, ['2013-03-01','2013-03-14'])
    

    
    # ............... SCALAR DATA LOADERS ............... #
    
    if False: # Run GEBCO
        params = {
            'file': '/home/habbot/Documents/Work/PolarRoute/datastore/bathymetry/GEBCO/gebco_2022_n-40.0_s-90.0_w-140.0_e0.0.nc',
            'downsample_factors': (5,5),
            'data_name': 'elevation',
            'aggregate_type': 'MAX'
        }

        split_conds = {
            'threshold': 620,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }

        gebco = factory.get_dataloader('GEBCO', bounds, params, min_dp = 5)

        print(gebco.get_value(bounds))
        print(gebco.get_hom_condition(bounds, split_conds))

    if False: # Run AMSR
        params = {
            'folder': '/home/habbot/Documents/Work/PolarRoute/datastore/sic/amsr_south/',
            # 'file': 'PolarRoute/datastore/sic/amsr_south/asi-AMSR2-s6250-20201110-v5.4.nc',
            'data_name': 'SIC',
            'aggregate_type': 'MEAN',
            'hemisphere': 'South'
        }

        split_conds = {
            'threshold': 35,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }

        amsr = factory.get_dataloader('AMSR_folder', bounds, params, min_dp = 5)

        print(amsr.get_value(bounds))
        print(amsr.get_hom_condition(bounds, split_conds))

    if False: # Run BSOSE Sea Ice
        params = {
            'file': '/home/habbot/Documents/Work/PolarRoute/datastore/sic/bsose/bsose_i122_2013to2017_1day_SeaIceArea.nc',
            'data_name': 'SIC',
            'aggregate_type': 'MEAN'
        }

        split_conds = {
            'threshold': 0.35,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }

        bsose_sic = factory.get_dataloader('bsose_sic', bounds, params, min_dp = 5)

        print(bsose_sic.get_value(bounds))
        print(bsose_sic.get_hom_condition(bounds, split_conds))

    if False: # Run BSOSE Depth         - NEED DATA TO TEST
        pass

    if False: # Run Baltic Sea Ice
        params = {
            'file': '/home/habbot/Documents/Work/PolarRoute/datastore/sic/baltic/BalticIceMar05.nc',
            'data_name': 'SIC',
            'aggregate_type': 'MEAN'
        }

        split_conds = {
            'threshold': 0.35,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }

        baltic_sic = factory.get_dataloader('baltic_sic', bounds, params, min_dp = 5)

        print(baltic_sic.get_value(bounds))
        print(baltic_sic.get_hom_condition(bounds, split_conds))

    if False: # Run MODIS               - NEED DATA TO TEST
        pass

    if False: # Run Dummy Scalar        - NEED DATA TO TEST
        pass

    # ............... VECTOR DATA LOADERS ............... #

    if False: # Run SOSE
        params = {
            'file': '/home/habbot/Documents/Work/PolarRoute/datastore/currents/sose_currents/SOSE_surface_velocity_6yearMean_2005-2010.nc',
            'aggregate_type': 'MEAN'
        }

        sose = factory.get_dataloader('SOSE', bounds, params, min_dp = 5)

        print(sose.get_value(bounds))

    if False: # Run Baltic Currents     - NEED DATA TO TEST
        pass

    if False: # Run ERA5 Wind
        params = {
            'file': '/home/habbot/Documents/Work/PolarRoute/datastore/wind/era5_wind_2013.nc',
            'aggregate_type': 'MEAN'
        }

        era5 = factory.get_dataloader('era5_wind', bounds, params, min_dp = 5)

        print(era5.get_value(bounds))

    if False: # Run North Sea Currents
        params = {
            'file': '/home/habbot/Documents/Work/PolarRoute/datastore/currents/north_atlantic/CS3_POLCOMS2006_11.nc',
            'aggregate_type': 'MEAN'
        }

        northsea_currents = factory.get_dataloader('northsea_currents', bounds, params, min_dp = 5)

        print(northsea_currents.get_value(bounds))

    if False: # Run ORAS5 Currents
        params = {
            'file': '/home/habbot/Documents/Work/PolarRoute/datastore/currents/oras5/oras5_2019.nc',
            'aggregate_type': 'MEAN'
        }

        oras5 = factory.get_dataloader('oras5_currents', bounds, params, min_dp = 5)

        print(oras5.get_value(bounds))


    if False: # Run Dummy Vector
        pass

    # ............... LOOKUP TABLE DATA LOADERS ............... #

    if False: # Run Thickness
        params = {
            'data_name': 'thickness',
        }
  
        split_conds = {
            'threshold': 0.5,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }
        
        thickness = factory.get_dataloader('thickness', bounds, params, min_dp = 1)

        print(thickness.get_value(bounds))
        print(thickness.get_hom_condition(bounds, split_conds))

    if False: # Run Density
        params = {
            'data_name': 'density',
        }
  
        split_conds = {
            'threshold': 900,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }
        
        density = factory.get_dataloader('density', bounds, params, min_dp = 1)

        print(density.get_value(bounds))
        print(density.get_hom_condition(bounds, split_conds))

    # ............... ABSTRACT SHAPE DATA LOADERS ............... #

    if True: # Run Circle
        params = {
            "data_name": "dummy_data",
            "value_fill_types": "parent",
            "nx": 201,
            "ny": 201,
            "radius": 2,
            "centre": [-62.5, -60],
        }

        split_conds = {
            'threshold': 0.5,
            'upper_bound': 0.99,
            'lower_bound': 0.01
        }

        circle = factory.get_dataloader('circle', bounds, params, min_dp = 5)

    if False: # Run Gradient
        params = {
            'n': 11,
            'vertical': False
        }
        split_conds = {
            'threshold': 0.5,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }
        gradient = factory.get_dataloader('gradient', bounds, params, min_dp = 1)
        
        print(gradient.get_value(bounds))
        print(gradient.get_hom_condition(bounds, split_conds))
    
    if True: # Run Checkerboard
        params = {
            'nx': 201,
            'ny': 201,
            'gridsize': (6,3)
        }
        split_conds = {
            'threshold': 0.5,
            'upper_bound': 0.85,
            'lower_bound': 0.15
        }
        checkerboard = factory.get_dataloader('checkerboard', bounds, params, min_dp = 5)
        
        print(checkerboard.get_value(bounds))
        print(checkerboard.get_hom_condition(bounds, split_conds))
            
    print('hi')