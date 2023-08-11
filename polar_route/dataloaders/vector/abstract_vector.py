from polar_route.dataloaders.dataloader_interface import DataLoaderInterface
from abc import abstractmethod

from pyproj import Transformer, CRS

import logging
import numpy as np
import xarray as xr
import pandas as pd

from polar_route.utils import round_to_sigfig


class VectorDataLoader(DataLoaderInterface):
    '''
    Abstract class for all vector Datasets.
    '''
    def __init__(self, bounds, params):
        '''
        This is where large-scale operations are performed, 
        such as importing data, downsampling, reprojecting, and renaming 
        variables
        
        Args:
            bounds (Boundary): 
                Initial mesh boundary to limit scope of data ingest
            params (dict): 
                Values needed by dataloader to initialise. Unique to each
                dataloader

        Attributes:
            self.data (pd.DataFrame or xr.Dataset): 
                Data stored by dataloader to use when called upon by the mesh.
                Must be saved in mercator projection (EPSG:4326), with 
                coordinates names 'lat', 'long', and 'time' (if applicable).
            self.data_name (str): 
                Name of scalar variable. Must be the column name if self.data
                is pd.DataFrame. Must be variable if self.data is xr.Dataset
        '''
        # Translates parameters from config input to desired inputs
        params = self.add_default_params(params)
        logging.info(f"Initialising {params['dataloader_name']} dataloader")
        # Creates a class attribute for all keys in params
        for key, val in params.items():
            setattr(self, key, val)
            
        self.data = self.import_data(bounds)
        # Read in and manipulate data to standard form
        if 'files' in params:
            logging.info('\tFiles read:')
            for file in self.files:
                logging.info(f'\t\t{file}')
        # If need to downsample data
        self.data = self.downsample()
        # If need to reproject data
        if self.in_proj != self.out_proj:
            self.data = self.reproject(
                                in_proj  = self.in_proj,
                                out_proj = self.out_proj,
                                x_col    = self.x_col,
                                y_col    = self.y_col
                                )

        # Get data name from column name if not set in params
        if self.data_name is None:
            logging.debug('\tSetting self.data_name from column name')
            self.data_name = self.get_data_col_name()
        # or if set in params, set col name to data name
        else:
            logging.debug(f'\tSetting data column name to {self.data_name}')
            self.data = self.set_data_col_name(self.data_name.split(','))
        # Store data names in a list for easier access in future
        self.data_name_list = self.data_name.split(',')
        
        # Add magnitude and direction to dataset
        self.data = self.add_mag_dir()
        
        # Calculate fraction of boundary that data covers
        data_coverage = self.calculate_coverage(bounds)
        logging.info("\tMercator data range (roughly) covers "+\
                    f"{np.round(data_coverage*100,0).astype(int)}% "+\
                     "of initial boundary")
        # If there's 0 datapoints in the initial boundary, raise ValueError
        if data_coverage == 0:
            logging.error('\tDataloader has no data in initial region!')
            raise ValueError(f"Dataloader {params['dataloader_name']}"+\
                              " contains no data within initial region!")
        else:
            # Cut dataset down to initial boundary
            logging.info(
                "\tTrimming data to initial boundary: {min} to {max}".format(
                    min=(bounds.get_lat_min(), bounds.get_long_min()),
                    max=(bounds.get_lat_max(), bounds.get_long_max())
                ))
            
            self.data = self.trim_datapoints(bounds)

    @abstractmethod
    def import_data(self, bounds):
        '''
        User defined method for importing data from files, or even generating 
        data from scratch
                
        Returns:
            xr.Dataset or pd.DataFrame:
                Coordinates and data being imported from file \n
                if xr.Dataset, 
                    - Must have coordinates 'lat' and 'long'
                    - Should have multiple data variables
                    
                if pd.DataFrame, 
                    - Must have columns 'lat' and 'long'
                    - Should have multiple data columns
                    
                Downsampling and reprojecting happen in __init__() method
        '''
        pass
    
    def add_default_params(self, params):
        '''
        Set default values for all scalar dataloaders. This function should be
        overloaded to include any extra params for a specific dataloader
        
        Args:
            name (str):
                Name of dataloader entry in dataloader_requirements. Used to
                specify default parameters for a specific dataloader.
            params (dict): 
                Dictionary containing attributes that are required for each 
                dataloader. 
            
        Returns:
            (dict): 
                Dictionary of attributes the dataloader will require, 
                completed with default values if not provided in config.
        '''
        # This should only pop up if using dataloader not in the factory
        if 'dataloader_name' not in params:
            params['dataloader_name'] = self.__class__.__name__
        
        if 'data_name' not in params:
            params['data_name'] = None

        if 'downsample_factors' not in params:
            params['downsample_factors'] = [1,1]

        if 'aggregate_type' not in params: 
            params['aggregate_type']  = 'MEAN'
            
        if 'min_dp' not in params:
            params['min_dp'] = 5
            
        if 'in_proj' not in params:
            params['in_proj'] = 'EPSG:4326'
            
        if 'out_proj' not in params:
            params['out_proj'] = 'EPSG:4326'
            
        if 'x_col' not in params:
            params['x_col'] = 'lat'

        if 'y_col' not in params:
            params['y_col'] = 'long'
            
        return params
    
    def add_mag_dir(self, data=None, data_names=None):
        '''
        Adds magnitude and direction variables/columns to data for easier
        retrieval of value 
        
        Args:
            data (pd.DataFrame or xr.Dataset):
                Data with 'lat' and 'long' columns/dimensions. Assumes that the
                existing data is in cartesian form (x and y components). 
                If None, will use self.data
            data_names (list):
                List of data columns/variables to use in calculation
                If None, will use self.data_name_list
                
        Returns:
            data (pd.DataFrame or xr.Dataset):
                Original dataset with two new columns/variables called 
                '_magnitude' and '_direction', containing the corresponding
                values for each.
        '''
        
        def add_mag_dir_to_df(data, names):
            '''
            Adds magnitude and direction columns to pd.DataFrame
            
            Args:
                Same as parent method
                
            Returns:
                Same as parent method
            '''
            x, y = names
            data['_magnitude'] = np.linalg.norm([data[x], data[y]], axis=0)
            data['_direction'] = np.arctan(data[y] / data[x])
            return data
        
        def add_mag_dir_to_xr(data, names):
            '''
            Adds magnitude and direction columns to xr.Dataset
            
            Args:
                Same as parent method
                
            Returns:
                Same as parent method
            '''
            x, y = names
            data['_magnitude'] = (data.dims, 
                                  np.linalg.norm([data[x].data, data[y].data], 
                                                 axis=0))
            data['_direction'] = (data.dims, 
                                  np.arctan(data[y].data / data[x].data))
            return data
        
        # Set defaults if not passed to method
        if data is None:        data = self.data
        if data_names is None:  names = self.data_name_list
        
        # Perform operation on appropriate datatype
        if type(data) == pd.core.frame.DataFrame:
            return add_mag_dir_to_df(data, names)
        elif type(data) == xr.core.dataset.Dataset:
            return add_mag_dir_to_xr(data, names)

    def calculate_coverage(self, bounds, data=None):
        """
        Calculates percentage of boundary covered by dataset

        Args:
            bounds (Boundary): 
                Boundary being compared against
            data (pd.DataFrame or xr.Dataset): 
                Dataset with 'lat' and 'long' coordinates. 
                Extent calculated from min/max of these coordinates. 
                Defaults to objects internal dataset.
        
        Returns:
            float:
                Decimal fraction of boundary covered by the dataset
        """
        def calculate_coverage_from_df(bounds, data):
            data = data.dropna().reset_index()
            # If empty dataframe, 0% coverage
            if data.empty:
                return 0
            # If no valid coordinates within data range, 0% coverage
            elif data.lat.size == 0 or data.long.size == 0:
                return 0
            # Otherwise, calculate coverage, assuming rectangular region 
            # in mercator projection
            else:
                # Get range of latitude values
                data_lat_range = data.lat.max() - data.lat.min()
                bounds_lat_range = bounds.get_lat_max() - bounds.get_lat_min()
                # Get range of longitude values
                data_long_range = data.long.max() - data.long.min()
                bounds_long_range = bounds.get_long_max() - bounds.get_long_min()
                # Calcualte area of each region
                data_area = data_lat_range * data_long_range
                bounds_area = bounds_lat_range * bounds_long_range
                # If data area completely covers bounds, 100% coverage
                if data_area >= bounds_area:
                    return 1
                # Otherwise return decimal fraction
                else:
                    return data_area / bounds_area
                
                
        def calculate_coverage_from_xr(bounds, data):
            # Remove all NaN columns/rows
            data = data.dropna(dim="lat", how="all")
            data = data.dropna(dim="long", how="all")
            
            # If no valid coordinates within data range, 0% coverage
            if data.lat.size == 0 or data.long.size == 0:
                return 0
            # Otherwise, calculate coverage, assuming rectangular region 
            # in mercator projection
            else:
                # Get range of latitude values
                data_lat_range = data.lat.max().item() - data.lat.min().item()
                bounds_lat_range = bounds.get_lat_max() - bounds.get_lat_min()
                # Get range of longitude values
                data_long_range = data.long.max().item() - data.long.min().item()
                bounds_long_range = bounds.get_long_max() - bounds.get_long_min()
                # Calcualte area of each region
                data_area = data_lat_range * data_long_range
                bounds_area = bounds_lat_range * bounds_long_range
                # If data area completely covers bounds, 100% coverage
                if data_area >= bounds_area:
                    return 1
                # Otherwise return decimal fraction
                else:
                    return data_area / bounds_area
        # Use self.data if not no explicit dataset specified
        if data is None:
            data = self.data
        # Calculate data coverage fraction
        if type(self.data) == pd.core.frame.DataFrame:
            return calculate_coverage_from_df(bounds, data)
        elif type(self.data) == xr.core.dataset.Dataset:
            return calculate_coverage_from_xr(bounds, data)

    def trim_datapoints(self, bounds, data=None):
        '''
        Trims datapoints from self.data within boundary defined by 'bounds'.
        self.data can be pd.DataFrame or xr.Dataset
        
        Args:
            bounds (Boundary): Limits of lat/long/time to select data from

        Returns:
            pd.DataFrame or xr.Dataset: 
                Trimmed dataset in same format as self.data
        '''
        def trim_datapoints_from_df(data, bounds):
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
                        
            # Return column of data from within bounds
            return data.loc[mask]

        def trim_datapoints_from_xr(data, bounds):
            '''
            Extracts data from a xr.Dataset
            '''

            # Select data region within spatial bounds
            # NOTE slice in xarray is inclusive of bounds
            data = data.sel(lat=slice(bounds.get_lat_min(),  bounds.get_lat_max() ))
            data = data.sel(long=slice(bounds.get_long_min(), bounds.get_long_max()))
            # Select data region within temporal bounds if time exists as a coordinate
            if 'time' in data.coords.keys():
                data = data.sel(time=slice(bounds.get_time_min(),  bounds.get_time_max()))

            # Trim off any data on the min boundary to be consistent with df
            if bounds.get_lat_min() in data.lat:
                data = data.where(data.lat  != bounds.get_lat_min(), drop=True)
            if bounds.get_long_min() in data.long:
                data = data.where(data.long != bounds.get_long_min(), drop=True)
            
            # Return column of data from within bounds
            return data
        
        # If no specific data passed in, default to entire dataset
        if data is None:
            data = self.data
            
        # Skip trimming if data already completely within bounds
        if data.lat.min() >  bounds.get_lat_min() and \
           data.lat.max() <= bounds.get_lat_max() and \
           data.long.min() >  bounds.get_long_min() and \
           data.long.max() <= bounds.get_long_max():
            logging.debug('\tData is already trimmed to bounds!')
            return data
        
        if type(data) == pd.core.frame.DataFrame:
            return trim_datapoints_from_df(data, bounds)
        elif type(data) == xr.core.dataset.Dataset:
            return trim_datapoints_from_xr(data, bounds)

    def get_dp_from_coord(self, long=None, lat=None, return_coords=False):
        '''
        Extracts datapoint from self.data with lat and long specified in kwargs.
        self.data can be pd.DataFrame or xr.Dataset. Will return multiple values
        if one set of coordinates have multiple entries (e.g. time series data)
        
        Args:
            long (float): Longitude coordinate to search for
            lat (float) : Latitude coordinate to search for
            
        Returns:
            pd.Dataframe:  
                Column of data values with chosen lat/long. Could be many 
                datapoints because either bad data or multiple time steps 
        '''
        def get_dp_from_coord_df(data, names, long, lat, return_coords):
            '''
            Extracts data from a pd.DataFrame
            '''
            # Mask off any positions not within spatial bounds
            mask = (data['lat']  == lat)  & \
                   (data['long'] == long) 

            # Include lat/long/time if requested
            if return_coords: columns = list(data.columns)
            else:             columns = names
            # Return column of data from within bounds
            return data.loc[mask][columns]
        
        def get_dp_from_coord_xr(data, names, long, lat, return_coords):
            '''
            Extracts data from a xr.Dataset
            '''
            # Select data region within spatial bounds
            data = data.sel(lat=lat, long=long)
            # Cast as a pd.DataFrame
            data = data.to_dataframe().reset_index()
            # Include lat/long/time if requested
            if return_coords: columns = list(data.columns)
            else:             columns = names
            # Return column of data from within bounds
            return data[columns]
        
        # Ensure that lat and long provided
        assert (lat is not None) and (long) is not None, \
            'Must provide lat and long to this method!'
            
        # Choose which method to retrieve data based on input type
        if type(self.data) == pd.core.frame.DataFrame:
            return get_dp_from_coord_df(self.data, self.data_name_list, long, lat, return_coords)
        elif type(self.data) == xr.core.dataset.Dataset:
            return get_dp_from_coord_xr(self.data, self.data_name_list, long, lat, return_coords)
    
    def get_value(self, bounds, agg_type=None, skipna=True):
        '''
        Retrieve aggregated value from within bounds
        
        Args:
            aggregation_type (str): Method of aggregation of datapoints within
                bounds. Can be upper or lower case. 
                Accepts 'MIN', 'MAX', 'MEAN', 'MEDIAN', 'STD', 'COUNT'
            bounds (Boundary): Boundary object with limits of lat/long
            skipna (bool): Defines whether to propogate NaN's or not
                Default = True (ignore's NaN's)

        Returns:
            dict: 
                {variable (str): aggregated_value (float)}
                Aggregated value within bounds following aggregation_type
                
        Raises:
            ValueError: aggregation type not in list of available methods
        '''
        def get_value_from_df(dps, variable_names, bounds, agg_type, skipna):
            '''
            Aggregates a value from a pd.Series.
            
            Args:
                dps (pd.Series): Datapoints within boundary
                bounds (Boundary): 
                    Boundary dps was trimmed to. Not used for any calculations,
                    just the logging.debug message.
                agg_type (str):
                    Method of aggregation for the value, 
                    e.g. agg_type = 'MIN' => min(dps) returned 
                skipna (bool): 
                    Flag for whether NaN's should be included in aggregation. 
            
            Returns:
                np.float64: Aggregated value
            '''
            data_count = len(dps)
            logging.debug(f"\t{data_count} datapoints found for attribute '{self.data_name}' within bounds '{bounds}'")
            # If no data
            if data_count == 0:
                values = [np.nan, np.nan]
            # If want the number of datapoints
            elif agg_type =='COUNT':
                values = [data_count, data_count]
            elif agg_type == 'MIN':
                index = dps['_magnitude'].idxmin(skipna=skipna)
                values = [dps[name][index] for name in variable_names]
            elif agg_type == 'MAX':
                index = dps['_magnitude'].idxmax(skipna=skipna)
                values = [dps[name][index] for name in variable_names]
            elif agg_type == 'MEAN':
                values = [dps[name].mean(skipna=skipna) for name in variable_names]
            elif agg_type == 'STD':
                values = [dps[name].std(skipna=skipna) for name in variable_names]
            elif agg_type == 'MEDIAN':
                raise ValueError('Aggregation type "MEDIAN" is non-sensical for vector dataset!')
            else:
                raise ValueError(f'Unknown aggregation type {agg_type}')
            
            return values

        
        def get_value_from_xr(dps, variable_names, bounds, agg_type, skipna):
            '''
            Aggregates a value from a xr.DataArray.
            
            Args:
                dps (xr.DataArray): Datapoints within boundary
                bounds (Boundary): 
                    Boundary dps was trimmed to. Not used for any calculations,
                    just the logging.debug message.
                agg_type (str):
                    Method of aggregation for the value, 
                    e.g. agg_type = 'MIN' => min(dps) returned 
                skipna (bool): 
                    Flag for whether NaN's should be included in aggregation. 
            
            Returns:
                dict:
                    {variable_name: np.float64}: Aggregated value in a dictionary
            '''
            # Info on size of array
            data_count = dps._magnitude.size 
            logging.debug(f"\t{data_count} datapoints found for attribute '{self.data_name}' within bounds '{bounds}'")
            # If no data, return np.nan for each variable
            if data_count == 0:
                values = [np.nan, np.nan]
            # If want count
            elif agg_type == 'COUNT':
                # If including nan's, just want size
                if skipna:  
                    values = [data_count, data_count]
                # Otherwise count non-nan values
                else:       
                    values = [dps[name].count().item() for name in variable_names]
            elif agg_type == 'MIN':
                # Get 2D index of minimum magnitude point
                index = np.unravel_index(
                            dps._magnitude.argmin(skipna=skipna), 
                            dps._magnitude.shape)
                # Get cartesian vector from index
                values = [dps[name][index].item() for name in variable_names]
            elif agg_type == 'MAX':
                # Get 2D index of minimum magnitude point
                index = np.unravel_index(
                            dps._magnitude.argmax(skipna=skipna), 
                            dps._magnitude.shape)
                # Get cartesian vector from index
                values = [dps[name][index].item() for name in variable_names]
            # And the rest self explanatory
            elif agg_type == 'MEAN':
                values = [dps[name].mean(skipna=skipna).item() for name in variable_names]
            elif agg_type == 'STD':
                values = [dps[name].std(skipna=skipna).item() for name in variable_names]
            elif agg_type == 'MEDIAN':
                raise ValueError('Aggregation type "MEDIAN" is non-sensical for vector dataset!')
            else:
                raise ValueError(f'Unknown aggregation type {agg_type}')
            
            return values
    

        # Set to params if no specific aggregate type specified
        if agg_type is None:
            agg_type = self.aggregate_type
            
        # Limit data to boundary
        dps = self.trim_datapoints(bounds)
        # Get list of values
        if type(self.data) == pd.core.frame.DataFrame:
            values = get_value_from_df(dps, self.data_name_list, bounds, agg_type, skipna)
        elif type(self.data) == xr.core.dataset.Dataset:
            values = get_value_from_xr(dps, self.data_name_list, bounds, agg_type, skipna)
            
        # Put in dict to map variable to values
        return {self.data_name_list[i]: values[i] for i in range(len(self.data_name_list))}

    def get_hom_condition(self, bounds, splitting_conds, agg_type='MEAN'):
        '''
        Retrieves homogeneity condition of data within boundary. 
         
        Args: 
            bounds (Boundary): Boundary object with limits of datarange to analyse
            splitting_conds (dict): Containing the following keys: \n
                'threshold':  
                    `(float)` The threshold at which data points of 
                    type 'value' within this CellBox are checked to be either 
                    above or below

        Returns:
            str:
                The homogeniety condtion returned is of the form: \n
                'MIN' = the cellbox contains less than a minimum number of 
                data points \n
                'HET' = Threshold values defined in config are exceeded \n
                'CLR' = None of the HET conditions were triggered \n
        '''
        # Get length of dataset in bounds  
        if type(self.data) == pd.core.frame.DataFrame:
            num_dp = len(self.trim_datapoints(bounds))
        elif type(self.data) == xr.core.dataset.Dataset:
            num_dp = min(self.trim_datapoints(bounds).count().values())

        # Set default homogeneity 
        hom_type = 'CLR'

        # Check to see if it's above the minimum threshold
        if num_dp < self.min_dp:
            hom_type = 'MIN'
        else:
            # To allow multiple modes of splitting, chuck them in the splitting conditions
            # Split if magnitude of curl(data) is larger than threshold 
            if 'curl' in splitting_conds:
                curl = self.calc_curl(bounds)
                if np.abs(curl) > splitting_conds['curl']:
                    hom_type =  'HET'
            # Split if max magnitude(any_vector - ave_vector) is larger than threshold
            if 'dmag' in splitting_conds:
                dmag = self.calc_dmag(bounds)
                if np.abs(dmag) > splitting_conds['dmag']:
                    hom_type = 'HET'
                
            # Split if Reynolds number is larger than threshold
            if 'reynolds' in splitting_conds:        
                reynolds = self.calc_reynolds_number(bounds)
                if reynolds > splitting_conds['reynolds']:
                    hom_type = 'HET'

        logging.debug(f"\thom_condition for attribute: '{self.data_name}' in bounds:'{bounds}' returned '{hom_type}'")
        
        return hom_type

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
            pd.DataFrame: 
                Reprojected data with 'lat', 'long' columns 
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
            if x_col != 'lat':  data = data.drop(x_col, axis=1)
            if y_col != 'long': data = data.drop(y_col, axis=1)
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
            logging.debug("\tself.reproject() called but don't need to")
            return self.data
        else:
            logging.info(f"\tReprojecting data from {in_proj} to {out_proj}")
        # Choose appropriate method of reprojection based on data type
        if type(self.data) == pd.core.frame.DataFrame:
            return reproject_df(self.data, in_proj, out_proj, x_col, y_col)
        elif type(self.data) == xr.core.dataset.Dataset:
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
            xr.Dataset or pd.DataFrame: 
                Downsampled data
        '''
        def downsample_xr(data, ds, agg_type):
            '''
            Downsample xarray dataset according to aggregation type
            
            Args:
                data (xr.Dataset):
                    Dataset containing data to be downsampled. Must have 
                    coordinates 'lat' and 'long'
                ds (int, int):
                    Downsampling factors. 
                    ds[0] is longitude
                    ds[1] is latitude
                agg_type (str):
                    Aggregation method to use for binning. Default is same as 
                    set in config, passed in by parent
            
            Returns:
                xr.Dataset:
                    Downsampled data
            '''
            if agg_type == 'MIN':
                # Returns min of bin
                data = data.coarsen(lat=ds[1],boundary='pad').min()
                data = data.coarsen(long=ds[0],boundary='pad').min()
            elif agg_type == 'MAX':
                # Returns max of bin
                data = data.coarsen(lat=ds[1],boundary='pad').max()
                data = data.coarsen(long=ds[0],boundary='pad').max()
            elif agg_type == 'MEAN':
                # Returns mean of bin
                data = data.coarsen(lat=ds[1],boundary='pad').mean()
                data = data.coarsen(long=ds[0],boundary='pad').mean()
            elif agg_type == 'MEDIAN':
                # Returns median of bin
                data = data.coarsen(lat=ds[1],boundary='pad').median()
                data = data.coarsen(long=ds[0],boundary='pad').median()
            elif agg_type == 'STD':
                # Returns std_dev of range
                data = data.coarsen(lat=ds[1],boundary='pad').std()
                data = data.coarsen(long=ds[0],boundary='pad').std()
            elif agg_type =='COUNT': 
                # Returns every first element in bin
                data = data.thin(lat=ds[1])
                data = data.thin(long=ds[0])
            return data
    
        def downsample_df(data, ds, agg_type):
            '''
            Downsample pandas dataframe
            Not implemented as it just adds to processing time, 
            defeating the purpose
            '''
            logging.warning(
                '\tDownsampling called on pd.DataFrame! Downsampling a df' \
                'too computationally expensive, returning original df'
                )
            return data

        # Set to params if no specific aggregate type specified
        if agg_type is None:
            agg_type = self.aggregate_type
        
        # If no downsampling
        if self.downsample_factors == (1,1) or \
           self.downsample_factors == [1,1]:
            logging.debug("\tself.downsample() called but don't have to")
            return self.data
        else:
            logging.info(f"\tDownsampling data by {self.downsample_factors}")
        # Otherwise, downsample appropriately
        if type(self.data) == pd.core.frame.DataFrame:
            return downsample_df(self.data, self.downsample_factors, agg_type)
        elif type(self.data) == xr.core.dataset.Dataset:
            return downsample_xr(self.data, self.downsample_factors, agg_type)
        
    def get_data_col_name(self):
        '''
        Retrieve name of data column (for pd.DataFrame), or variable 
        (for xr.Dataset). Used for when data_name not defined in params.
        Variable names are appended and comma seperated

        Returns:
            str: 
                Name of data columns, comma seperated
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
        
        logging.debug(f"\tRetrieving data name from {type(self.data)}")
        # Choose method of extraction based on data type
        if type(self.data) == pd.core.frame.DataFrame:
            return get_data_names_from_df(self.data)
        elif type(self.data) == xr.core.dataset.Dataset:
            return get_data_names_from_xr(self.data)

    def get_data_col_name_list(self):
        '''
        Retrieve names of data columns (for pd.DataFrame), or variable 
        (for xr.Dataset). Used for when data_name not defined in params.

        Returns:
            list: 
                Contains strings of data namesk
        '''
        return self.get_data_col_name().split(',')

    def set_data_col_name(self, new_names):
        '''
        Sets name of data column/data variables from a comma-seperated string
        
        Args:
            name_dict (dict): 
                Dictionary mapping old variable names to new variable names,
                of the form {old_name (str): new_name (str)}

        Returns:
            xr.Dataset or pd.DataFrame: 
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
        # Get existing column names
        old_names = self.get_data_col_name().split(',')
        # Ensure that can do replacement of columns
        assert len(old_names) == len(new_names)
        # Set up mapping of old names to new names
        name_dict = {old_col: new_names[i] 
                     for i, old_col in enumerate(old_names)}
        # Change names
        # Change data name depending on data type
        if type(self.data) == pd.core.frame.DataFrame:
            return set_names_df(self.data, name_dict)
        elif type(self.data) == xr.core.dataset.Dataset:
            return set_names_xr(self.data, name_dict)
        
    def set_data_col_name_list(self, new_names):
        '''
        Sets name of data column/data variables from a list of strings.
        Also updates self.data_name_list with new names from list
        
        Args:
            new_names (list):
                List of strings containing new variable names
        
        Returns:
            pd.DataFrame or xr.Dataset:
                Original dataset with data variables renamed
        '''
        # Check validity of input
        assert type(new_names) == list, f"'new_names' must be a list! Instead it is a {type(new_names)}"
        assert len(new_names) == 2, f"'new_names' must have a length of 2! Instead it has length {len(new_names)}"
        str_items = [isinstance(name, str) for name in new_names]
        assert all(str_items), f"'new_names' must be list of 'str'. Currently {sum(str_items)} / 2 are strings!"
        new_data_name = ','.join(new_names)
        
        # Set names
        logging.info(f'\tSetting data names to {new_names}')
        self.data_name_list = new_names
        return self.set_data_col_name(new_data_name)

    def calc_reynolds_number(self, bounds):
        '''
        Calculates an approximate Reynolds number from the mean vector velocity
        and cellbox size.
        
        CURRENTLY ASSUMES DENSITY AND VISCOSITY OF SEAWATER AT 4°C! 
        WILL NEED MINOR REWORKING TO INCLUDE DIFFERENT FLUIDS
        
        Args:
            bounds (Boundary): 
                Cellbox boundary to calculate characteristic length from
                
        Returns:
            float:
                Reynolds number of cellbox
        '''
        # Extract the speed
        velocity = self.get_value(bounds, agg_type='MEAN')
        speed = np.linalg.norm(list(velocity.values())) # Calculates magnitude
        # Extract the characteristic length
        length = bounds.calc_size()
        # Calculate the reynolds number and return
        logging.warning("\tReynold number used for splitting, this function assumes properties of ocean water!")
        return 1028 * 0.00167 * speed * length

    def calc_divergence(self, bounds, data=None, collapse=True, agg_type='MAX'):
        '''
        Calculates the divergence of vectors in a cellbox
        
        Args:
            bounds (Boundary):
                Cellbox boundary in which all relevant vectors are contained
            data (pd.DataFrame or xr.Dataset):
                Dataset with 'lat' and 'long' columns/dimensions with vectors
            collapes (bool): 
                Flag determining whether to return an aggregated value, or a 
                vector field (values for each individual vector).
            agg_type (str):
                Method of aggregation if collapsing value. 
                Accepts 'MAX' or 'MEAN'
        
        Returns:
            float or pd.DataFrame:
                float value of aggregated div if collapse=True, or
                pd.DataFrame of div vector field if collapse=False 

        Raises:
            ValueError: If agg_type is not 'MAX' or 'MEAN'
        '''
        if data is None:    dps = self.trim_datapoints(bounds, data=data)
        else:               dps = data
        
        # Create a meshgrid of vectors from the data
        vector_field = self._create_vector_meshgrid(dps, self.data_name_list)

        # Get component values for each vector
        fx, fy = vector_field[:, :, 0], vector_field[:, :, 1]
        # If not enough datapoints to compute gradient
        if 1 in fx.shape or 1 in fy.shape:
            logging.debug('\tUnable to compute gradient across cell for divergence calculation')
            div = np.nan
        else:
            # Compute partial derivatives
            dfx_dy = np.gradient(fx, axis=1)
            dfy_dx = np.gradient(fy, axis=0)
            # Compute curl
            div = dfy_dx + dfx_dy
        
        # If div is nan
        if np.isnan(div).all():
            logging.debug('\tAll NaN cellbox encountered')
            return np.nan
        # If want to collapse to max mag value, return scalar
        elif collapse:   
            if agg_type == 'MAX':       return max(np.nanmax(div), np.nanmin(div), key=abs)
            elif agg_type == 'MEAN':    return np.nanmean(div)
            else: 
                raise ValueError(f"agg_type '{agg_type}' not understood! Requires 'MAX' or 'MEAN'")
        # Else return field
        else:
            return div


    def calc_curl(self, bounds, data=None, collapse=True, agg_type='MAX'):
        '''
        Calculates the curl of vectors in a cellbox
        
        Args:
            bounds (Boundary):
                Cellbox boundary in which all relevant vectors are contained
            data (pd.DataFrame or xr.Dataset):
                Dataset with 'lat' and 'long' columns/dimensions with vectors
            collapes (bool): 
                Flag determining whether to return an aggregated value, or a 
                vector field (values for each individual vector).
            agg_type (str):
                Method of aggregation if collapsing value. 
                Accepts 'MAX' or 'MEAN'
        
        Returns:
            float or pd.DataFrame:
                float value of aggregated curl if collapse=True, or
                pd.DataFrame of curl vector field if collapse=False
                
        Raises:
            ValueError: If agg_type is not 'MAX' or 'MEAN'
        '''
        if data is None:    dps = self.trim_datapoints(bounds, data=data)
        else:               dps = data
        # Create a meshgrid of vectors from the data
        vector_field = self._create_vector_meshgrid(dps, self.data_name_list)
        # Get component values for each vector
        fx, fy = vector_field[:, :, 0], vector_field[:, :, 1]
        # If not enough datapoints to compute gradient
        if 1 in fx.shape or 1 in fy.shape:
            logging.debug('\tUnable to compute gradient across cell for curl calculation')
            curl = np.nan
        else:
            # Compute partial derivatives
            dfx_dy = np.gradient(fx, axis=1)
            dfy_dx = np.gradient(fy, axis=0)
            # Compute curl
            curl = dfy_dx - dfx_dy

        # If div is nan
        if np.isnan(curl).all():
            logging.debug('\tAll NaN cellbox encountered')
            return np.nan
        # If want to collapse to max mag value, return scalar
        elif collapse:
            if agg_type == 'MAX': return max(np.nanmax(curl), np.nanmin(curl), key=abs)
            elif agg_type == 'MEAN':    return np.nanmean(curl)
            else: 
                raise ValueError(f"agg_type '{agg_type}' not understood! Requires 'MAX' or 'MEAN'")
        # Else return field
        else:
            return curl

    def calc_dmag(self, bounds, data=None, collapse=True, agg_type='MEAN'):
        '''
        Calculates the dmag of vectors in a cellbox.
        dmag is defined as being the difference in magnitudes between 
        each vector and the average vector within the bounds.\n
        dmag = mag(vector - mean_vector)
        
        Args:
            bounds (Boundary):
                Cellbox boundary in which all relevant vectors are contained
            data (pd.DataFrame or xr.Dataset):
                Dataset with 'lat' and 'long' columns/dimensions with vectors
            collapes (bool): 
                Flag determining whether to return an aggregated value, or a 
                vector field (values for each individual vector).
            agg_type (str):
                Method of aggregation if collapsing value. 
                Accepts 'MAX' or 'MEAN'
        
        Returns:
            float or pd.DataFrame:
                float value of aggregated dmag if collapse=True, or
                pd.DataFrame of dmag vector field if collapse=False
                
        Raises:
            ValueError: If agg_type is not 'MAX' or 'MEAN'
        '''
        if data is None:    dps = self.trim_datapoints(bounds, data=data)
        else:               dps = data
            
        data_names = self.data_name_list
        each_vector = dps[data_names].to_numpy()
        ave_vector = list(self.get_value(bounds, agg_type=agg_type).values())
        
        delta_vector = each_vector - ave_vector
        
        d_mag = np.linalg.norm(delta_vector, axis=1)
        if len(d_mag) == 0:
            logging.debug('\tEmpty cellbox encountered')
            return np.nan
        # If div is nan
        elif np.isnan(d_mag).all():
            logging.debug('\tAll NaN cellbox encountered')
            return np.nan
        # If want to collapse to max mag value, return scalar
        elif collapse:
            if agg_type == 'MAX':       return np.nanmax(d_mag)
            elif agg_type == 'MEAN':    return np.nanmean(d_mag)
            else:
                raise ValueError(f"agg_type '{agg_type}' not understood! Requires 'MAX' or 'MEAN'")
        # Else return field
        else:          return d_mag

    
    @staticmethod
    def _create_vector_meshgrid(data, data_name_list):
        '''
        Creates a np.meshgrid containing 2D vectors from a pd.DataFrame
        
        Args:
            data (pd.DataFrame): 
                Dataframe with columns 'lat', 'long', and vector x/y components
                lat | long | {v_x} | {v_y}
            data_name_list (list): 
                List of strings containing the vector component names
        
        Returns:
            np.array:
                Table containing vectors as np.arrays at each coord
        
        '''
        def meshgrid_from_df(data, data_name_list):

            # Manipulate into meshgrid of 2D vectors
            x, y = data_name_list
            # Fields of each vector component
            vector_x_field = data.pivot(index='lat', columns='long', values=x)
            vector_y_field = data.pivot(index='lat', columns='long', values=y)
            # Combine into field of vectors
            vector_field = np.stack((vector_x_field, vector_y_field), axis=-1)
            vector_field = np.swapaxes(vector_field, 0, 1)
            return vector_field
        
        def meshgrid_from_xr(data, data_name_list):
            # Extract out each variable and combine as tuple
            data_arrays = (data[name].values 
                           for name in data_name_list)
            # Zip them together to make 2D array of n-dimensional vectors
            return np.dstack(data_arrays)
        
        if type(data) == pd.core.frame.DataFrame:
            return meshgrid_from_df(data, data_name_list)
        elif type(data) == xr.core.dataset.Dataset:
            return meshgrid_from_xr(data, data_name_list)
        
        

    
    