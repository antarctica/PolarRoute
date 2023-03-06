from polar_route.dataloaders.scalar.abstractScalar import ScalarDataLoader
from datetime import datetime

import logging
import glob

# TODO Remove these after accepting reg test changes
from pyproj import Transformer, CRS
import xarray as xr
import pandas as pd
import numpy as np

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
        '''
        Initialises AMSR dataset. Reprojects data.
        
       Args:
            bounds (Boundary): 
                Initial boundary to limit the dataset to
            params (dict):
                Dictionary of {key: value} pairs. Keys are attributes 
                this dataloader requires to function
        '''
        logging.info("Initalising AMSR dataloader")
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
        Reads in data from a AMSR NetCDF file, or folder of files. 
        Renames coordinates to 'lat' and 'long', and renames variable to 
        'elevation'
        
        Args:
            bounds (Boundary): Initial boundary to limit the dataset to
            
        Returns:
            xr.Dataset: 
                BSOSE Depth dataset within limits of bounds. 
                Dataset has coordinates 'lat', 'long', and variable 'elevation'
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

        # Remove unnecessary column
        # Below line was causing regression tests to fail!
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
        logging.info('- Limiting to initial bounds')
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
            return_coords (boolean): 
                Flag to determine if coordinates are provided for each 
                datapoint found. Default is False.
                            
            
        Returns:
            pd.DataFrame: 
                Column of data values within selected region. If return_coords
                is true, also returns with coordinate columns 
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
        Retrieve name of data column (for pd.DataFrame), or variable 
        (for xr.Dataset). Used for when data_name not defined in params.

        Returns:
            str: 
                Name of data column
            
        Raises:
            ValueError: 
                If multiple possible data columns found, can't retrieve data 
                name
        '''
        # Store name of data column for future reference
        
        columns = self.data.columns
        # Filter out lat, long, time columns leaves us with data column name
        filtered_cols = filter(lambda col: col not in ['lat','long','time'], columns)
        data_name = list(filtered_cols)[0]
        return data_name

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
            float: 
                Aggregated value within bounds following aggregation_type
                
        Raises:
            ValueError: aggregation type not in list of available methods
        '''
        # Set to params if no specific aggregate type specified
        if agg_type is None:
            agg_type = self.aggregate_type
        # Remove lat, long and time column if they exist
        dps = self.get_datapoints(bounds).dropna()
        # If no data
        if len(dps) == 0:
            return {self.data_name :np.nan}
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
        Retrieves homogeneity condition of data within
        boundary.
         
        Args: 
            bounds (Boundary): Boundary object with limits of datarange to analyse
            splitting_conds (dict): Containing the following keys: \n
                'threshold':  
                    `(float)` The threshold at which data points of 
                    type 'value' within this CellBox are checked to be either 
                    above or below
                'upper_bound': 
                    `(float)` The lowerbound of acceptable percentage 
                    of data_points of type value within this boundary that are 
                    above 'threshold'
                'lower_bound': 
                    `(float)` The upperbound of acceptable percentage 
                    of data_points of type value within this boundary that are 
                    above 'threshold'

        Returns:
            str:
                The homogeniety condtion returned is of the form: \n
                'CLR' = the proportion of data points within this cellbox over a 
                given threshold is lower than the lowerbound \n
                'HOM' = the proportion of data points within this cellbox over a
                given threshold is higher than the upperbound \n
                'MIN' = the cellbox contains less than a minimum number of 
                data points \n
                'HET' = the proportion of data points within this cellbox over a
                given threshold if between the upper and lower bound
                
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




# # class AMSRDataLoader(ScalarDataLoader):
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
