from abc import ABC, abstractmethod
from datetime import date, timedelta

import xarray as xr
import pandas as pd
import numpy as np

import logging
import glob

from polar_route.Boundary import Boundary

class DataLoaderFactory:

    def get_dataloader(name, params, min_dp=5):
        # If file or folder passed into config
        if 'file' in params:     file_location = params['file']
        elif 'folder' in params: file_location = params['folder']
        else:                    raise ValueError('File not specified')
        
        if 'downsample_factors' in params:  ds = params['downsample_factors']
        else:                               ds = None

        if 'data_name' in params: data_name = params['data_name'] 
        else: 					  data_name = None
        
        if 'aggregate_type' in params: agg_type = params['aggregate_type']  
        else:                          agg_type = 'MEAN'


        if   name == 'GEBCO':     data_loader = GEBCODataLoader
        elif name == 'AMSR':      data_loader = AMSRDataLoader
        # elif name == 'SOSE':      data_loader = SOSEDataLoader
        elif name == 'thickness': data_loader = ThicknessDataLoader
        elif name == 'density':   data_loader = DensityDataLoader
        # elif ...
        else: raise ValueError(f'{name} not in known list of DataLoaders')

        return data_loader(file_location, min_dp=min_dp, ds=ds, data_name=data_name, aggregate_type=agg_type)


class ScalarDataLoader(ABC):
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
    def __init__(self, file_location, bounds, min_dp=5, ds=None, data_name=None, aggregate_type="MEAN"):
        self.file_location  = file_location
        self.min_dp         = min_dp
        self.ds             = ds

        # Cast string to uppercase to accept mismatched case
        self.aggregate_type = aggregate_type.upper()

        self.data = self.import_data(bounds)
        # If no data name specified, retrieve from self.data
        self.data_name = data_name if data_name else self.get_data_name()
        
        logging.debug(f'- Successfully extracted {self.data_name}')

    @abstractmethod
    def import_data(self, bounds):
        '''
        Import data from whatever format this datasource is in and
        reproject it to EPSG:4326 (our working projection)
        '''
        pass

    @abstractmethod
    def get_datapoints(self, bounds):
        '''
        Retrieve datapoints within bounds, unique per implementation of 
        self.import_data
        '''
        pass
    
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
        dps = self.get_datapoints(bounds)
        # If no data
        if len(dps) == 0:
            return None
        # Return float of aggregated value
        elif self.aggregate_type == 'MIN':
            return float(dps.min(skipna=skipna))
        elif self.aggregate_type == 'MAX':
            return float(dps.max(skipna=skipna))
        elif self.aggregate_type == 'MEAN':
            return float(dps.mean(skipna=skipna))
        elif self.aggregate_type == 'MEDIAN':
            return float(dps.median(skipna=skipna))
        elif self.aggregate_type == 'STD':
            return float(dps.std(skipna=skipna))
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


class GEBCODataLoader(ScalarDataLoader):

    def import_data(self, bounds):
        '''
        Load GEBCO netCDF and downsample
        '''
        logging.debug("Importing GEBCO data...")
        # Import raw data

        # Open dataset and cast to pandas df
        logging.debug(f"- Opening file {self.file_location}")
        raw_data = xr.open_dataset(self.file_location)
        # Downsample data by extracting every (y,x)th element
        # Downsampling before xarray.to_dataframe() cuts loading time hugely
        if self.ds: # If downsampling not 'None'
            logging.debug(f"- Downsampling GEBCO data by {self.ds} for (y,x), i.e. (lat, long)")
            # Taking max because bathymetry
            raw_data = raw_data.coarsen(lat=self.ds[1]).max()
            raw_data = raw_data.coarsen(lon=self.ds[0]).max()
        
        return raw_data

    def get_datapoints(self, bounds):
        '''
        Retrieve all datapoints within bounds

        Args:
            bounds (Boundary): Boundary object with limits of lat/long
        
        Returns:
            dps (pd.Series): Datapoints within boundary limits
        '''
        dps = self.data
        dps = dps.sel(lon=slice(bounds.get_long_min(), bounds.get_long_max()))
        dps = dps.sel(lat=slice(bounds.get_lat_min(),  bounds.get_lat_max() ))

        dps = dps.to_dataframe().reset_index()

        return dps[self.data_name]

class AMSRDataLoader(ScalarDataLoader):

    def import_data(self, bounds):
        '''
        Load AMSR netCDF from folder
        '''
        def retrieve_data(filename):
            data = xr.open_dataset(filename)
            # Extract date from filename
            date = filename.split('-')[-2]
            date = f'{date[:4]}-{date[4:6]}-{date[6:]}'
            # Add date to data
            data = data.assign_coords(time=date)
            return data
        
        logging.debug("Importing AMSR data...")

        # If single NetCDF File specified
        if self.file_location[-3:] == '.nc':
            logging.debug(f"- Opening file {self.file_location}")
            raw_data = retrieve_data(self.file_location)
        # If folder specified
        elif self.file_location[-1] in ('/','\\'):
            # Open folder and read in files
            logging.debug(f"- Searching folder {self.file_location}")
            raw_data_array = []
            for file in glob.glob(f'{self.file_location}*.nc'):
                logging.debug(f"- Opening file {file}")
                raw_data_array.append(retrieve_data(file))
            raw_data = xr.concat(raw_data_array,'time')
        else:
            raise ValueError(f'{self.file_location} not a valid .nc or folder!')
        
        raw_df = raw_data.to_dataframe().reset_index()
        # AMSR data is in a EPSG:3412 projection and must be reprojected into
        # EPSG:4326
        in_proj = CRS('EPSG:3412')
        out_proj = CRS('EPSG:4326')
        
        logging.debug(f'- Reprojecting from {in_proj} to {out_proj}')
        x, y = Transformer.from_crs(in_proj, out_proj, always_xy=True).transform(
            raw_data['x'].to_numpy(), raw_data['y'].to_numpy())

        # Format final output dataframe
        reprojected_df = pd.DataFrame({
            'lat': y,
            'long': x,
            'SIC': raw_df['z'],
            'time': pd.to_datetime(raw_df['time'])
        })
        return reprojected_df

    def get_datapoints(self, bounds):

        mask = (self.data['lat']  >= bounds.get_lat_min())  & \
               (self.data['lat']  <  bounds.get_lat_max())  & \
               (self.data['long'] >= bounds.get_long_min()) & \
               (self.data['long'] <  bounds.get_long_max()) & \
               (self.data['time'] >= bounds.get_time_min()) & \
               (self.data['time'] <  bounds.get_time_max())
                   
        return self.data.loc[mask]
        
class SOSEDataLoader(ScalarDataLoader):
    
    def import_data(self, ):
        pass
  
class ThicknessDataLoader(ScalarDataLoader):
    
    def import_data(self):
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
                                         bounds.get_long_min(), 0.05)]
        
        # lats = [lat for lat in np.arange(-90, 90, 0.5)]
        # lons = [lon for lon in np.arange(-180, 180, 0.5)]
        
        start_date = date(2013, 3, 1)
        end_date = date(2013, 3, 14)
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
        
        thickness_df = thickness_data.to_dataframe().reset_index()
        thickness_df = thickness_df.set_index(['lat','long','time']).reset_index()
        
        return thickness_df
    
class DensityDataLoader(ScalarDataLoader):
    
    def import_data(self):
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
        
        
        lats = [lat for lat in np.arange(-65, -60, 0.05)]
        lons = [lon for lon in np.arange(-70, -50, 0.05)]
        
        # lats = [lat for lat in np.arange(-90, 90, 0.5)]
        # lons = [lon for lon in np.arange(-180, 180, 0.5)]
        
        start_date = date(2013, 3, 1)
        end_date = date(2013, 3, 14)
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
        
        density_df = density_data.to_dataframe().reset_index()
        density_df = density_df.set_index(['lat','long','time']).reset_index()
        
        return density_df

if __name__=='__main__':

    factory = DataLoaderFactory
    bounds = Boundary([-65,-60], [-70,-50], ['1970-01-01','2021-12-31'])
    
    if True: # Run GEBCO
        params = {
            'file': 'PolarRoute/datastore/bathymetry/GEBCO/gebco_2022_n-40.0_s-90.0_w-140.0_e0.0.nc',
            'downsample_factors': (5,5),
            'data_name': 'elevation',
            'aggregate_type': 'MAX'
        }

        split_conds = {
            'threshold': 620,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }

        gebco = factory.get_dataloader('GEBCO', params, min_dp = 5)

        print(gebco.get_value(bounds))
        print(gebco.get_hom_condition(bounds, split_conds))

    if False: # Run AMSR
        params = {
            'folder': 'PolarRoute/datastore/sic/amsr_south/',
            # 'file': 'PolarRoute/datastore/sic/amsr_south/asi-AMSR2-s6250-20201110-v5.4.nc',
            'data_name': 'SIC',
            'aggregate_type': 'MEAN'
        }

        split_conds = {
            'threshold': 80,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }

        amsr = factory.get_dataloader('AMSR', params, min_dp = 5)

        print(amsr.get_value(bounds))
        print(amsr.get_hom_condition(bounds, split_conds))

    if False: # Run Thickness
        params = {
            'file': '',
            'data_name': 'thickness',
        }
  
        split_conds = {
            'threshold': 0.5,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }
        
        thickness = factory.get_dataloader('thickness', params, min_dp = 1)

        print(thickness.get_value(bounds))
        print(thickness.get_hom_condition(bounds, split_conds))

    if False: # Run Density
        params = {
            'file': '',
            'data_name': 'thickness',
        }
  
        split_conds = {
            'threshold': 900,
            'upper_bound': 0.9,
            'lower_bound': 0.1
        }
        
        density = factory.get_dataloader('thickness', params, min_dp = 1)

        print(density.get_value(bounds))
        print(density.get_hom_condition(bounds, split_conds))
