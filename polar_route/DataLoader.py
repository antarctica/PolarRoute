from abc import ABC, abstractmethod
from datetime import datetime

import xarray as xr
import pandas as pd
import numpy as np

import logging
import glob

from polar_route.Boundary import Boundary

class DataLoaderFactory:

	def get_dataloader(name, params, min_dp=5):
		# TODO Replace **kwargs with params
		if 'file' in params:  file_location = params['file']  
		else:                 raise ValueError('File not specified')
		
		if 'downsample_factors' in params:  ds = params['downsample_factors']
		else:                               ds = None

		if 'data_name' in params: data_name = params['data_name'] 
		else: 					  data_name = None
		
		if 'aggregate_type' in params: agg_type = params['aggregate_type']  
		else:                          agg_type = 'MEAN'


		if   name == 'GEBCO':  data_loader = GEBCO_DataLoader
		elif name == 'AMSR':  data_loader = AMSR_DataLoader
		elif name == 'SOSE':  data_loader = SOSE_DataLoader
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
	def __init__(self, file_location, min_dp=5, ds=None, data_name=None, aggregate_type="MEAN"):
		self.file_location  = file_location
		self.min_dp         = min_dp
		self.ds             = ds

		# Cast string to uppercase to accept mismatched case
		self.aggregate_type = aggregate_type.upper()

		# Sets self.data from user deined functions _set_data()
		self._set_data() 
		# If no data name specified, retrieve from self.data
		self.data_name = data_name if data_name else self._get_data_name()
		
		logging.debug(f'- Successfully extracted {self.data_name}')


	@abstractmethod
	def _set_data(self, **kwargs):
		'''
		Import data from whatever format this datasource is in and
		reproject it to EPSG:4326 (our working projection)
		'''
		pass


	def _get_data_name(self):
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


	def get_datapoints(self, bounds):
		'''
		Retrieve all datapoints within bounds

		Args:
			bounds (Boundary): Boundary object with limits of lat/long
		
		Returns:
			dps (pd.Series): Datapoints within boundary limits
		'''
		dps = self.data
		dps = dps[dps['lat'].between(bounds.get_lat_min(), 
									bounds.get_lat_max())]
		dps = dps[dps['long'].between(bounds.get_long_min(), 
									bounds.get_long_max())]

		return dps[self.data_name]


class GEBCO_DataLoader(ScalarDataLoader):

	def _set_data(self):
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
		
		# Final output form
		raw_data = raw_data.rename({'lon': 'long'})
		self.data = raw_data.to_dataframe().reset_index()


if __name__=='__main__':

	params = {
		'file': 'PolarRoute/datastore/bathymetry/GEBCO/gebco_2022_n-40.0_s-90.0_w-140.0_e0.0.nc',
		'downsample_factors': (5,5),
		'data_name': 'elevation',
		'aggregate_type': 'MAX'
	}

	factory = DataLoaderFactory
	gebco = factory.get_dataloader('GEBCO', params, min_dp = 5)

	bounds = Boundary([-85,-84.9], [-135,-134.9], ['1970-01-01','2021-12-31'])


	print(gebco.get_value(bounds))

	split_conds = {
	'threshold': 620,
	'upper_bound': 0.9,
	'lower_bound': 0.1
	}
	print(gebco.get_hom_condition(bounds, split_conds))
