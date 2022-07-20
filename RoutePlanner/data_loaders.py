"""
    functions for loading datasets into the py_RoutePlanner.
    the pyRoutePlanner requires data as a pandas dataframe
    in for following format:

    long | lat | (time)* | value_1 | ... | value_n

    *time is optional

    long and lat values must be in a EPSG:4326 projection
"""

import xarray as xr
import pandas as pd
import numpy as np
import math
from pyproj import Transformer
from pyproj import CRS

def load_amsr(params, long_min, long_max, lat_min,
    lat_max, time_start, time_end):
    """
        Load AMSR2 data from a netCDF file and transform
        it into a format injestable by the pyRoutePlanner
    """

    amsr = xr.open_dataset(params['file'])
    amsr = amsr.sel(time=slice(time_start, time_end))
    amsr_df = amsr.to_dataframe()
    amsr_df = amsr_df.reset_index()

    # AMSR data is in a EPSG:3412 projection and must be reprojected into
    # EPSG:4326
    in_proj = CRS('EPSG:3412')
    out_proj = CRS('EPSG:4326')

    x,y = Transformer.from_crs(in_proj, out_proj,always_xy=True).transform(
        amsr_df['x'].to_numpy(),amsr_df['y'].to_numpy())

    amsr_df['lat'] = y
    amsr_df['long'] = x
    amsr_df = amsr_df[['lat','long','z','time']]

    if "data_name" in params:
        amsr_df = amsr_df.rename(columns={'z':params['data_name']})
    else:
        amsr_df = amsr_df.rename(columns={'z':'amsr_SIC'})

    

    amsr_df = amsr_df[amsr_df['long'].between(long_min, long_max)]
    amsr_df = amsr_df[amsr_df['lat'].between(lat_min, lat_max)]

    return amsr_df

def load_bsose(params, long_min, long_max, lat_min,
    lat_max, time_start, time_end):
    """
        Load BSOSE data from a netCDF file and transform it
        into a format injestable by the pyRoutePlanner
    """

    bsose = xr.open_dataset(params['file'])
    bsose = bsose.sel(time = slice(time_start, time_end))
    bsose_df = bsose.to_dataframe()
    bsose_df = bsose_df.reset_index()

    # BSOSE data is indexed between 0:360 degrees in longitude where as the route planner
    # requires data index between -180:180 degrees in longitude
    bsose_df['long'] = bsose_df['XC'].apply(lambda x: x - 360 if x > 180 else x)
    bsose_df['lat'] = bsose_df['YC']
    bsose_df = bsose_df[['lat','long','SIarea','time']]

    if "data_name" in params:
        bsose_df = bsose_df.rename(columns = {'SIarea':params['data_name']})
    else:
        bsose_df = bsose_df.rename(columns = {'SIarea':'bsose_SIC'})

    bsose_df = bsose_df[bsose_df['long'].between(long_min, long_max)]
    bsose_df = bsose_df[bsose_df['lat'].between(lat_min, lat_max)]

    return bsose_df

def load_bsose_depth(params, long_min, long_max, lat_min,
    lat_max, time_start, time_end):
    """
        Load BSOSE data from a netCDF file and transform it
        into a format injestable by the pyRoutePlanner
    """

    bsose = xr.open_dataset(params['file'])
    bsose = bsose.sel(time = slice(time_start, time_end))
    bsose_df = bsose.to_dataframe()
    bsose_df = bsose_df.reset_index()

    # BSOSE data is indexed between 0:360 degrees in longitude where as the route planner
    # requires data index between -180:180 degrees in longitude
    bsose_df['long'] = bsose_df['XC'].apply(lambda x: x - 360 if x > 180 else x)
    bsose_df['lat'] = bsose_df['YC']
    bsose_df = bsose_df[['lat','long','Depth','time']]
    bsose_df['Depth'] = -bsose_df['Depth']

    if "data_name" in params:
        bsose_df = bsose_df.rename(columns ={'Depth':params['data_name']})
    else:
        bsose_df = bsose_df.rename(columns ={'Depth':'depth'})

    bsose_df = bsose_df[bsose_df['long'].between(long_min, long_max)]
    bsose_df = bsose_df[bsose_df['lat'].between(lat_min, lat_max)]

    return bsose_df


def load_gebco(params, long_min, long_max, lat_min,
    lat_max, time_start, time_end):
    """
        Load GEBCO data from a netCDF file and transform it
        into a format injestable by the pyRoutePlanner
    """

    gebco = xr.open_dataset(params['file'])
    gebco_df = gebco.to_dataframe()
    gebco_df = gebco_df.reset_index()
    gebco_df = gebco_df.rename(columns = {'lon':'long'})

    if "data_name" in params:
        gebco_df = gebco_df.rename(columns = {'elevation':params['data_name']})

    gebco_df = gebco_df[gebco_df['long'].between(long_min, long_max)]
    gebco_df = gebco_df[gebco_df['lat'].between(lat_min,lat_max)]

    return gebco_df

def load_sose_currents(params, long_min, long_max, lat_min,
    lat_max, time_start, time_end):
    """
        Load SOSE current data from a netCDF file and#
        transform it into a format that is injestable
        by the pyRoutePlanner
    """

    sose = xr.open_dataset(params['file'])
    sose_df = sose.to_dataframe()
    sose_df = sose_df.reset_index()

    # SOSE data is indexed between 0:360 degrees in longitude where as the route planner
    # requires data index between -180:180 degrees in longitude
    sose_df['long'] = sose_df['lon'].apply(lambda x: x - 360 if x > 180 else x)
    
    sose_df = sose_df[['lat','long','uC', 'vC']]

    sose_df = sose_df[sose_df['long'].between(long_min, long_max)]
    sose_df = sose_df[sose_df['lat'].between(lat_min, lat_max)]

    return sose_df

def load_modis(params, long_min, long_max, lat_min,
    lat_max, time_start, time_end):
    """
        Load MODIS data from a netCDF file and transform it
        into a format that is injestable by the pyRoutePlanner
    """

    modis = xr.open_dataset(params['file'])
    modis_df = modis.to_dataframe()
    modis_df = modis_df.reset_index()

    # MODIS Sea Ice Concentration data is partially obsured by cloud cover.
    # Where a datapoint indicates that there is cloud cover above it, 
    # set the SIC of that datapoint to NaN
    modis_df['iceArea'] = np.where(modis_df['cloud'] == 1, np.NaN, modis_df['iceArea'])
    modis_df = modis_df.rename(columns = {'iceArea':'modis_SIC'})

    modis_df = modis_df[modis_df['long'].between(long_min, long_max)]
    modis_df = modis_df[modis_df['lat'].between(lat_min, lat_max)]

    return modis_df

def load_era5_wind(params, long_min, long_max, lat_min,
    lat_max, time_start, time_end):
    """
        Load era5 wind data from a netCDF file and transform it
        into a format that is injestable by the pyRoutePlanner
    """

    era5_wind = xr.open_dataset(params['file'])

    # era5_wind data is available in monthly slices, not daily. 
    # time_start is set to the start of the given month to ensure that data is loaded.
    time_start_split = time_start.split('-')
    time_start = time_start_split[0] + "-" + time_start_split[1] + "-01"

    era5_wind = era5_wind.sel(time=slice(time_start, time_end))

    era5_wind_df = era5_wind.to_dataframe()
    era5_wind_df = era5_wind_df.reset_index()

    era5_wind_df = era5_wind_df.rename(columns = {'longitude':'long', 'latitude':'lat'})
    era5_wind_df = era5_wind_df[era5_wind_df['long'].between(long_min, long_max)]
    era5_wind_df = era5_wind_df[era5_wind_df['lat'].between(lat_min, lat_max)]

    return era5_wind_df
    