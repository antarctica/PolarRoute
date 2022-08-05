"""
    functions for loading datasets into the py_RoutePlanner.
    The pyRoutePlanner requires data as a pandas dataframe
    in the following format:

    long | lat | (time)* | value_1 | ... | value_n

    *time is optional

    Note:
        long and lat values must be in a EPSG:4326 projection
"""

import xarray as xr
import pandas as pd
import numpy as np
from pyproj import Transformer
from pyproj import CRS
from datetime import timedelta, datetime
import glob

def load_amsr_folder(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    """
        Load AMSR2 data from a folder containing seperate
        days of AMSR2 data in netCDF files and transform
        it into a format ingestable by Polar Route

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retreived
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

            params (dict): A dictionary containing optional parameters. This
                function requires -

                params['folder'] (string): folder location of the AMSR2 dataset.
                    Files within the folder must be named in the form
                    *asi-AMSR2-s6250-<year><month><day>-v.5.4.nc*

        Returns:
            amsr_df (Dataframe): A dataframe containing AMSR2 Sea Ice Concentration
                data. The dataframe is of the format -

                lat | long | time | SIC
    """

    # strip year/month/day from time_start and end_start strings.
    start_year = float(time_start.split('-')[0])
    start_month = float(time_start.split('-')[1])
    start_day = float(time_start.split('-')[2])

    end_year = float(time_end.split('-')[0])
    end_month = float(time_end.split('-')[1])
    end_day = float(time_end.split('-')[2])

    amsr_array = []
    time_array = []

    # iterate through all files in folder passed as a parameter
    for file in sorted(glob.glob(params['folder'] + '/*.nc')):
        # string year/month/day out of file name.
        # files in the folder must be named in the form 
        # asi-AMSR2-s6250-<year><month><day>-v.5.4.nc
        year = int(file.split('-')[-2][0:4])
        month = int(file.split('-')[-2][4:6])
        day = int(file.split('-')[-2][6:])

        if start_year <= year <= end_year:
            if start_month <= month <= end_month:
                if start_day <= day <= end_day:

                    amsr = xr.open_dataset(file)
                    amsr_array.append(amsr)

                    time = str(year) + '-' + str(month) + '-' + str(day)
                    time_array.append(time)

    amsr_concat = xr.concat(amsr_array, pd.Index(time_array, name="time"))
    amsr_df = amsr_concat.to_dataframe()
    amsr_df = amsr_df.reset_index()

    in_proj = CRS('EPSG:3412')
    out_proj = CRS('EPSG:4326')

    x,y = Transformer.from_crs(in_proj, out_proj, always_xy=True).transform(
        amsr_df['x'].to_numpy(), amsr_df['y'].to_numpy())

    amsr_df['lat'] = y
    amsr_df['long'] = x
    amsr_df = amsr_df[['lat', 'long', 'z', 'time']]

    amsr_df = amsr_df.rename(columns={'z': 'SIC'})

    amsr_df = amsr_df[amsr_df['long'].between(long_min, long_max)]
    amsr_df = amsr_df[amsr_df['lat'].between(lat_min, lat_max)]


    return amsr_df

def load_amsr(params, long_min, long_max, lat_min,
    lat_max, time_start, time_end):
    """
        Load AMSR2 data from a netCDF file and transform
        it into a format ingestable by the pyRoutePlanner

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retreived
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

            params (dict): A dictionary containing optional parameters. This
                function requires -

                params['file'] (string): file location of the AMSR2 dataset

        Returns:
            amsr_df (Dataframe): A dataframe containing AMSR2 Sea Ice Concentration
                data. The dataframe is of the format -

                lat | long | time | SIC
    """

    amsr = xr.open_dataset(params['file'])
    amsr = amsr.sel(time=slice(time_start, time_end))
    amsr_df = amsr.to_dataframe()
    amsr_df = amsr_df.reset_index()

    # AMSR data is in a EPSG:3412 projection and must be reprojected into
    # EPSG:4326
    in_proj = CRS('EPSG:3412')
    out_proj = CRS('EPSG:4326')

    x,y = Transformer.from_crs(in_proj, out_proj, always_xy=True).transform(
        amsr_df['x'].to_numpy(), amsr_df['y'].to_numpy())

    amsr_df['lat'] = y
    amsr_df['long'] = x
    amsr_df = amsr_df[['lat', 'long', 'z', 'time']]

    amsr_df = amsr_df.rename(columns={'z': 'SIC'})

    amsr_df = amsr_df[amsr_df['long'].between(long_min, long_max)]
    amsr_df = amsr_df[amsr_df['lat'].between(lat_min, lat_max)]

    return amsr_df


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def load_density(params, long_min, long_max, lat_min,
              lat_max, time_start, time_end):
    """
        Create ice density dataframe for given time and region and put it into a format ingestable by the pyRoutePlanner.
        Data taken from Table 3 in: doi:10.1029/2007JC004254

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retreived
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

        Returns:
            density_df (Dataframe): A dataframe containing ice density
                data. The dataframe is of the format -

                lat | long | time | density
    """

    def icedensity(d):
        seasons = {1: 'su', 2: 'su', 3: 'a', 4: 'a', 5: 'a', 6: 'w', 7: 'w', 8: 'w', 9: 'sp', 10: 'sp', 11: 'sp',
                   12: 'su'}
        densities = {'su': 875.0, 'sp': 900.0, 'a': 900.0, 'w': 920.0}

        month = int(d[5:7])
        season = seasons[month]
        den = densities[season]
        return den

    dense_data = []

    start_date = datetime.strptime(time_start, "%Y-%m-%d").date()
    end_date = datetime.strptime(time_end, "%Y-%m-%d").date()

    for single_date in daterange(start_date, end_date):
        dt = single_date.strftime("%Y-%m-%d")
        for lat in np.arange(lat_min, lat_max, 0.05):#0.16):
            for long in np.arange(long_min, long_max,0.05):# 0.16):
                dense_data.append({'time': dt, 'lat': lat, 'long': long, 'density': icedensity(dt)})

    dense_df = pd.DataFrame(dense_data).set_index(['lat', 'long', 'time'])
    dense_df = dense_df.reset_index()

    return dense_df


def load_thickness(params, long_min, long_max, lat_min,
              lat_max, time_start, time_end):
    """
        Create ice thickness dataframe for given time and region and put it into a format ingestable by the pyRoutePlanner.
        Data taken from Table 3 in: doi:10.1029/2007JC004254

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retreived
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

        Returns:
            thickness_df (Dataframe): A dataframe containing ice thickness
                data. The dataframe is of the format -

                lat | long | time | thickness
    """

    def icethickness(d, long):
        """
            Returns ice thickness. Data taken from Table 3 in: doi:10.1029/2007JC004254
        """
        # The table has missing data points for Bellinghausen Autumn and Weddell W Winter, may require further thought
        thicknesses = {'Ross': {'w': 0.72, 'sp': 0.67, 'su': 1.32, 'a': 0.82, 'y': 1.07},
                       'Bellinghausen': {'w': 0.65, 'sp': 0.79, 'su': 2.14, 'a': 0.79, 'y': 0.90},
                       'Weddell E': {'w': 0.54, 'sp': 0.89, 'su': 0.87, 'a': 0.44, 'y': 0.73},
                       'Weddell W': {'w': 1.33, 'sp': 1.33, 'su': 1.20, 'a': 1.38, 'y': 1.33},
                       'Indian': {'w': 0.59, 'sp': 0.78, 'su': 1.05, 'a': 0.45, 'y': 0.68},
                       'West Pacific': {'w': 0.72, 'sp': 0.68, 'su': 1.17, 'a': 0.75, 'y': 0.79}
                       }
        seasons = {1: 'su', 2: 'su', 3: 'a', 4: 'a', 5: 'a', 6: 'w', 7: 'w', 8: 'w', 9: 'sp', 10: 'sp', 11: 'sp',
                   12: 'su'}
        month = int(d[5:7])
        season = seasons[month]
        sea = None

        if -130 <= long < -60:
            sea = 'Bellinghausen'
        elif -60 <= long < -45:
            sea = 'Weddell W'
        elif -45 <= long < 20:
            sea = 'Weddell E'
        elif 20 <= long < 90:
            sea = 'Indian'
        elif 90 <= long < 160:
            sea = 'West Pacific'
        elif (160 <= long < 180) or (-180 <= long < -130):
            sea = 'Ross'

        return thicknesses[sea][season]

    thick_data = []
    start_date = datetime.strptime(time_start, "%Y-%m-%d").date()
    end_date = datetime.strptime(time_end, "%Y-%m-%d").date()

    for single_date in daterange(start_date, end_date):
        dt = single_date.strftime("%Y-%m-%d")
        for lat in np.arange(lat_min, lat_max, 0.1):
            for lng in np.arange(long_min, long_max, 0.1):
                thick_data.append({'time': dt, 'lat': lat, 'long': lng, 'thickness': icethickness(dt, lng)})

    thick_df = pd.DataFrame(thick_data).set_index(['lat', 'long', 'time'])
    thick_df = thick_df.reset_index()

    return thick_df


def load_bsose(params, long_min, long_max, lat_min,
    lat_max, time_start, time_end):
    """
        Load BSOSE data from a netCDF file and transform it
        into a format ingestable by the pyRoutePlanner

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retreived
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

            params (dict): A dictionary containing optional parameters. This
                function requires -

                params['file'] (string): file location of the BSOSE dataset

        Returns:
            bsose_df (Dataframe): A dataframe containing BSOSE Sea Ice Concentration
                data. The dataframe is of the format -

                lat | long | time | SIC
    """

    bsose = xr.open_dataset(params['file'])
    bsose = bsose.sel(time=slice(time_start, time_end))
    bsose_df = bsose.to_dataframe()
    bsose_df = bsose_df.reset_index()

    # BSOSE data is indexed between 0:360 degrees in longitude where as the route planner
    # requires data index between -180:180 degrees in longitude
    bsose_df['long'] = bsose_df['XC'].apply(lambda x: x - 360 if x > 180 else x)
    bsose_df['lat'] = bsose_df['YC']
    bsose_df = bsose_df[['lat', 'long', 'SIarea', 'time']]

    bsose_df = bsose_df.rename(columns={'SIarea': 'SIC'})
    bsose_df['SIC'] = bsose_df['SIC']*100.

    bsose_df = bsose_df[bsose_df['long'].between(long_min, long_max)]
    bsose_df = bsose_df[bsose_df['lat'].between(lat_min, lat_max)]

    return bsose_df


def load_bsose_depth(params, long_min, long_max, lat_min,
    lat_max, time_start, time_end):
    """
        Load BSOSE data from a netCDF file and transform it
        into a format ingestable by the pyRoutePlanner

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retreived
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

            params (dict): A dictionary containing optional parameters. This
                function requires -

                params['file'] (string): file location of the BSOSE dataset

        Returns:
            bsose_df (Dataframe): A dataframe containing BSOSE bathymetry
                data. The dataframe is of the format -

                lat | long | time | elevation
    """

    bsose = xr.open_dataset(params['file'])
    bsose = bsose.sel(time=slice(time_start, time_end))
    bsose_df = bsose.to_dataframe()
    bsose_df = bsose_df.reset_index()

    # BSOSE data is indexed between 0:360 degrees in longitude where as the route planner
    # requires data index between -180:180 degrees in longitude
    bsose_df['long'] = bsose_df['XC'].apply(lambda x: x - 360 if x > 180 else x)
    bsose_df['lat'] = bsose_df['YC']
    bsose_df = bsose_df[['lat', 'long', 'Depth', 'time']]
    bsose_df['Depth'] = -bsose_df['Depth']

    bsose_df = bsose_df.rename(columns={'Depth': 'elevation'})

    bsose_df = bsose_df[bsose_df['long'].between(long_min, long_max)]
    bsose_df = bsose_df[bsose_df['lat'].between(lat_min, lat_max)]

    return bsose_df


def load_gebco(params, long_min, long_max, lat_min,
    lat_max, time_start, time_end):
    """
        Load GEBCO data from a netCDF file and transform it
        into a format ingestable by the pyRoutePlanner

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retreived
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

            params (dict): A dictionary containing optional parameters. This
                function requires -

                params['file'] (string): file location of the GEBCO dataset

        Returns:
            gebco_df (Dataframe): A dataframe containing GEBCO elevation
                data. The dataframe is of the format -

                lat | long | time | elevation
    """

    gebco = xr.open_dataset(params['file'])
    gebco_df = gebco.to_dataframe()
    gebco_df = gebco_df.reset_index()
    gebco_df = gebco_df.rename(columns={'lon': 'long'})

    gebco_df = gebco_df[gebco_df['long'].between(long_min, long_max)]
    gebco_df = gebco_df[gebco_df['lat'].between(lat_min, lat_max)]

    return gebco_df


def load_sose_currents(params, long_min, long_max, lat_min,
    lat_max, time_start, time_end):
    """
        Load SOSE current data from a netCDF file and#
        transform it into a format that is ingestable
        by the pyRoutePlanner

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retreived
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

            params (dict): A dictionary containing optional parameters. This
                function requires -

                params['file'] (string): file location of the SOSE dataset

        Returns:
            sose_df (Dataframe): A dataframe containing SOSE current
                data. The dataframe is of the format -

                lat | long | time | uC | vC
    """

    sose = xr.open_dataset(params['file'])
    sose_df = sose.to_dataframe()
    sose_df = sose_df.reset_index()

    # SOSE data is indexed between 0:360 degrees in longitude where as the route planner
    # requires data index between -180:180 degrees in longitude
    sose_df['long'] = sose_df['lon'].apply(lambda x: x - 360 if x > 180 else x)
    
    sose_df = sose_df[['lat', 'long', 'uC', 'vC']]

    sose_df = sose_df[sose_df['long'].between(long_min, long_max)]
    sose_df = sose_df[sose_df['lat'].between(lat_min, lat_max)]

    return sose_df


def load_modis(params, long_min, long_max, lat_min,
    lat_max, time_start, time_end):
    """
        Load MODIS data from a netCDF file and transform it
        into a format that is ingestable by the pyRoutePlanner

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retreived
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

            params (dict): A dictionary containing optional parameters. This
                function requires -
  
                params['file'] (string): file location of the MODIS dataset

        Returns:
            modis_df (Dataframe): A dataframe containing MODIS Sea Ice Concentration
                data. The dataframe is of the format -

                lat | long | time | SIC | cloud
    """

    modis = xr.open_dataset(params['file'])
    modis_df = modis.to_dataframe()
    modis_df = modis_df.reset_index()

    # MODIS Sea Ice Concentration data is partially obscured by cloud cover.
    # Where a datapoint indicates that there is cloud cover above it, 
    # set the SIC of that datapoint to NaN
    modis_df['iceArea'] = np.where(modis_df['cloud'] == 1, np.NaN, modis_df['iceArea'])
    modis_df = modis_df.rename(columns={'iceArea': 'SIC'})
    modis_df['SIC'] = modis_df['SIC']*10.

    modis_df = modis_df[modis_df['long'].between(long_min, long_max)]
    modis_df = modis_df[modis_df['lat'].between(lat_min, lat_max)]

    return modis_df


def load_era5_wind(params, long_min, long_max, lat_min,
    lat_max, time_start, time_end):
    """
        Load era5 wind data from a netCDF file and transform it
        into a format that is ingestable by the pyRoutePlanner

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retreived
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

            params (dict): A dictionary containing optional parameters. This
                function requires -

                params['file'] (string): file location of the era5 dataset

        Returns:
            era5_wind_df (Dataframe): A dataframe containing era5 wind
                data. The dataframe is of the format -

                lat | long | time | u10 | v10
    """

    era5_wind = xr.open_dataset(params['file'])

    # era5_wind data is available in monthly slices, not daily. 
    # time_start is set to the start of the given month to ensure that data is loaded.
    time_start_split = time_start.split('-')
    time_start = time_start_split[0] + "-" + time_start_split[1] + "-01"

    era5_wind = era5_wind.sel(time=slice(time_start, time_end))

    era5_wind_df = era5_wind.to_dataframe()
    era5_wind_df = era5_wind_df.reset_index()

    era5_wind_df = era5_wind_df.rename(columns={'longitude': 'long', 'latitude': 'lat'})
    era5_wind_df = era5_wind_df[era5_wind_df['long'].between(long_min, long_max)]
    era5_wind_df = era5_wind_df[era5_wind_df['lat'].between(lat_min, lat_max)]

    return era5_wind_df
    