"""
    functions for loading datasets into PolarRoute.
    PolarRoute requires data as a pandas dataframe
    in the following format:

    long | lat | (time)* | value_1 | ... | value_n

    *time is optional

    Note:
        long and lat values must be in a EPSG:4326 projection
"""

import glob
import logging

from datetime import datetime

import xarray as xr
import pandas as pd
import numpy as np

from pyproj import Transformer
from pyproj import CRS

from polar_route.utils import date_range, timed_call


@timed_call
def load_amsr_folder(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    """
        Load AMSR2 data from a folder containing separate
        days of AMSR2 data in netCDF files and transform
        it into a format ingestable by PolarRoute

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retrieved
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
    logging.debug("opening folder {}".format(params['folder']))

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
                    logging.debug("found file {}".format(file))
                    amsr_array.append(amsr)

                    time = str(year) + '-' + str(month) + '-' + str(day)
                    time_array.append(time)

    amsr_concat = xr.concat(amsr_array, pd.Index(time_array, name="time"))
    amsr_df = amsr_concat.to_dataframe()
    amsr_df = amsr_df.reset_index()

    if ('Hemisphere' in params.keys()) and (params['Hemisphere'] == 'North'):
        in_proj = CRS('EPSG:3411')
    elif ('Hemisphere' in params.keys()) and (params['Hemisphere'] == 'South'):
        in_proj = CRS('EPSG:3412')
    else:
        in_proj = CRS('EPSG:3412')
    out_proj = CRS('EPSG:4326')

    logging.debug("reprojecting to EPSG:4326")
    x, y = Transformer.from_crs(in_proj, out_proj, always_xy=True).transform(
        amsr_df['x'].to_numpy(), amsr_df['y'].to_numpy())

    amsr_df['lat'] = y
    amsr_df['long'] = x
    amsr_df = amsr_df[['lat', 'long', 'z', 'time']]

    amsr_df = amsr_df.rename(columns={'z': 'SIC'})

    amsr_df = amsr_df[amsr_df['long'].between(long_min, long_max)]
    amsr_df = amsr_df[amsr_df['lat'].between(lat_min, lat_max)]

    logging.debug("returned {} datapoints".format(len(amsr_df.index)))

    return amsr_df


@timed_call
def load_amsr(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    """
        Load AMSR2 data from a netCDF file and transform
        it into a format ingestable by PolarRoute

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retrieved
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
    logging.debug("opening file {}".format(params['file']))
    amsr = xr.open_dataset(params['file'])
    amsr = amsr.sel(time=slice(time_start, time_end))
    amsr_df = amsr.to_dataframe()
    amsr_df = amsr_df.reset_index()

    # AMSR data is in a EPSG:3412 projection and must be reprojected into
    # EPSG:4326

    logging.debug("reprojecting to EPSG:4326")
    in_proj = CRS('EPSG:3412')
    out_proj = CRS('EPSG:4326')

    x, y = Transformer.from_crs(in_proj, out_proj, always_xy=True).transform(
        amsr_df['x'].to_numpy(), amsr_df['y'].to_numpy())

    amsr_df['lat'] = y
    amsr_df['long'] = x
    amsr_df = amsr_df[['lat', 'long', 'z', 'time']]

    amsr_df = amsr_df.rename(columns={'z': 'SIC'})

    amsr_df = amsr_df[amsr_df['long'].between(long_min, long_max)]
    amsr_df = amsr_df[amsr_df['lat'].between(lat_min, lat_max)]

    logging.debug("returned {} datapoints".format(len(amsr_df.index)))
    return amsr_df


@timed_call
def load_density(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    """
        Create ice density dataframe for given time and region and put it into a format ingestable by the pyRoutePlanner.
        Data taken from Table 3 in: doi:10.1029/2007JC004254

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retrieved
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

    seasons = {1: 'su', 2: 'su', 3: 'a', 4: 'a', 5: 'a', 6: 'w', 7: 'w', 8: 'w', 9: 'sp', 10: 'sp', 11: 'sp',
               12: 'su'}
    densities = {'su': 875.0, 'sp': 900.0, 'a': 900.0, 'w': 920.0}

    def ice_density(d):
        month = d.month
        season = seasons[month]
        den = densities[season]
        return den

    start_date = datetime.strptime(time_start, "%Y-%m-%d").date()
    end_date = datetime.strptime(time_end, "%Y-%m-%d").date()

    lats = [lat for lat in np.arange(lat_min, lat_max, 0.05)]
    lons = [lon for lon in np.arange(long_min, long_max, 0.05)]
    dates = [single_date for single_date in date_range(start_date, end_date)]

    density_data = xr.DataArray(
        data=[[[ice_density(dt)
                for _ in lons]
               for _ in lats]
              for dt in dates],
        coords=dict(
            lat=lats,
            long=lons,
            time=[dt.strftime("%Y-%m-%d") for dt in dates],
        ),
        dims=("time", "lat", "long"),
        name="density",
    )

    density_df = density_data.\
        to_dataframe().\
        reset_index().\
        set_index(['lat', 'long', 'time']).reset_index()

    logging.debug("returned {} datapoints".format(len(density_df.index)))
    return density_df


@timed_call
def load_thickness(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    """
        Create ice thickness dataframe for given time and region and put it into a format ingestable by PolarRoute.
        Data taken from Table 3 in: doi:10.1029/2007JC004254

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retrieved
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
    thicknesses = {'Ross': {'w': 0.72, 'sp': 0.67, 'su': 1.32, 'a': 0.82, 'y': 1.07},
                   'Bellinghausen': {'w': 0.65, 'sp': 0.79, 'su': 2.14, 'a': 0.79, 'y': 0.90},
                   'Weddell E': {'w': 0.54, 'sp': 0.89, 'su': 0.87, 'a': 0.44, 'y': 0.73},
                   'Weddell W': {'w': 1.33, 'sp': 1.33, 'su': 1.20, 'a': 1.38, 'y': 1.33},
                   'Indian': {'w': 0.59, 'sp': 0.78, 'su': 1.05, 'a': 0.45, 'y': 0.68},
                   'West Pacific': {'w': 0.72, 'sp': 0.68, 'su': 1.17, 'a': 0.75, 'y': 0.79},
                   'None': {'w': 0.72, 'sp': 0.67, 'su': 1.32, 'a': 0.82, 'y': 1.07}}
    seasons = {1: 'su', 2: 'su', 3: 'a', 4: 'a', 5: 'a', 6: 'w', 7: 'w', 8: 'w', 9: 'sp', 10: 'sp', 11: 'sp',
               12: 'su'}

    def ice_thickness(d, long):
        """
            Returns ice thickness. Data taken from Table 3 in: doi:10.1029/2007JC004254
        """
        # The table has missing data points for Bellinghausen Autumn and Weddell W Winter, may require further thought
        month = d.month
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
        else:
            sea = 'None'

        return thicknesses[sea][season]

    start_date = datetime.strptime(time_start, "%Y-%m-%d").date()
    end_date = datetime.strptime(time_end, "%Y-%m-%d").date()

    lats = [lat for lat in np.arange(lat_min, lat_max, 0.05)]
    lons = [lon for lon in np.arange(long_min, long_max, 0.05)]
    dates = [single_date for single_date in date_range(start_date, end_date)]

    thick_data = xr.DataArray(
        data=[[[ice_thickness(dt, lng)
                for lng in lons]
               for _ in lats]
              for dt in dates],
        coords=dict(
            lat=lats,
            long=lons,
            time=[dt.strftime("%Y-%m-%d") for dt in dates],
        ),
        dims=("time", "lat", "long"),
        name="thickness",
    )

    thick_df = thick_data.\
        to_dataframe().\
        reset_index().\
        set_index(['lat', 'long', 'time']).reset_index()

    logging.debug("returning {} datapoints".format(len(thick_df.index)))
    return thick_df


@timed_call
def load_baltic_thickness_density(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    """
        Create ice thickness and density dataframe for baltic route

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retrieved
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

        Returns:
            thickness_df (Dataframe): A dataframe containing ice thickness and density
                data. The dataframe is of the format -

                lat | long | time | thickness | density
    """

    baltic_thick_data = []
    start_date = datetime.strptime(time_start, "%Y-%m-%d").date()
    end_date = datetime.strptime(time_end, "%Y-%m-%d").date()

    for single_date in date_range(start_date, end_date):
        dt = single_date.strftime("%Y-%m-%d")
        for lat in np.arange(lat_min, lat_max, 0.1):
            for lng in np.arange(long_min, long_max, 0.1):
                baltic_thick_data.append({'time': dt, 'lat': lat, 'long': lng, 'thickness': 0.3, 'density': 900.})

    baltic_thick_df = pd.DataFrame(baltic_thick_data).set_index(['lat', 'long', 'time'])
    baltic_thick_df = baltic_thick_df.reset_index()

    logging.debug("returned {} datapoints".format(len(baltic_thick_df.index)))
    return baltic_thick_df


@timed_call
def load_bsose(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    """
        Load BSOSE data from a netCDF file and transform it
        into a format ingestable by PolarRoute

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retrieved
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

            params (dict): A dictionary containing optional parameters. This
                function requires -

                params['file'] (string): file location of the BSOSE dataset

                params['units'] (string)(optional): The units SIC will be measured
                    in. <percentage> | <fraction>. Default is percentage

        Returns:
            bsose_df (Dataframe): A dataframe containing BSOSE Sea Ice Concentration
                data. The dataframe is of the format -

                lat | long | time | SIC
    """
    logging.debug("opening file {} ".format(params['file']))
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

    if 'units' in params.keys():
        if params['units'] == "percentage":
            bsose_df['SIC'] = bsose_df['SIC']*100
        if params['units'] == 'fraction':
            """""" #BSOSE source data is fractional, no transformation required
    else:
        bsose_df['SIC'] = bsose_df['SIC']*100

    bsose_df = bsose_df[bsose_df['long'].between(long_min, long_max)]
    bsose_df = bsose_df[bsose_df['lat'].between(lat_min, lat_max)]

    logging.debug("returned {} datapoints".format(len(bsose_df.index)))
    return bsose_df


@timed_call
def load_baltic_sea_ice(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    """
        Load Sea ice data for the baltic sea from a netCDF file and transform it
        into a format ingestable by PolarRoute
        Data Source: https://doi.org/10.48670/moi-00131

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retrieved
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

            params (dict): A dictionary containing optional parameters. This
                function requires -

                params['file'] (string): file location of the Baltic ice dataset

        Returns:
            baltic_df (Dataframe): A dataframe containing Baltic Sea Ice Concentration
                data. The dataframe is of the format -

                lat | long | time | SIC
    """
    logging.debug("opening file {}".format(params['file']))
    baltic = xr.open_dataset(params['file'])
    baltic = baltic.sel(time=slice(time_start, time_end))
    baltic_df = baltic.to_dataframe()
    baltic_df = baltic_df.reset_index()

    # The Baltic ice data has spatial variables: 'lon' and 'lat'
    baltic_df['long'] = baltic_df['lon']

    baltic_df = baltic_df[['lat', 'long', 'ice_concentration', 'time']]

    baltic_df = baltic_df.rename(columns={'ice_concentration': 'SIC'})

    baltic_df = baltic_df[baltic_df['long'].between(long_min, long_max)]
    baltic_df = baltic_df[baltic_df['lat'].between(lat_min, lat_max)]

    logging.debug("returning {} datapoints".format(len(baltic_df.index)))
    return baltic_df


@timed_call
def load_bsose_depth(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    """
        Load BSOSE data from a netCDF file and transform it
        into a format ingestable by PolarRoute

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retrieved
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
    logging.debug("opening file {} ".format(params['file']))
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

    logging.debug("returning {} datapoints".format(len(bsose_df.index)))
    return bsose_df


@timed_call
def load_gebco(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    """
        Load GEBCO data from a netCDF file and transform it
        into a format ingestable by the pyRoutePlanner

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retrieved
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

            params (dict): A dictionary containing optional parameters. This
                function requires -

                params['file'] (string): file location of the GEBCO dataset
                params['downsample_factors'] ([int,int]): Downsample factors in horizontal and vertical respectively.

        Returns:
            gebco_df (Dataframe): A dataframe containing GEBCO elevation
                data. The dataframe is of the format -

                lat | long | time | elevation 
    """
    logging.debug("opening file {}".format(params['file']))
    gebco = xr.open_dataset(params['file'])

    
    if 'downsample_factors' in params.keys():
        logging.debug("downsampling dataset by a factor of [{},{}]".format(params['downsample_factors'][0],
            params['downsample_factors'][1] ))
        elev = gebco['elevation'][::params['downsample_factors'][0],::params['downsample_factors'][1]]
        gebco = xr.Dataset()
        gebco['elevation'] = elev

    gebco_df = gebco.to_dataframe()
    gebco_df = gebco_df.reset_index()
    gebco_df = gebco_df.rename(columns={'lon': 'long'})

    gebco_df = gebco_df[gebco_df['long'].between(long_min, long_max)]
    gebco_df = gebco_df[gebco_df['lat'].between(lat_min, lat_max)]

    logging.debug("returned {} datapoints".format(len(gebco_df.index)))
    return gebco_df


@timed_call
def load_sose_currents(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    """
        Load SOSE current data from a netCDF file and#
        transform it into a format that is ingestable
        by the pyRoutePlanner

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retrieved
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

            params (dict): A dictionary containing optional parameters. This
                function requires -

                params['file'] (string): file location of the SOSE dataset

                params['units'] (string)(optional) : The units of measurements
                    uC and vC will be given in - <km/h> | <m/s>. Default is m/s

        Returns:
            sose_df (Dataframe): A dataframe containing SOSE current
                data. The dataframe is of the format -

                lat | long | time | uC | vC
    """
    logging.debug("opening file {}".format(params['file']))
    sose = xr.open_dataset(params['file'])
    sose_df = sose.to_dataframe()
    sose_df = sose_df.reset_index()

    # SOSE data is indexed between 0:360 degrees in longitude where as the route planner
    # requires data index between -180:180 degrees in longitude
    sose_df['long'] = sose_df['lon'].apply(lambda x: x - 360 if x > 180 else x)
    sose_df = sose_df[['lat', 'long', 'uC', 'vC']]

    sose_df = sose_df[sose_df['long'].between(long_min, long_max)]
    sose_df = sose_df[sose_df['lat'].between(lat_min, lat_max)]

    if 'units' in params.keys():
        if params['units'] == "km/h":
            sose_df['uC'] = sose_df['uC'] * 3.6
            sose_df['vC'] = sose_df['vC'] * 3.6
        if params['units'] == 'm/s':
            """""" # SOSE source data is in m/s, no transformation required

    logging.debug("returning {} datapoints".format(len(sose_df.index)))
    return sose_df


@timed_call
def load_baltic_currents(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    """
        Load Baltic Sea current data from a netCDF file and
        transform it into a format that is ingestable
        by the pyRoutePlanner
        Data source: https://doi.org/10.48670/moi-00013

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retrieved
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

            params (dict): A dictionary containing optional parameters. This
                function requires -

                params['file'] (string): file location of the Baltic current dataset

        Returns:
            bc_df (Dataframe): A dataframe containing Baltic Sea current
                data. The dataframe is of the format -

                lat | long | time | uC | vC
    """
    logging.debug("opening file {}".format(params['file']))
    bc = xr.open_dataset(params['file'])
    bc_df = bc.to_dataframe()
    bc_df = bc_df.reset_index()

    bc_df = bc_df[['latitude', 'longitude', 'uo', 'vo']]

    bc_df = bc_df.rename(columns={'longitude': 'long', 'latitude': 'lat', 'uo': 'uC', 'vo': 'vC'})

    bc_df = bc_df[bc_df['long'].between(long_min, long_max)]
    bc_df = bc_df[bc_df['lat'].between(lat_min, lat_max)]

    logging.debug("returned {} datapoints".format(len(bc_df.index)))
    return bc_df


@timed_call
def load_modis(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    """
        Load MODIS data from a netCDF file and transform it
        into a format that is ingestable by the pyRoutePlanner

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retrieved
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
    logging.debug("opening file {}".format(params['file']))
    modis = xr.open_dataset(params['file'])
    modis_df = modis.to_dataframe()
    modis_df = modis_df.reset_index()

    # MODIS Sea Ice Concentration data is partially obscured by cloud cover.
    # Where a data point indicates that there is cloud cover above it,
    # set the SIC of that datapoint to NaN
    modis_df['iceArea'] = np.where(modis_df['cloud'] == 1, np.NaN, modis_df['iceArea'])
    modis_df = modis_df.rename(columns={'iceArea': 'SIC'})
    modis_df['SIC'] = modis_df['SIC']*10.

    modis_df = modis_df[modis_df['long'].between(long_min, long_max)]
    modis_df = modis_df[modis_df['lat'].between(lat_min, lat_max)]

    logging.debug("returned {} datapoints".format(len(modis_df.index)))
    return modis_df


@timed_call
def load_era5_wind(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    """
        Load era5 wind data from a netCDF file and transform it
        into a format that is ingestable by the pyRoutePlanner

        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retrieved
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
    logging.debug("opening file {}".format(params['file']))
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

    logging.debug("returned {} datapoints".format(len(era5_wind.index)))
    return era5_wind_df


@timed_call
def load_north_sea_currents(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    """
        Args:
            long_min (float): The minimum longitude of the data to be retrieved
            long_max (float): The maximum longitude of the data to be retrieved
            lat_min (float): The minimum latitude of the data to be retrieved
            lat_max (float): The maximum latitude of the data to be retrieved
            time_start (string): The start time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"
            time_end (string): The end time of the data to be retrieved,
                must be given in the format "YYYY-MM-DD"

            params (dict): A dictionary containing optional parameters. This
                function requires -

                params['file'] (string): file location of the Baltic current dataset

        Returns:
            bc_df (Dataframe): A dataframe containing Baltic Sea current
                data. The dataframe is of the format -

                lat | long | time | uC | vC
    """
    logging.debug("opening file {}".format(params['file']))
    bc = xr.open_dataset(params['file'])
    bc_df = bc.to_dataframe()

    bc_df = bc_df.reset_index()
    bc_df = bc_df[['lat', 'lon', 'U', 'V']]
    bc_df = bc_df.rename(columns={'lon': 'long', 'lat': 'lat', 'U': 'uC', 'V': 'vC'})
    logging.debug("returned {} datapoints".format(len(bc_df.index)))
    return bc_df


@timed_call
def load_oras5(params, long_min, long_max, lat_min, lat_max, time_start, time_end):
    logging.debug("opening file {}".format(params['file']))
    oras5 = xr.open_dataset(params['file'])
    oras5 = oras5.sel(time=slice(time_start, time_end))
    oras5_df = oras5.to_dataframe()
    oras5_df = oras5_df.reset_index()

    oras5_df = oras5_df.rename(columns={'uo': 'uC','vo': 'vC','longitude':'long','latitude':'lat'})
    oras5_df = oras5_df[oras5_df['long'].between(long_min, long_max)]
    oras5_df = oras5_df[oras5_df['lat'].between(lat_min, lat_max)]
    logging.debug("returned {} datapoints".format(len(oras5_df.index)))
    return oras5_df
