.. _abstract-scalar-dataloader-index:

******************
Scalar Dataloaders
******************



^^^^^^^^^^^^^^^^^^^^^^^^^^
Abstract Scalar Base Class
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1
   :glob:

   ./abstractScalar

The Abstract Base Class of the scalar dataloaders holds most of the 
functionality that would be needed to manipulate the data to work 
with the mesh. When creating a new dataloader, the user must define
how to open the data files, and what methods are required to manipulate
the data into a standard format. More details are provided on the 
:ref:`abstractScalar doc page<abstract-scalar-dataloader>`

^^^^^^^^^^^^^^^^^^^^^^^^^^
Scalar Dataloader Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^
Data must be imported and saved as an xarray.Dataset, or a pandas.DataFrame object.
Below is a simple example of how to load in a NetCDF file::
    
    from polar_route.Dataloaders.Scalar.AbstractScalar import ScalarDataLoader
    import xarray as xr
    import logging

    class MyDataLoader(ScalarDataLoader):
        def __init__(self, bounds, params):
            # Creates a class attribute for all keys in params
            for key, val in params.items():
                logging.debug(
                    f"Reading in {key}:{value} (dtype={type(value)}) from params"
                    )
                setattr(self, key, val)
            
            # Import data from file
            self.data = self.import_data(bounds)

            # Retrieve data name from variable name in NetCDF
            self.data_name = self.get_data_col_name()

            logging.info(f"Successfully loaded {self.data_name} from {self.file}")
            
        def import_data(self, bounds):
            logging.debug("Importing my data...")
            # Open Dataset
            logging.debug(f"- Opening file {self.file}")
            data = xr.open_dataset(self.file)

            # Rename coordinate columns to 'lat', 'long', 'time' if they aren't already
            data = data.rename({'lon':'long'})

            # Limit to initial boundary
            data = data.sel(lat=slice(bounds.get_lat_min(),bounds.get_lat_max()))
            data = data.sel(long=slice(bounds.get_long_min(),bounds.get_long_max()))
            data = data.sel(time=slice(bounds.get_time_min(),bounds.get_time_max()))

            return data


Sometimes the data needs to be reprojected if it is not initially in mercator 
projection. It may also need to be downsampled if the dataset is very large. 
The following code handles both of these cases::

    class MyDataLoader(ScalarDataLoader):
        def __init__(self, bounds, params):
            # Creates a class attribute for all keys in params
            for key, val in params.items():
                logging.debug(
                    f"Reading in {key}:{value} (dtype={type(value)}) from config params"
                    )
                setattr(self, key, val)
            
            # Import data from file
            self.data = self.import_data(bounds)
            # Downsampling data by 'downsample_factors' defined in config params
            self.data = self.downsample()
            # Reprojecting dataset from EPSG:3412 to 'EPSG:4326'. 
            # Coordinate names 'x', 'y' will be replaced with 'long', 'lat' 
            self.data = self.reproject( in_proj  = 'EPSG:3412',
                                        out_proj = 'EPSG:4326',
                                        x_col    = 'x',
                                        y_col    = 'y')

            # Limit to initial boundary. 
            # Note: Reprojection converts data to pandas dataframe
            idx = self.get_datapoints(bounds).index
            self.data = self.data.loc[idx]
            
            # Manually overwriting data name
            self.data_name = "my_variable"
            self.data = self.set_data_col_name(self.data_name)

            logging.info(f"Successfully loaded {self.data_name} from {self.file}")
            
        def import_data(self, bounds):
            logging.debug("Importing my data...")
            
            # Open Dataset
            logging.debug(f"- Opening file {self.file}")
            data = xr.open_dataset(self.file)

            # Can't easily determine bounds of data in wrong projection, so skipping for now
            return data

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Implemented Scalar Dataloaders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :glob:

   ./implemented/*