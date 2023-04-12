.. _abstract-vector-dataloader-index:

******************
Vector Dataloaders
******************


^^^^^^^^^^^^^^^^^^^^^^^^^^
Abstract Vector Base Class
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1
   :glob:

   ./abstractVector

The Abstract Base Class of the vector dataloaders holds most of the 
functionality that would be needed to manipulate the data to work 
with the mesh. When creating a new dataloader, the user must define
how to open the data files, and what methods are required to manipulate
the data into a standard format. More details are provided on the 
:ref:`abstractVector doc page<abstract-vector-dataloader>`


^^^^^^^^^^^^^^^^^^^^^^^^^^
Vector Dataloader Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^
Creating a vector dataloader is almost identical to creating a 
:ref:`scalar dataloader<abstract-scalar-dataloader>`. The key differences 
are that the `VectorDataLoader` abstract base class must be used, and that
the `data_name` is a comma seperated string of the vector component names.
e.g. a dataloader storing a vector with column names :code:`uC` and 
:code:`vC` will have an attribute :code:`self.data_name = 'uC,vC'`
Data must be imported and saved as an xarray.Dataset, or a 
pandas.DataFrame object. Below is a simple example of how to load in a 
NetCDF file::
    
    from polar_route.Dataloaders.Scalar.AbstractScalar import VectorDataLoader
    import xarray as xr
    import logging

    class MyDataLoader(VectorDataLoader):
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


Similar to scalar data loaders, sometimes the data needs to be reprojected 
if it is not initially in mercator projection. It may also need to be downsampled 
if the dataset is very large. The following code handles both of these cases::

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
            
            # Manually overwriting data names for each vector component
            self.data_name = "v_x,v_y"
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
Implemented Vector Dataloaders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :glob:

   ./implemented/*