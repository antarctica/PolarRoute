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
:ref:`abstractScalar doc page<abstract-scalar-dataloader>`.

^^^^^^^^^^^^^^^^^^^^^^^^^^
Scalar Dataloader Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^
Data must be imported and saved as an xarray.Dataset, or a pandas.DataFrame object.
Below is a simple example of how to load in a NetCDF file::
    
    from polar_route.Dataloaders.Scalar.AbstractScalar import ScalarDataLoader
    import xarray as xr
    import logging

    class MyDataLoader(ScalarDataLoader):
      
        def import_data(self, bounds):
            logging.debug("Importing my data...")
            # Open Dataset
            data = xr.open_mfdataset(self.files)

            # Rename coordinate columns to 'lat', 'long', 'time' if they aren't already
            data = data.rename({'lon':'long'})

            # Limit to initial boundary
            data = self.trim_data(bounds, data=data)

            return data


Sometimes there are parameters that are constant for a data source, but are not 
constant for all data sources. Default values may be defined either in the
dataloader factory, or within the dataloader itself. Below is an example of 
setting default parameters for reprojection of a dataset::

    class MyDataLoader(ScalarDataLoader):
        def add_params(self, params):
            # Define projection of dataset being imported
            params['in_proj'] = 'EPSG:3412'
            # Define projection required by output 
            params['out_proj'] = 'EPSG:4326' # default is EPSG:4326, so strictly
                                             # speaking this line is not necessary

            # Coordinates in dataset that will be reprojected into long/lat
            params['x_col'] = 'x' # Becomes 'long'
            params['y_col'] = 'y' # Becomes 'lat'
            
        def import_data(self, bounds):
            # Open Dataset
            data = xr.open_mfdataset(self.files)

            # Can't easily determine bounds of data in wrong projection, so skipping for now
            return data

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Implemented Scalar Dataloaders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :glob:

   ./implemented/*