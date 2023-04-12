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
        def import_data(self, bounds):
            logging.debug("Importing my data...")
            # Open Dataset
            logging.debug(f"- Opening file {self.file}")
            data = xr.open_dataset(self.file)

            # Rename coordinate columns to 'lat', 'long', 'time' if they aren't already
            data = data.rename({'lon':'long'})

            # Limit to initial boundary
            data = self.trim_data(bounds, data=data)

            return data


Similar to scalar data loaders, sometimes there are parameters that are constant 
for a data source, but are not constant for all data sources. Default values may 
be defined either in the dataloader factory, or within the dataloader itself. 
Below is an example of setting default parameters for reprojection of a dataset::

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
            data = xr.open_mfdataset(self.file)

            # Can't easily determine bounds of data in wrong projection, so skipping for now
            return data

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Implemented Vector Dataloaders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :glob:

   ./implemented/*