.. _dataloaders-overview:

*******************
Dataloader Overview
*******************

.. toctree::
   :maxdepth: 1
   :glob:

   ./DataLoaderInterface
   ./Factory
   ./lut/index
   ./scalar/index
   ./vector/index


Overview
########

In this section we discuss functions for loading data into the PolarRoute Mesh.
Data can be added to the Mesh using the *.add_data_points()* function of the Mesh.
This function takes a single argument, a dataframe containing datapoints in a
EPSG:4326 projection. The format in which dataframe must be given is:

+-----+------+------+---------+-----+---------+
| Lat | Long | Time | value_1 | ... | value_n |
+=====+======+======+=========+=====+=========+
| ... | ...  | ...  | ...     | ... | ...     |
+-----+------+------+---------+-----+---------+
|     |      |      |         |     |         |
+-----+------+------+---------+-----+---------+

The data loaders provided collect or create data from heterogeneous data sources
and transform them into the correct format for use by the *.add_data_points()*
function of the Mesh.

^^^^^^^^^^^^^^^^
Dataloader Types
^^^^^^^^^^^^^^^^

Scalar dataloaders are ...

Vector Dataloaders are ...

Look-up Table Dataloaders are...



^^^^^^^^^^^^^^^^^^^^^
Abstract Data Loaders
^^^^^^^^^^^^^^^^^^^^^
These are the templates to be used when implementing new dataloaders into PolarRoute. 
They have been split into three seperate categories: Scalar, Vector, and LUT.

The scalar dataloader is to be used on scalar datasets; i.e. datasets with a single value
per latitude/longitude(/time) coordinate. An example of this would be bathymetry, where there
is a single sea depth per coordinate.

The vector dataloader is to be used on vector datasets; i.e. datasets with multiple values
per latitude/longitude(/time) coordinate. An example of this would be sea currents, where there
is are both x and y components to the current at any given location.

The look-up table (LUT) dataloader is to be used on datasets where boundaries define a value.
Real data is always prefered to this method, however in the case where there is no data, the LUT
can provide an alternative. An example of this would be ice density, where the values are defined
by a paper ####REF####, and are defined by ocean or sea (i.e. with location boundaries), and per
season.






.. .. autoclass:: polar_route.AbstractDataLoaders
..     :members:


.. Implemented Data Loaders
.. ##############
.. Below is a list of all the dataloaders currently implemented in PolarRoute. 

.. .. autoclass:: polar_route.DataLoaders
..     :members:
