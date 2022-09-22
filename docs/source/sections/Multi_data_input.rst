********
Data Loaders
********

Overview
##############

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

Data loader functions located in the file **./polar_route/data_loaders.py** can
be referenced in a configuration file and called in the initialisation of the Mesh.
See the :ref:`Configuration` section of this document for further details. 



.. automodule:: polar_route.data_loaders
    :members:
