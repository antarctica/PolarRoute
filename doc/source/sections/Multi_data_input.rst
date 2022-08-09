********
Data Loaders
********

Overview
##############

In this section we discuss functions for loading data into the polar_route Mesh.
Can be added to the Mesh using the *.add_data_points()* function of the Mesh.
This function takes a single argument, a dataframe containing datapoints in a
EPSG:4326 projection. The format of that dataframe must be given is:

+-----+------+------+---------+-----+---------+
| Lat | Long | Time | value_1 | ... | value_n |
+=====+======+======+=========+=====+=========+
| ... | ...  | ...  | ...     | ... | ...     |
+-----+------+------+---------+-----+---------+
|     |      |      |         |     |         |
+-----+------+------+---------+-----+---------+

The data-loaders provoided collect or create data from hetrogenous data sources
and tranform them into the correct format for use by the *.add_data_points()* 
fuction of the Mesh. 

data-loader functions location in the file **./polar_route/data_loaders.py** can
be referenced in a configuation file and called in the initialsation of the Mesh.
See the :ref:`Configuration` section of this document for further details. 



.. automodule:: polar_route.data_loaders
    :members:
