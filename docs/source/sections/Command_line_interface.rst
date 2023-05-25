###############################
Command Line Interface
###############################

The PolarRoute package provides 4 CLI entry points, intended to be used in succession to plan a route through a digital enviroment.

.. figure:: ./Figures/PolarRoute_CLI.png
   :align: center
   :width: 700

   *Overview figure of the Command Line Interface entry points of PolarRoute*

^^^^^^^^^^^^^^^^^^
create_mesh
^^^^^^^^^^^^^^^^^^
The *create_mesh* entry point builds a digital enviroment file from a collection of source data, which can then be used 
by the vessel performance modeller and route planner. 

::

    create_mesh <config.json>

positional arguments:

::

    config : A configuration file detailing how to build the digital enviroment. JSON parsable

The format of the required *<config.json>* file can be found in the :ref:`configuration - mesh construction` section of the documentation.

optional arguments:

::

    -v (verbose logging)
    -o <output location> (set output location for mesh)


The format of the returned mesh.json file is explain in :ref:`the mesh.json file` section of the documentation.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
add_vehicle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The *add_vehicle* command allows vehicle specific simulations to be performed on the digital enviroment. This vehicle specific 
information is then encoded into the digital enviroment file.

::

    add_vehicle <vessel.json> <mesh.json>

positional arguments:

::

    vessel : A configuration file detailing the vessel to be simulated in the digital enviroment.
    mesh : A digital enviroment file.

The format for the required *<vessel.json>* file can be found in the :ref:`configuration - vessel performance modeller` section of the documentation.
The required *<mesh.json>* file can be created using the :ref:`create_mesh` command shown above.

optional arguments are

::

    -v (verbose logging)
    -o <output location> (set output location for mesh)

The format of the return Vessel_Mesh.json file is explain in :ref:`the vessel_mesh.json file` section of the documentation.

^^^^^^^^^^^^^^^^^^
optimise_routes
^^^^^^^^^^^^^^^^^^
Optimal routes through a mesh can be calculated using the command:

::

    optimise_routes <vessel_mesh.json> <route_config.json> <waypoints.csv>

positional parameters:

::

    vessel_mesh : A digital enviroment file with added vessel specific simulations.
    route_config : A configuration file detailing optimisation parameters to be used when route planning.
    waypoints: A .csv file containing waypoints to be travelled between.


The format for the required *<route_config.json>* file can be found in the :ref:`configuration - route planning` section of the documentation.
The required *<vessel_mesh.json>* file can be generated using the :ref:`add_vehicle` command shown above.
The format for the requried *<waypoints.csv>* file is as follows:

As table:

+------------------+---------------+---------------+---------+---------------+
| Name             | Lat           | Long          | Source  | Destination   |
+==================+===============+===============+=========+===============+
| Halley           | -75.26722     | -27.21694     |         | X             |
+------------------+---------------+---------------+---------+---------------+
| Rothera          | -68.3892      | -95.2436      |         |               |
+------------------+---------------+---------------+---------+---------------+
| South Georiga    | -54.87916667  | -37.26416667  | X       |               |
+------------------+---------------+---------------+---------+---------------+
| Falklands        | -55.63472222  | -64.88        |         |               |
+------------------+---------------+---------------+---------+---------------+
| Elephant Island  | -60.54722222  | -55.18138889  |         |               |
+------------------+---------------+---------------+---------+---------------+

As .csv:

::

    Name,Lat,Long,Source,Destination
    Halley,-75.26722,-27.21694,,X
    Rothera,-68.3892,-95.2436,,
    South Georiga,-54.87916667,-37.26416667,X,
    Falklands,-55.63472222,-64.88,,
    Elephant Island,-60.54722222,-55.18138889,,

Additional waypoints may be added by extending the '<waypoints.csv>' file. Which waypoints are navigated between is determined by 
added a **X** in either the *Source* or *Destination* columns. When processed, the route planner will create routes from all 
waypoints marked with an **X** in the source column to all waypoints marked with a **X** in the *destination* column. 

optional arguments are

::

    -v (verbose logging)
    -o <output location> (set output location for mesh)
    -p (output only the caculated path, not the entire mesh)
    -d (output Dijkstra path as well as smoothed path)


The format of the returned *<route.json>* file is explain in :ref:`the route.json file` section of the documentation.

^^^^^^^^^^^^^^^^^^
export_mesh
^^^^^^^^^^^^^^^^^^
Once a mesh has been built using the :ref:`create_mesh` command, it can be exported other file types for 
use in other systems (such as GIS software) using the the *export_mesh* command.

::

    export_mesh <mesh.json> <output_location> <output_format> 

positional arguments:

::

    mesh : A digital enviroment file.
    output_location : The location to save the exported mesh.
    output_format : The format to export the mesh to.


supported output formats are:
  * .json (default) [JSON]
  * geo.json (collection of polygons for each cell in the mesh) [GEOJSON]
  * .tif (rasterised mesh) [TIF]

optional arguments:

::

    -v : verbose logging
    -o : output location
    -format_conf: configuration file for output format (required for TIF export)

the format of the *<format_conf.json>* file required for .tif export is as follows:

::

    {
        "data_name": "elevation",
        "sampling_resolution": [
            150,
            150
        ],
        "projection": "3031",
        "color_conf": "path to/color_conf.txt"
    }

where the variable are as follows:
* **data_name** : The name of the data to be exported. This is the name of the data layer in the mesh.
* **sampling_resolution** : The resolution of the exported mesh. This is a list of two values, the first being the x resolution and the second being the y resolution.
* **projection** : The projection of the exported mesh. This is a string of the EPSG code of the projection.
* **color_conf** : The path to the color configuration file. This is a text file containing the color scheme to be used when exporting the mesh. The format of this file is as follows:
                                    
::

    0 240 250 160  
    30 230 220 170  
    60 220 220 220 
    100 250 250 250 

The color_conf.txt contains 4 columns per line: the data_name value and the 
corresponding red, green, blue value between 0 and 255.

^^^^^^^^^^^^^^^^^^
Plotting
^^^^^^^^^^^^^^^^^^
Meshes produced at any stage in the route planning process can be visualised using the GeoPlot 
library found at `Link <https://github.com/antarctica/GeoPlot>`. Meshes and routes can also be plotted in 
other GIS software such as QGIS my exporting the mesh the a common format such as .geojson or .tif using 
the :ref:`export_mesh` command.
