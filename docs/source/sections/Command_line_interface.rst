###############################
Command Line Interface
###############################

The PolarRoute package provides 4 CLI entry points, intended to be used in succession to plan a route through a digital environment.

.. figure:: ./Figures/PolarRoute_CLI.png
   :align: center
   :width: 700

   *Overview figure of the Command Line Interface entry points of PolarRoute*

^^^^^^^^^^^
add_vehicle
^^^^^^^^^^^
The *add_vehicle* command allows vehicle specific simulations to be performed on the digital environment. This vehicle specific
information is then encoded into the digital environment file.

::

    add_vehicle <vessel_config.json> <mesh.json>

positional arguments:

::

    vessel_config : A configuration file detailing the vessel to be simulated in the digital environment.
    mesh : A digital environment file.

The format for the required *<vessel.json>* file can be found in the :ref:`configuration - vessel performance modeller` section of the documentation.
The required *<mesh.json>* file can be created using the *create_mesh* command from the `MeshiPhi <https://github.com/antarctica/MeshiPhi>`_ package.

optional arguments are

::

    -v (verbose logging)
    -o <output location> (set output location for mesh)

The format of the return Vessel_Mesh.json file is explain in :ref:`the vessel_mesh.json file` section of the documentation.

^^^^^^^^^^^^^^^
optimise_routes
^^^^^^^^^^^^^^^
Optimal routes through a mesh can be calculated using the command:

::

    optimise_routes <route_config.json> <vessel_mesh.json> <waypoints.csv>

positional parameters:

::

    vessel_mesh : A digital environment file with added vessel specific simulations.
    route_config : A configuration file detailing optimisation parameters to be used when route planning.
    waypoints: A .csv file containing waypoints to be travelled between.


The format for the required *<route_config.json>* file can be found in the :ref:`configuration - route planning` section of the documentation.
The required *<vessel_mesh.json>* file can be generated using the :ref:`add_vehicle` command shown above.
The format for the required *<waypoints.csv>* file is as follows:

As table:

+------------------+---------------+---------------+---------+---------------+
| Name             | Lat           | Long          | Source  | Destination   |
+==================+===============+===============+=========+===============+
| Halley           | -75.26722     | -27.21694     |         | X             |
+------------------+---------------+---------------+---------+---------------+
| Rothera          | -68.3892      | -95.2436      |         |               |
+------------------+---------------+---------------+---------+---------------+
| South Georgia    | -54.87916667  | -37.26416667  | X       |               |
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
    South Georgia,-54.87916667,-37.26416667,X,
    Falklands,-55.63472222,-64.88,,
    Elephant Island,-60.54722222,-55.18138889,,

Additional waypoints may be added by extending the '<waypoints.csv>' file. Which waypoints are navigated between is determined by 
added a **X** in either the *Source* or *Destination* columns. When processed, the route planner will create routes from all 
waypoints marked with an **X** in the source column to all waypoints marked with a **X** in the *destination* column. 

optional arguments are

::

    -v (verbose logging)
    -o <output location> (set output location for mesh)
    -p (output only the calculated path, not the entire mesh)
    -d (output Dijkstra path as well as smoothed path)


The format of the returned *<route.json>* file is explained in :ref:`the route.json file` section of the documentation.

^^^^^^^^^^^^^^^
calculate_route
^^^^^^^^^^^^^^^
The cost of a user-defined route through a pre-generated mesh containing vehicle information can be calculated using the command:

::

    calculate_route <vessel_mesh.json> <route>

positional parameters:

::

    vessel_mesh : A digital environment file with added vessel specific simulations.
    route : A route file containing waypoints on a user-defined path.

optional arguments:

::

    -v : verbose logging
    -o : output location

Running this command will calculate the cost of a route between a set of waypoints provided in either csv or geojson
format. The route is assumed to travel from waypoint to waypoint in the order they are given, following a rhumb line.
The format of the output *<route.json>* file is identical to that from the :ref:`optimise_routes` command.
This is explained in :ref:`the route.json file` section of the documentation. The time and fuel cost of the route will
also be logged out once the route file has been generated. If the user-defined route crosses a cell in the mesh that is
considered inaccessible to the vessel then a warning will be displayed and no route will be saved.

^^^^^^^^
Plotting
^^^^^^^^
Meshes produced at any stage in the route planning process can be visualised using the GeoPlot 
library found at the relevant `GitHub page <https://github.com/antarctica/GeoPlot>`_. Meshes and routes can also be
plotted in other GIS software such as QGIS by exporting the mesh to a common format such as .geojson or .tif using
the :ref:`export_mesh` command.
