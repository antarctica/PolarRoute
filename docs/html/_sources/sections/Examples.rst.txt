###############################
Command Line Interface Examples
###############################

The CLI provides multiple entry-points through which the PolarRoute package can be used. Each command is described in the 
:ref:`Command Line Interface <cli>` section of these docs.

To summarise, the basic process is to create an environment mesh, add vehicle performance characteristics to each
cell in that mesh and then find an optimal route between waypoints located within that mesh. At any stage, `GeoPlot <https://github.com/antarctica/GeoPlot>`_
can be used to visualise the outputs.

::

    # From MeshiPhi
    create_mesh <mesh_config_file> -o <mesh_output_file>
    
    # From PolarRoute
    add_vehicle <vessel_config_file> <mesh_output_file> -o <vessel_output_file>
    optimise_routes <route_config_file> <vessel_output_file> <waypoints_file> -o <route_output_file>
    
    # From GeoPlot
    plot_mesh <any_output_file> -o <output_file>


Above are the commands to run in order to fulfill this process. If you have successfully installed PolarRoute and would
like to try it out, :download:`here<https://raw.githubusercontent.com/antarctica/PolarRoute/main/docs/source/sections/Examples/example_3.zip>`
is some example data which you can use. Simply extract the configs out of the zip archive, and run the commands on the
appropriate files. To map the commands to the files in the zip archive:

* :code:`<mesh_config_file>` is called :code:`grf_example.config.json`
* :code:`<vessel_config_file>` is called :code:`ship.config.json`
* :code:`<route_config_file>` is called :code:`traveltime.config.json`
* :code:`<waypoints_file>` is called :code:`waypoints_example.csv`

.. note::
    By default the :code:`plot_mesh` command will plot a basemap showing the location of your mesh on Earth.
    When working with entirely synthetic data, e.g. when running any of the GRF examples below, the spatial coordinates
    used do not correspond to a real location and we recommend running :code:`plot_mesh` with the :code:`-b` option to
    disable this basemap.

Several notebooks have been created that will guide you through each stage in using PolarRoute, from mesh creation
through to route planning. These notebooks are available via Google Colab and use the CLI entry-points to show how
someone would typically interact with PolarRoute through the terminal.

^^^^^^^^^^^^^^^^^^
Empty Mesh Example 
^^^^^^^^^^^^^^^^^^
Here we provide two examples of empty meshes that are simple to process to get you started. Since these are empty meshes,
we expect the optimal calculated route to be a straight line between two waypoints. Over long distances this is seen as
a great circle arc on the mercator projection that GeoPlot uses to display the mesh.

* :download:`Uniform Mesh<https://raw.githubusercontent.com/antarctica/PolarRoute/main/docs/source/sections/Examples/example_1.zip>`
* :download:`Non-Uniform Mesh<https://raw.githubusercontent.com/antarctica/PolarRoute/main/docs/source/sections/Examples/example_2.zip>`
* `See on Google Colab <https://colab.research.google.com/drive/1N1mxOy2oX7bEGtPy7Ztshrs4Fs_7lBpV?usp=sharing>`_

^^^^^^^^^^^^^^^^^^^^^^
Synthetic Data Example 
^^^^^^^^^^^^^^^^^^^^^^
In this example, we provide synthetic data in the form of Gaussian Random Fields (GRFs), which provide a random, yet somewhat
realistic representation of real-world features such as bathymetry. Here we walk through every step involved in PolarRoute, 
from creating the mesh through to optimising a route within it.

* :download:`Gaussian Random Field data<https://raw.githubusercontent.com/antarctica/PolarRoute/main/docs/source/sections/Examples/example_3.zip>`
* `Synthetic Data Example <https://colab.research.google.com/drive/1BOzTyBjpCbAJ6PMJi0GS55shuaMu72h5?usp=sharing>`_

^^^^^^^^^^^^^^^^^
Real Data Example 
^^^^^^^^^^^^^^^^^
Real world data has been used to generate these meshes around the coast of Antarctica. This data is publicly available,
however is not included here to avoid violating data sharing policies. Instead, we provide a mesh file after the 'create_mesh' stage 
since that is a derived product. The data files used to construct the mesh can be seen in the :code:`data_sources` field of
the config contained within the provided mesh. See `Dataloaders <https://antarctica.github.io/MeshiPhi/html/sections/Dataloaders/overview.html>`_
in the MeshiPhi docs for more info on each source of data that PolarRoute currently supports.

* :download:`Real-world data 1<https://raw.githubusercontent.com/antarctica/PolarRoute/main/docs/source/sections/Examples/example_4.zip>`
* :download:`Real-world data 2<https://raw.githubusercontent.com/antarctica/PolarRoute/main/docs/source/sections/Examples/example_6.zip>`
* `Real Data Example <https://colab.research.google.com/drive/1atTQFk4eK_SKImHofmEXIfoN9oAP1cJb?usp=sharing>`_

###############
Python Examples
###############

Route planning may also be done in a python interpreter. In this case, the CLI is not required but the steps required for route planning
follow the same format - create a digital environment; simulated a vessel against it; optimise a route plan through the digital environment.
To perform the steps detailed in this section, a mesh must first be generated using `MeshiPhi <https://github.com/antarctica/MeshiPhi>`_.

The files used in the following example are those used in the synthetic example from the notebook section above. Download them
:download:`here<https://raw.githubusercontent.com/antarctica/PolarRoute/main/docs/source/sections/Examples/example_3.zip>`.
 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Creating the digital environment.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A configuration file is needed to initialise the **EnvironmentMesh** object which forms the digital environment. This configuration file
is of the same format used in the :ref:`create_mesh` CLI entry-point, and may either be loaded from a *json* file or constructed 
within a python interpreter.

Loading configuration from *json* file:
::

    import json
    # Read in config file
    with open('/path/to/grf_example.config.json', 'r') as f:
        config = json.load(f)    


The **EnvironmentMesh** object can then be initialised. This mesh object will be constructed using the parameters in its
configuration file. This mesh object can then be manipulated further, such as increasing its resolution through further
splitting, adding additional data sources or altering its configuration parameters. See the relevant section of the `MeshiPhi docs <https://antarctica.github.io/MeshiPhi/html/sections/Configuration/Mesh_construction_config.html>`_
for a more in-depth explanation. The **EnvironmentMesh** object can then be cast to a json object and saved to a file.
::

    from meshiphi.mesh_generation.mesh_builder import MeshBuilder

    # Create mesh from config
    cg = MeshBuilder(config).build_environmental_mesh()
    mesh = cg.to_json()

    # Save output file
    with open('/path/to/grf_example.mesh.json', 'w+') as f:
        config = json.dump(mesh, f, indent=4)    

.. note::
    We are saving the file after each stage, but if you are running the code snippets 
    back to back, there is no need to save the json output and then load it in again. 
    Just pass the dictionary created from the :code:`to_json()` call into the next function


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Simulating a Vessel in a Digital Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a digital environment **EnvironmentMesh** object has been created with `MeshiPhi <https://github.com/antarctica/MeshiPhi>`_,
a vessel's performance when travelling within it may be simulated. The **VesselPerformanceModeller** object requires a
digital environment in *json* format and vessel specific configuration parameters, also in *json* format. These may either
be loaded from a file, or created within any python interpreter.

Loading mesh and vessel from *json* files:
::

    # Loading digital environment from file
    with open('/path/to/grf_example.mesh.json', 'r') as f:
        mesh = json.load(f)  

    # Loading vessel configuration parameters from file
    with open('/path/to/ship.json', 'r') as f:
        vessel = json.load(f) 

The **VesselPerformanceModeller** object can then be initialised. This can be used to simulate the performance of the
vessel and encode this information into the digital environment.
::

   from polar_route.vessel_performance.vessel_performance_modeller import VesselPerformanceModeller
   vp = VesselPerformance(mesh, vessel)
   vp.model_accessibility() # Method to determine any inaccessible areas, e.g. land
   vp.model_performance() # Method to determine the performance of the vessel in accessible regions, e.g speed or fuel consumption

The **VesselPerformanceModeller** object can then be cast to a json object and saved to a file. This *vessel_mesh.json* file can then
be used by the CLI entry-point :ref:`optimise_routes`, or the json object can be passed to the **RoutePlanner** object in a python
console.
::

    vessel_mesh = vp.to_json()
    # Save to output file
    with open('/path/to/grf_example.vessel.json', 'w+') as f:
        json.dump(vessel_mesh, f, indent=4)

^^^^^^^^^^^^^^^^^^
Route Optimisation
^^^^^^^^^^^^^^^^^^
Now that the vessel dependent environmental mesh is defined, and represented in the **VesselPerformanceModeller** object,
we can construct routes, with parameters defined by the user in the :ref:`route config file <route config>`.

Waypoints are passed as an input file path, `waypoints.csv`, discussed more in the Inputs section of the manual pages.
The route construction is performed in two stages: construction of the meshed dijkstra optimal routes, using
`.compute_routes()`, and the smoothing of the dijkstra routes to further optimise the solution and reduce mesh
dependency, using `.compute_smooth_routes()`. During the execution of `.compute_routes()` the paths are stored as an
attribute of the **RoutePlanner** object under `paths`. These are then replaced by the smoothed paths after running
`.compute_smooth_routes()`. An additional entry `waypoints` is generated to store the waypoints information used in
route construction. For further details about the structure of the outputs of the route planner please see the
:ref:`Outputs` section of this documentation.

::

    from polar_route.route_planner import RoutePlanner
    rp = RoutePlanner('/path/to/grf_example.vessel.json', 
                      '/path/to/traveltime.config.json', 
                      '/path/to/waypoints_example.csv')
    # Calculate optimal dijkstra path between waypoints
    rp.compute_routes()
    # Smooth the dijkstra routes
    rp.compute_smoothed_routes()

    route_mesh = rp.to_json()
    # Save to output file
    with open('/path/to/grf_example.route.json', 'w+') as f:
        json.dump(route_mesh, f, indent=4)


^^^^^^^^^^^^^^^^^^^
Visualising Outputs
^^^^^^^^^^^^^^^^^^^

The **EnvironmentMesh** object can be visualised using the GeoPlot package, also developed by BAS. This package is not
included in the distribution of PolarRoute, but can be installed using the following command:

:: 

    pip install bas_geoplot

GeoPlot can then be used to visualise the **EnvironmentMesh** object using the following code in an iPython notebook or
any python interpreter:

::
    
    from bas_geoplot.interactive import Map

    mesh = pd.DataFrame(mesh_json['cellboxes'])
    mp = Map(title="GRF Example")

    mp.Maps(mesh, 'MeshGrid', predefined='cx')
    mp.Maps(mesh, 'SIC', predefined='SIC')
    mp.Maps(mesh, 'Elevation', predefined='Elev', show=False)
    mp.Vectors(mesh,'Currents', show=False, predefined='Currents')
    mp.Vectors(mesh, 'Winds', predefined='Winds', show=False)

    mp.show()
