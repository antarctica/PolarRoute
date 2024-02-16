######################
Command Line Interface
######################

The CLI provides multiple entry-points through which the PolarRoute package can be used. Each command is described in the 
:ref:`Command Line Interface <cli>` section of these docs.

Several notebooks have been created that will guide you through each stage in PolarRoute, from mesh creation through to route planning. 
These notebooks use the CLI entry-points to show how someone would typically interact with PolarRoute through the terminal.

^^^^^^^^^^^^^^^^^^
Empty Mesh Example 
^^^^^^^^^^^^^^^^^^
Here we provide two examples of empty meshes that are simple to process to get you started. Since these are empty meshes,
we expect the optimal calculated route to be a straight line between two waypoints, which is seen as a great circle arc on
the mercator projection that GeoPlot provides. 
* :download:`Uniform Mesh<.Examples/example_1.zip>`
* :download:`Non-Uniform Mesh<.Examples/example_2.zip>`
* `See on Google Colab <https://colab.research.google.com/drive/1N1mxOy2oX7bEGtPy7Ztshrs4Fs_7lBpV?usp=sharing>`_

^^^^^^^^^^^^^^^^^^^^^^
Synthetic Data Example 
^^^^^^^^^^^^^^^^^^^^^^
In this example, we provide synthetic data in the form of Gaussian Random Fields, which provide a random, yet somewhat
realistic representation of real-world features such as bathymetry. Here we walk through every step involved in PolarRoute, 
from creating the mesh through to optimising a route through it. 
* :download:`Gaussian Random Field data<.Examples/example_3.zip>`
* `Synthetic Data Example <https://colab.research.google.com/drive/1BOzTyBjpCbAJ6PMJi0GS55shuaMu72h5?usp=sharing>`_

^^^^^^^^^^^^^^^^^
Real Data Example 
^^^^^^^^^^^^^^^^^
Real world data has been used to generate these meshes around the coast of Antarctica. This data is publically available,
however is not included here to avoid violating data sharing policies. Instead, we provide a mesh file after the 'create_mesh' stage 
since that is a derived product. See `Dataloaders <https://antarctica.github.io/MeshiPhi/html/sections/Dataloaders/overview.html>`_ 
in the MeshiPhi docs for more info on each source of data that PolarRoute currently supports.

* :download:`Real-world data 1<.Examples/example_4.zip>`
* :download:`Real-world data 2<.Examples/example_5.zip>`
* `Real Data Example <https://colab.research.google.com/drive/1atTQFk4eK_SKImHofmEXIfoN9oAP1cJb?usp=sharing>`_

######
Python
######

Route planning may also be done using a python terminal. In this case, the CLI is not required but the steps required for route planning 
follow the same format - create a digital environment; simulated a vessel against it; optimise a route plan through the digital environment.
To perform the steps detailed in this section, a mesh must first be generated using `MeshiPhi <https://github.com/antarctica/MeshiPhi>`_.

The files used in the following example are those used in the synthetic example from the notebook section above. Download them
:download:`here<.Examples/example_3.zip>`.
 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Creating the digital environment.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A configuration file is needed to initialise the **`Mesh`** object which forms the digital environment. This configuration file
is of the same format used in the :ref:`create_mesh` CLI entry-point, and may either be loaded from a *json* file or constructed 
within the python terminal.

Loading configuration from *json* file:
::

    import json
    with open('examples/environment_config/grf_example.config.json', 'r') as f:
        config = json.load(f)    


The digital environment **`Mesh`** object can then be initialised. This mesh object will be constructed using parameters in it
configuration file. This mesh object can be manipulated further, such as increasing its resolution through further 
splitting, adding additional data sources or altering is configuration parameters using functions listed in 
the :ref:`Methods - Mesh Construction` section of the documentation. The digital environment **`Mesh`** object can then be cast to 
a json object and saved to a file. 
::

    from meshiphi.mesh_generation.mesh_builder import MeshBuilder

    cg = MeshBuilder(config).build_environmental_mesh()
    
    mesh = cg.to_json()


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Simulating a Vessel in a Digital Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a digital environment **EnvironmentMesh** object has been created with `MeshiPhi <https://github.com/antarctica/MeshiPhi>`_, a vessels performance when travelling within it may be simulated. The **VesselPerformanceModeller**
object requires a digital environment in *json* format and vessel specific configuration parameters, also in *json* format. These may either
be loaded from a file, or created within the python terminal.

Loading mesh and vessel from *json* files:
::

    import json
    # Loading digital environment from file
    with open('mesh.json', 'r') as f:
        mesh = json.load(f)  

    # Loading vessel configuration parameters from file
    with open('vessel.json', 'r') as f:
        vessel = json.load(f) 

The **VesselPerformanceModeller** object can then be initialised. This can be used to simulate the performance of the vessel and encode this information
into the digital environment.
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
    with open('vessel_mesh.json') as f:
        json.dumps(vessel_mesh)

^^^^^^^^^^^^^^^^^^
Route Optimisation
^^^^^^^^^^^^^^^^^^
Now that the vessel dependent environmental mesh is defined, and represented in the `VesselPerformanceModeller` object, we can
construct routes, with parameters defined by the user in the configuration file. Waypoints are passed as an input 
file path, `waypoints.csv`, discussed more in the Inputs section of the manual pages.  The route construction is done 
in two stages: construction of the meshed dijkstra optimal routes, `.compute_routes()`; and, the smoothing of the 
dijkstra routes to further optimise the solution and reduce mesh dependencies, `.compute_smooth_routes()`. 
During `.compute_routes()` the paths are appended to the object as an entry `paths`, which are replaced by the 
smoothed paths after running `.compute_smooth_routes()`. An additional entry `waypoints` is generated to give the 
waypoints information used in route construction. For further info about the structure of the outputs of the 
paths please see the Outputs section of the manual.

::

    from polar_route.route_planner import RoutePlanner
    rp = RoutePlanner(vessel_mesh, route_config, waypoints)
    rp.compute_routes()
    rp.compute_smoothed_routes()
    info = rp.to_json()


^^^^^^^^^^^^^^^^^^^
Visualising Outputs
^^^^^^^^^^^^^^^^^^^

The **`Mesh`** object can be visualised using the **`GeoPlot`** package, also developed by BAS. This package is not included in the distribution 
of MeshiPhi, but can be installed using the following command:

:: 

    pip install bas_geoplot

**`GeoPlot`** can be used to visualise the **`Mesh`** object using the following code in an iPython notebook:

::
    
    from bas_geoplot.interactive import Map

    mesh = pd.DataFrame(mesh_json['cellboxes'])
    mp = Map(title="GRF Example")

    mp.Maps(mesh, 'MeshGrid', predefined='cx')
    mp.Maps(mesh, 'SIC', predefined='SIC')
    mp.Maps(mesh, 'Elevation', predefined='Elev', show=False)
    mp.Vectors(mesh,'Currents - Mesh', show=False, predefined='Currents')
    mp.Vectors(mesh, 'Winds', predefined='Winds', show=False)

    mp.show()
