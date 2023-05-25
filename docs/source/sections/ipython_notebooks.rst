###############################
Python & iPython Notebooks
###############################

Route planning may also be done using a python terminal. This is case, the CLI is not required but the steps required for route planning 
follow the same format - create a digital enviroment; simulated a vessel against it; optimise a route plan through the digital enviroment.
 

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Creating the digital enviroment.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A configuration file is needed to initialise the **`Mesh`** object which forms the digital enviroment. This configuration file 
is of the same format used in the :ref:`create_mesh` CLI entry-point, and may either be loaded from a *json* file or constructed 
within the python terminal.

Loading configuration from *json* file:
::

    import json
    with open('config.json', 'r') as f:
        config = json.load(f)    


The digital enviroment **`Mesh`** object can then be initialised. This mesh object will be constructed using parameters in it 
configuration file. This mesh object can be manipulated further, such as increasing its resolution through further 
splitting, adding additional data sources or altering is configuration parameters using functions listed in 
the :ref:`Methods - Mesh Construction` section of the documentation.
::

   from polar_route.mesh import Mesh
   cg = Mesh(config)
   
The digital enviroment **`Mesh`** object can then be cast to a json object and saved to a file. This *mesh.json* file can then 
be used by the CLI entry-point :ref:`add_vehicle`, or the json object can be passed to the **`VesselPerformance`** object in a python 
terminal.
::

    mesh = cg.to_json()
    with open('mesh.json') as f:
        json.dumps(mesh)


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Simulating a Vessel in a Digital Enviroment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a digital enviroment **`Mesh`** object has been created, how a vessel interacts with it may be simulated. The **`VesselPerformance`** 
object requires a digital enviroment in *json* format and vessel specific configuration parameters, also in *json* format. These may either 
be loaded from a file, or created within the python terminal.

Loading mesh and vessel from *json* files:
::

    import json
    # Loading digital enviroment from file
    with open('mesh.json', 'r') as f:
        mesh = json.load(f)  

    # Loading vessel configuration parameters from file
    with open('vessel.json', 'r') as f:
        vessel = json.load(f) 

The **`VesselPerformance`** object can then be initialised. This will simulate the performance of the vessel and encodes this information 
into the digital enviroment.
::

   from polar_route.vessel_performance import VesselPerformance
   vp = VesselPerformance(mesh, vessel)

The **`VesselPerformance`** object can then be cast to a json object and saved to a file. This *vessel_mesh.json* file can then 
be used by the CLI entry-point :ref:`optimise_routes`, or the json object can be passed to the **`RoutePlanner`** object in a python 
terminal.
::

    vessel_mesh = vp.to_json()
    with open('vessel_mesh.json') as f:
        json.dumps(vessel_mesh)

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Route Optimisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now that the vessel dependent environmental mesh is defined, and represented in the `vessel_performance` object, we can 
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
    rp = RoutePlanner(vessel_mesh, route_config , waypoints)
    rp.compute_routes()
    rp.compute_smoothed_routes()
    info = rp.to_json()