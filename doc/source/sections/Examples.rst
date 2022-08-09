********
Examples
********

========================
Python/iPython Notebooks
========================

Inside the folder `./Examples` is an example ipython notebook that can be used for the generation of an example route between two waypoints 'Halley' and 'Rothera'. In this example the processing procedure is split into the three main stages with an outline below showing this usecase. All the data required for processing this example can be found at https://drive.google.com/drive/folders/1EgtNrcXyVQMYPKqce6BNO2GbllAeaJJd?usp=sharing, please place all the files from the link into a folder at `./Examples/Data`.

Configuration Loading
^^^^^^^^^^^^^^^^^^
The configuration file is loaded into the codebase. As this file will be appened to throughout the construction of the paths we name the ouput file `info`. More information can be found in the configuration section of the manual for the construction of this input file
::
    import json
    with open('config.json', 'r') as f:
        info = json.load(f)    


Discrete Mesh
^^^^^^^^^^^^^^^^^^
The `info` object is passed to the polar_route software to construct a mesh. Once the mesh is constructed we output the json object, now with the appended mesh info, back out as `info`. More information of the output from this section can be found in the outputs section of the manual pages.
::

   from polar_route.mesh import Mesh
   cg = Mesh(info)
   info = cg.to_json()

Vehicles Specifics
^^^^^^^^^^^^^^^^^^
The `info` object now with mesh information is used in the vessel performance. The section of the code-base alters the mesh information to append new quantaties and alter the neighbourhood graph. This altered object is then output to `info`
::
   from polar_route.vessel_performance import VesselPerformance
   vp = VesselPerformance(info)
   info = vp.to_json()

Route Optimisation
^^^^^^^^^^^^^^^^^^
Now that the vessel depened environmental mesh is defined, and representing in the `info` object, we can construct routes, with parameters defined by the user in the configuration file. The route construction is done in two stages: construction of the meshed dijkstra optimal routes, `.compute_routes()`; and, the smoothing of the dijkstra routes to further optimise the solution and reduce mesh dependencies, `.compute_smooth_routes()`. During `.compute_routes()` the paths are appened to the `info` object as a entry `paths`, which are replaced by the smoothed paths after running `.compute_smooth_routes()`. An additional entry `waypoints` is generated to give the waypoints information used in route construction. For further info about the structure of the outputs of the paths please see the Outputs section of the manual.
::
    from polar_route.route_planner import RoutePlanner
    rp = RoutePlanner(info)
    rp.compute_routes()
    rp.compute_smoothed_routes()
    info = rp.to_json()




Plotting
^^^^^^^^^^^^^^^^^^
To plot the routes constructed and mesh information we use the python package `geoplot`. This is an external package that can be found at `https://github.com/antarctica/GeoPlot`. The code used to plot the above example would be:
::
    from geoplot.interactive import Map
    import pandas as pd

    config    = info_dict['config']
    mesh      = pd.DataFrame(info_dict['cellboxes'])
    paths     = info_dict['paths']
    waypoints = pd.DataFrame(info_dict['waypoints'])

    mp = Map(config,title='Example Test 1')
    mp.Maps(mesh,'SIC',predefined='SIC')
    mp.Maps(mesh,'Extreme Ice',predefined='Extreme Sea Ice Conc')
    mp.Maps(mesh,'Land Mask',predefined='Land Mask')
    mp.Paths(paths,'Routes',predefined='Route Traveltime Paths')
    mp.Points(waypoints,'Waypoints',names={"font_size":10.0})
    mp.MeshInfo(mesh,'Mesh Info',show=False)
    mp.show()

========================
Command Line Execution
========================

In the previous section we outlined how to run the codebase from within a Python file or in iPython notebooks. In this section we will outline how the code can be run directly from command line by passing a configuration file to a exicutable python file found in `./exec/` from the root directly. 

The command line execution

::

   python ./exec/polar_route.py config.json 

In addition, within the exec folder there is the independent stages used within the route planner. These include:

* `mesh.py` - Discrete Meshing
* `vessel_performance.py` - Vehicle Specifics applied to pre-computed mesh
* `route_planner.py` - Route planning on pre-computed mesh and vehicle specifics.