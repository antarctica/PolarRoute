.. _route config:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Configuration - Route Planning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

   {
    "objective_function": "traveltime",
    "path_variables": [
      "fuel",
      "traveltime"
    ],
    "vector_names": ["uC","vC"]
   }

above is the minimal set of configuration parameters necessary to run the route planner where the variables are as follows:

* **objective_function** *(string)* : Defining the objective function to minimise for the route construction. This variable can either be defined as 'traveltime', 'battery' or 'fuel', depending on the vessel type .
* **path_variables** *(list<(string)>)* : A list of strings of the route variables to return in the output geojson.
* **vector_names** *(list<(string)>)* : The definition of the horizontal and vertical components of the vector acting on the ship in each cell. Data for these names must be within the cells of the input mesh.

::

   {
    "objective_function": "traveltime",
     "path_variables": [
       "fuel",
       "traveltime"
     ],
     "vector_names": ["uC","vC"],
     "time_unit": "days",
     "adjust_waypoints": true,
     "zero_currents": false,
     "fixed_speed": false,
     "waypoint_splitting": false,
     "smoothing_max_iterations": 2000,
     "smoothing_blocked_sic": 10.0,
     "smoothing_merge_separation": 1e-3,
     "smoothing_converged_separation": 1e-3
    }

above is the full set of configuration parameters that can be used for route planning, where the additional variables are as follows:

* **time_unit** *(string)* : The time unit to use in the route output . Currently only takes 'days', but will support 'hrs' in future releases.
* **adjust_waypoints** *(bool)* : Used to enable or disable the process that moves waypoints placed in inaccessible regions to the nearest accessible location.
* **zero_currents** *(bool)* : For development use only. Removes the effect of currents acting on the ship, setting all current vectors to zero.
* **fixed_speed** *(bool)*  : For development use only. Removes the effect of variable speed acting on the ship, ship speed set to max speed defined in the vessel config.
* **waypoint_splitting** *(bool)* : Used to enable or disable splitting around the input waypoints. If enabled, all cells containing waypoints will be split to the maximum split depth given in the mesh config.
* **smoothing_max_iterations** *(int)* : For development use only. Maximum number of iterations in the path smoothing. For most paths convergence is met 100x earlier than this value.
* **smoothing_blocked_sic** *(float)* : For development use only. The maximum difference in sea ice concentration allowed before a cell is blocked for the smoothing.
* **smoothing_merge_sep** *(float)* : For development use only. Minimum difference between two path smoothing iterations before a merge is triggered.
* **smoothing_converged_sep** *(float)* : For development use only. Minimum difference between two path smoothing iterations before convergence is triggered.

