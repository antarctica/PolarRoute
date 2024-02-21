.. _route config:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Configuration - Route Planning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

   {
     "Objective_Function": "traveltime",
     "Path_Variables": [
        "fuel",
        "traveltime"
     ],
     "WayPoints": "./WayPoints_org.csv",
     "Source_Waypoints": ["LongPathStart"],
     "End_Waypoints": [],
     "Vector Names": ["uC","vC"],
     "Zero_Currents": false,
     "Variable_Speed": true,
     "Time_unit": "days",
     "Early_Stopping_Criterion": true,
     "Save_Dijkstra_Graphs": false,
     "Smooth Path":{
        "Max Iteration Number":1000,
        "Minimum Difference": 1e-3
     }
   }

above is a typical set of configuration parameters used for route planning where the variables are as follows:

* **objective_function** *(string)* : Defining the objective function to minimise for the construction of the mesh based Dijkstra routes. This variable can either be defined as 'traveltime' or 'fuel' .
* **path_variables** *(list<(string)>)* : A list of strings of the route variables to return in the output geojson. 
* **waypoints_path** *(string)* : A filepath to a CSV containing the user defined waypoints with columns including: 'Name','Lat',"Long"
* **source_waypoints** *(list<(string)>)*: The source waypoints to define the routes from. The names in this list must be the same as names within the `waypoints_path` file. If left blank then routes will be determined from all waypoints.
* **end_waypoints** *(list<(string)>)* : The end waypoints to define the routes to. The names in this list must be the same as names within the `waypoints_path` file. If left blank then routes will be determined to all waypoints.
* **vector_names** *(list<(string)>)* : The definition of the horizontal and vertical components of the vector acting on the ship within each CellBox. These names must be within the 'cellboxes'.
* **zero_currents** *(bool)* : For development use only. Removes the effect of currents acting on the ship, setting all current vectors to zero.
* **Variable_Speed** *(bool)*  : For development use only. Removes the effect of variable speed acting on the ship, ship speed set to max speed defined by 'Vessel':{'Speed':...}.
* **time_unit** *(string)* : The time unit to output the route path information. Currently only takes 'days', but will support 'hrs' in future releases.
* **early_stopping_criterion** *(bool)* : For development use only. Dijkstra early stopping criterion. For development use only if the full objective_function from each starting waypoint is required. Should be used in conjunction with `save_dijkstra_graphs`.
* **save_dijkstra_graphs** *(bool)* : For development use only. Saves the full dijkstra graph representing the objective_function value across all mesh cells.
* **Smooth Path**
   * **max_iteration_number** *(int)* : For development use only. Maximum number of iterations in the path smoothing. For most paths convergence is met 100x earlier than this value. 
   * **minimum_difference** *(float)* : For development use only. Minimum difference between two path smoothing iterations before convergence is triggered

