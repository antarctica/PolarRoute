********
Input - Configuration
********

In this section we will outline the standard structure for a configuration file used in all portions of the PolarRoute software package.

Outlined below is an example configuration file for running PolarRoute. Using this as a template we will go through each of the definitions in turn, describing what each portion does with the subsections in the manual given by the main sections in the configuration file.
::
   {
      "Mesh_info":{
         "Region": {
            "latMin": -77.5,
            "latMax": -55,
            "longMin": -120,
            "longMax": -10,
            "startTime": "2017-02-01",
            "endTime": "2017-02-14",
            "cellWidth":5,
            "cellHeight":2.5
         },
         "Data_sources": [
            {
            "loader":"load_bsose_depth",
            "params":{
               "file":"../../Data/BSOSE/bsose_i122_2013to2017_1day_SeaIceArea.nc",
               "data_name": "elevation"
            }
            }, 
            {
            "loader":"load_amsr",
            "params":{
               "file":"../../Data/AMSR/asi-AMSR-2017.nc"
            }
            },
            {
            "loader":"load_sose_currents",
            "params":{
               "file":"../../Data/SOSE_surface_velocity_6yearMean_2005-2010.nc"
            }
            },
            {
            "loader":"load_thickness",
            "params":{
            }
            },
            {
            "loader":"load_density",
            "params":{
            }
            }
         ],
         "splitting": {
            "split_depth":3,
            "minimum_datapoints":5,
            "splitting_conditions":[
               {"elevation":{
                  "threshold":-10,
                  "upperBound": 1,
                  "lowerBound":0
               }},
               {"SIC":{
                  "threshold":35,
                  "upperBound": 0.9,
                  "lowerBound":0.1
               }}
            ],
            "value_fill_types":{
               "SIC": "parent",
               "uC": "parent",
               "vC": "parent"
            }
         },
         "value_output_types":{
            "elevation":"MAX"
         }
      },
      "Vessel": {
         "Speed": 26.5,
         "Unit": "km/hr",
         "Beam": 24.0,
         "HullType": "slender",
         "ForceLimit": 96634.5,
         "MaxIceExtent": 80,
         "MinDepth": -10
      },
      "Route_Info": {
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
         "Time_Unit": "days",
         "Early_Stopping_Criterion": true,
         "Save_Dijkstra_Graphs": false,
         "Smooth Path":{
            "Max Iteration Number":1000,
            "Minimum Difference": 1e-3
         }
      }
   }

The Configuration file is composed of three distinct sections 'Mesh_info', 'Vessel', and 'Route_Info'.
Each of these contain configuration information for the various stages of the route planning pipeline.

^^^^^^^^^^^^^^^^^^
Mesh_info
^^^^^^^^^^^^^^^^^^
Mesh_info contains information required to build the discretised environment in which the route planner
operates. Information here dictates the region in which the mesh is constructed, the data contained within
the mesh and how the mesh is split to a non-uniform resolution. 

The 'Mesh_info' section of the configuration file contains three primary sections:

################
Region
################
The region section gives detailed information for the construction of the Discrete Mesh. The main definitions are the bounding region and temporal portion of interest (`longMin`, `latMin`, `longMax`, `latMax`, `startTime`, `endTime`), but also the starting shape of the spatial grid cell boxes (`cellWidth`, `cellHeight`) is defined before splitting is applied . Further detail on each parameter is given below:

::

   "Region": {
      "latMin": -77.5,
      "latMax": -55,
      "longMin": -120,
      "longMax": -10,
      "startTime": "2017-02-01",
      "endTime": "2017-02-14",
      "cellWidth":5,
      "cellHeight":2.5
   }
    
where the variables are as follows:

* **longMin**      *(float, degrees)*      : Minimum Longitude Edge Mesh
* **longMax**      *(float, degrees)*      : Maximum Longitude Edge Mesh
* **latMin**       *(float, degrees)*      : Minimum Latitude Edge Mesh  
* **latMax**       *(float, degrees)*      : Maximum Latitude Edge Mesh  
* **startTime**    *(string, 'YYYY-mm-dd')*   : Start Datetime of Time averaging 
* **endTime**      *(string, 'YYYY-mm-dd')*   : End Datetime of Time averaging   
* **cellWidth**    *(float, degrees)*      : Initial Cell Box Width prior to splitting 
* **cellHeight**   *(float, degrees)*      : Initial Cell Box Height prior to splitting 

#################
Data_sources
#################

The 'Data_sources' section of the configuration file dictates which information will be added to the
mesh when constructed. Each item in the list of data sources represents a single data set to be added
to the mesh.

::

   "Data_sources": [
      {
         "loader":"load_bsose_depth",
         "params":{
            "file":"../../Data/BSOSE/bsose_i122_2013to2017_1day_SeaIceArea.nc",
            "data_name": "elevation"
         }
      }, 
      {
         "loader":"load_amsr",
         "params":{
            "file":"../../Data/AMSR/asi-AMSR-2017.nc"
         }
      },
      {
         "loader":"load_sose_currents",
         "params":{
            "file":"../../Data/SOSE_surface_velocity_6yearMean_2005-2010.nc"
         }
      },
      {
         "loader":"load_thickness",
         "params":{
         }
      },
      {
         "loader":"load_density",
         "params":{
         }
      }
   ]
   

where the variables are as follows:


* **loader** *(string)* : The function name of the data loader to be used to add this data source to the mesh
      see section 'Multi Data Input' for further information about data loader functions.
* **params** *(dict)* : A dictionary containing optional parameters which may be required by data loader function
      named in variable 'loader'.

##############
splitting
##############

The splitting section of the Configuration file determines how the CellBoxes that form the
Mesh will be sub-divided based on the homogeneity of the data points contained within to form a mesh
of non-uniform spatial resolution.

Non-uniform mesh refinement is done by selectively sub-dividing cells. Cell sub-division is performed 
whenever a cell (of any size) is determined to be inhomogeneous with respect to a specific characteristic 
of interest such as SIC or ocean depth. For example, considering SIC, we define a range, from a lower bound 
*lb* to an upper bound *ub*, and a threshold, *t*. Then, a cell is considered inhomogeneous if between *lb* and *ub* 
of the ice measurements in that cell are at *t%* or higher.  If the proportion of ice in the cell above the 
*t%* concentration is below *lb%*, we consider the cell to be homogeneous open water: such a cell can be navigated 
through so does not require splitting based on this homogeneity condition (though may still be split based on others).
 At the other end of the range, if the proportion is greater than *ub%*, then the cell is considered 
homogeneous ice: such a cell cannot be navigated through all will not be split on this or any subsequent splitting conditions. 
If the proportion is between these bounds, then the cell is inhomogeneous and must be split so that the homogeneous sub-cells
 can be found.

::

   "splitting": {
      "split_depth":3,
      "minimum_datapoints":5,
      "splitting_conditions":[
        {"elevation":{
            "threshold":-10,
            "upperBound": 1,
            "lowerBound":0
        }},
        {"SIC":{
            "threshold":35,
            "upperBound": 0.9,
            "lowerBound":0.1
        }}
      ],
      "value_fill_types":{
        "SIC": "parent",
        "uC": "parent",
        "vC": "parent"
      }
    }

where the variables are as follows:

* **split_depth** *(float)* : The number of times the mesh will sub-divided each initial cellbox
* **minimum_datapoints** *(float)* : The minimum number of datapoints a cellbox must contain for each value type to be able to split
* **splitting_conditions** *(list)* : The conditions which determine if a cellbox should be split.
   * **<value_name>** *(string)* : The name of the value which the splitting condition will be applied to.
   * **threshold** *(float)* : The threshold above or below which CellBoxes will be sub-divided to separate the datapoints into homogeneous cells.
   * **upperBound** *(float)* : A percentage normalised between 0 and 1. A CellBox is deemed homogeneous in a given data type if greater than this percentage of data points are above the given threshold.
   * **lowerBound** *(float)* : A percentage normalised between 0 and 1. A Cellbox is deemed homogeneous in a given data type if less than this percentage of data points are below the given threshold.
* **value_fill_types** *(dict)* : Determines the actions taken if a cellbox is generated with no data for a given value type
   * **<value_name>** *(string)* : The name of the value which the fill type will be applied to.
   * **<fill_type>** *(string)* : <parent | zero | nan>
.. note:: 
   splitting conditions are applied in the order they are specified in the configuration file.


#############
value_output_types (optional)
#############

The value_output_types section is an optional section which may be added to Mesh_info. This dictates how data
of each value of a cellbox is returned when outputting the (CellBox) or (Mesh). By default values associated
with a (CellBox) are calculated by taking the mean of all data points of a given value within the CellBoxes bounds.
*value_output_type* allows this default to be changed to either the minimum or maximum of data-points.

::

   "value_output_types":{
      "<value_name>":< "MIN" | "MAX" | "MEAN" >
    }

* **<value_name>** *(string)* : The name of the value which the output type change will be applied to 

^^^^^^^^
Vessel
^^^^^^^^

The Vessel section of the configuration file provides all the necessary information about the vessel that will execute
the routes such that performance parameters (e.g. speed or fuel consumption) can be calculated by the `VesselPerformance`
class for this vessel.


::

   "Vessel": {
         "Speed": 26.5,
         "Unit": "km/hr",
         "Beam": 24.0,
         "HullType": "slender",
         "ForceLimit": 96634.5,
         "MaxIceExtent": 80,
         "MinDepth": -10
      },

where the variables are as follows:

* **Speed** *(float)* : The maximum speed of the vessel in open water.
* **Unit** *(string)* : The units of measurement for the speed of the vessel (currently only "km/hr" is supported).
* **Beam** *(float)* : The beam (width) of the ship in metres.
* **HullType** *(string)* : The hull profile of the ship (should be one of either "slender" or "blunt").
* **ForceLimit** *(float)* : The maximum allowed resistance force, specified in Newtons.
* **MaxIceExtent** *(float)* : The maximum Sea Ice Concentration the vessel is able to travel through given as a percentage.
* **MinDepth** *(float)* : The minimum depth of water the vessel is able to travel through in metres. Negative values correspond to a depth below sea level.


^^^^^^^^^^^^^^^
Route_Info
^^^^^^^^^^^^^^^

TODO intro to route info

::

   "Route_Info": {
         "objective_function": "traveltime",
         "path_variables": [
            "fuel",
            "traveltime"
         ],
         "waypoints_path": "./WayPoints_org.csv",
         "source_waypoints": ["LongPathStart"],
         "end_waypoints": [],
         "vector_names": ["uC","vC"],
         "zero_currents": false,
         "variable_speed": true,
         "time_unit": "days",
         "early_stopping_criterion": true,
         "save_dijkstra_graphs": false,
         "smooth_path":{
            "max_iteration_number":1000,
            "minimum_difference": 1e-3
         }
      }

where the variables are as follows:

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