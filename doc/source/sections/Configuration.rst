********
Examples
********

=================
Configuration
=================

In this section we will outline the standard structure for a configuration file used in all portions of the RoutePlanner software.

Outlined below is an example configuration file for running the RoutePlanner. Using this as a template we will go through each of the definitions in turn, descibing what each portion does with the subsections in the manual given by the main sections in the configuration file.
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
            "cellHeight":2.5,
            "splitDepth":4
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

The Configuration file is composed of three distict sections 'Mesh_info', 'Vessel', and 'Route_Info'.
Each of these contain configuration infromation for the various stages of the route planning pipeline.

^^^^^^^^^^^^^^^^^^
Mesh_info
^^^^^^^^^^^^^^^^^^
Mesh_info contains information required to build the discretised enviroment in which the route planner
operates. Information here dictates the region in which the mesh is constructed, the data contained within
the mesh and how the mesh is split to a non-uniform resolution. 

The 'Mesh_info' section of the configuration file contains three primary sections:

################
Region
################
The region section gives detailed information for the construction of the Discrete Mesh. The main defintions are the bounding region and temporal portion of interest (`longMin`, `latMin`, `longMax`, `latMax`, `startTime`, `endTime`), but also the starting shape of the spatial grid cell boxes (`cellWidth`, `cellHeight`) is defined before splitting is applied to a max split depth level (`splitDepth`). Further detail in each parameter is given below:

::

   "Region": {
      "latMin": -77.5,
      "latMax": -55,
      "longMin": -120,
      "longMax": -10,
      "startTime": "2017-02-01",
      "endTime": "2017-02-14",
      "cellWidth":5,
      "cellHeight":2.5,
      "splitDepth":4
   }
    
where the variables are as follows:

* **longMin**      *(float, degress)*      : Minimum Longitude Edge Mesh 
* **longMax**      *(float, degress)*      : Maximum Longitude Edge Mesh 
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
      named in variable 'loader'

##############
splitting_conditions
##############

The splitting_conditions section of the Configuration file determines how the CellBoxes that form the
CellGrid will be sub-divided based on the homogeneity of the data points contained within to form a mesh
of non-uniform spatial resolution.

::

   "splitting_conditions":[
      {"<value_name>":{
         "threshold":...,
         "upperBound": ...,
         "lowerBound":...
      }},
      {"<value_name>":{
         "threshold":...,
         "upperBound": ...,
         "lowerBound":...
      }}
   ]

where the variables are as follows:

* **<value_name>** *(string)* : The name of the value which the splitting condition will be applied to.
* **threshold** *(float)* : CellBoxes will be sub-divided as to seperate the data points contained within
   into CellBox which contain either above or below the given threshold.
* **upperBound** *(float)* : A percentage normalized between 0 and 1. A CellBox is deemed homogenous in 
   a given data type if greater than this percentage of data-points within are above the given threshold.
* **lowerBound** *(float)* : A percentage normalized between 0 and 1. A Cellbox is deemed homogenous in
   a given data type if less than this percentage of data-points within are below the given threshold.

.. note:: 
   splitting conditions are applied in the order they are specified in the configuaration file.


#############
value_output_types (optional)
#############

The value_output_types section is an optional section which may be added to Mesh_info. This dicates how data
of each value of a cellbox is returned when outputing the (CellBox) or (CellGrid). By default values associated
with a (CellBox) are calculated by taking the mean of all data-point of a given value within the CellBoxes bounds.
*value_output_type* allows this default to be changed to either the minimum or maximum of data-points.

::

   "value_output_types":{
      "<value_name>":< "MIN" | "MAX" | "MEAN" >
    }

* **<value_name>** *(string)* : The name of the value which the output type change will be applied to 

^^^^^^^^
Vessel
^^^^^^^^

TODO intro to vessel peformance


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

* **Speed** *(float)* : The maximum speed of the vessel in open water 
* **Unit** *(string)* : The units of measurement for the speed of the vessel
* **Beam** *(float)* : 
* **HullType** *(string)* :
* **ForceLimit** *(float)* :
* **MaxIceExtent** *(float)* : The maximum Sea Ice Concentration the vessel is able to travel though
   given as a percentage
* **MinDepth** *(float)* : The minimum depth of water the vessel is able to travel through


^^^^^^^^^^^^^^^
Route_Info
^^^^^^^^^^^^^^^

TODO intro to route info

::

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

where the variables are as follows:

* **Objective_Function** *(string)* :
* **Path_Variables** *(list<(string)>)* :
* **WayPoints** *(string)* :
* **Source_Waypoints** *(list<(string)>)*:
* **End_Waypoints** *(list<(string)>)* :
* **Vector Names** *(list<(string)>)* :
* **Zero_Currents** *(bool)* :
* **Variable_Speed** *(bool)* :
* **Time_Unit** *(string)* :
* **Early_Stopping_Criterion** *(bool)* :
* **Save_Dijkstra_Graphs** *(bool)* :
* **Smooth Path**
   * **Max Iteration Number** *(int)* :
   * **Minimum Difference** *(float)* :

========================
Python/iPython Notebooks
========================


Discrete Mesh
^^^^^^^^^^^^^^^^^^
In this section outline an example usecase ...


::

   from RoutePlanner.CellGrid import CellGrid
   mesh = CellGrid(config)

This requires some definition of the datasets to load 

Vehicles Specifics
^^^^^^^^^^^^^^^^^^
In this section outline an example usecase ...


Route Optimisation
^^^^^^^^^^^^^^^^^^
In this section outline an example usecase ...

.. raw:: html
   :file: example_routepath.html


Command Line Execution
##############
In the previous section we outlined how to run the codebase from within a Python file or in iPython notebooks. In this section we will outline how the code can be run directly from command line by passing a configuration file to a exicutable python file found in `./exec/` from the root directly. 

The command line execution

::

   python ./exec/routeplanner.py config.json 

In addition, within the exec folder there is the independet stages used within the route planner. These include:

* `meshing.py` - Discrete Meshing
* `vehiclespecs.py` - Vehicle Specifics
* `routes.py` - Route planning on pre-computed mesh and vehicle specifics.
