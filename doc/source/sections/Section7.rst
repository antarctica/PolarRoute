********
Examples
********

Configuration
##############

In this section we will outline the standard structure for a configuration file used in all portions of the RoutePlanner software.

Outlined below is an example configuration file for running the RoutePlanner. Using this as a template we will go through each of the definitions in turn, descibing what each portion does with the subsections in the manual given by the main sections in the configuration file.
::
   {
   "Region": {
      "latMin": 50,
      "latMax": 89,
      "longMin": -180,
      "longMax": 180,
      "startTime": "2017-01-01",
      "endTime": "2017-02-01",
      "cellWidth":5,
      "cellHeight":2.5,
      "splitDepth":0
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
      "Mesh_Filename": "./cellgrid_dataframe_speed.csv",
      "Paths_Filename": "./paths_traveltime.json",
      "Smoothpaths_Filename":"./paths_traveltime_smooth.json",
      "Objective_Function":"traveltime",
      "Path_Variables": ["Fuel","traveltime"],
      "WayPoints": "./Waypoints.csv",
      "Source_Waypoints": ["Harwich"],
      "End_Waypoints": [],
      "Zero_Currents": true,
      "Variable_Speed": true,
      "Time_Unit":"days",
      "Early_Stopping_Criterion": true,
      "Save_Dijkstra_Graphs": false
   },


Region
^^^^^^^^^^^^^^^^^^
The region section gives detailed information for the construction of the Discrete Mesh. The main defintions are the bounding region and temporal portion of interest (`longMin`, `latMin`, `longMax`, `latMax`, `startTime`, `endTime`), but also the starting shape of the spatial grid cell boxes (`cellWidth`, `cellHeight`) is defined before splitting is applied to a max split depth level (`splitDepth`). Further detail in each parameter is given below:

::

   "longMin" = Minimum Longitude Edge Mesh (float, degress)
   "longMax" = Maximum Longitude Edge Mesh (float, degress) 
   "latMin" = Minimum Latitude Edge Mesh  (float, degrees)
   "latMax" = Maximum Latitude Edge Mesh  (float, degrees)
   "startTime" = Start Datetime of Time averaing (str, 'YYYY-mm-dd')
   "endTime"  = End Datetime of Time averaing   (str, 'YYYY-mm-dd')
   "cellWidth" = Initial Cell Box Width prior to splitting (float, degrees)
   "cellHeight" = Initial Cell Box Height prior to splitting (float, degrees)


Data Sources
^^^^^^^^^^^^^^^^^^
The data soruces describes the different intput datasets and the required homogenous splitting conditions to apply in a hieracical form. The splitting is applied in order of the data sources described. The standard structure of the Data sources takes the form of:

::
   
   "Data_sources": [
      {
         "path": "../../Data/depth_map.nc",
         "latName": "lat",
         "longName": "long",
         "values": [
         {
            "sourceName": "depth",
            "destinationName": "depth",
            "splittingCondition": {
               "threshold": -10,
               "lowerBound": 0.000,
               "upperBound": 1.0
            }
         }
         ]
      },
      ...
   ]

where the variables are as follows:

:: 

   "path"     = Path to dataset to load (str)
   "latName"  = Dataset varible name for the latitude information (str) 
   "longName" = Dataset variable name for the longitude information (str)
   "values"   = Splitting conditions to apply to the dataset on a specific defined variable (list)
   "sourceName"      = Dataset varible name within the origional dataset (str) 
   "destinationName" = Dataset varible output used within Mesh construction & later vehicles specs (str) 
   "splittingContion" = Dictionary composed of "threshold" (float), "lowerBound" (float) and "upperBound"(float). The defintion for setting the splitting condition can be found in the earlier Section on Discrete Meshing.


Python/iPython Notebooks
##############


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
