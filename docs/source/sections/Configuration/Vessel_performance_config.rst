^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Configuration - Vessel Performance Modeller
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Vessel configuration file provides all the necessary information about the vessel that will execute
the routes such that performance parameters (e.g. speed or fuel consumption) can be calculated by the
`VesselPerformanceModeller` class. A file of this structure is also used as a command line argument for
the 'add_vehicle' entry point.

::

   {
     "VesselType": "SDA",
     "MaxSpeed": 26.5,
     "Unit": "km/hr",
     "Beam": 24.0,
     "HullType": "slender",
     "ForceLimit": 96634.5,
     "MaxIceConc": 80,
     "MinDepth": -10
   }

Above are a typical set of configuration parameters used for a vessel where the variables are as follows:

* **VesselType** *(string)* : The specific vessel class to use for performance modelling.
* **MaxSpeed** *(float)* : The maximum speed of the vessel in open water.
* **Unit** *(string)* : The units of measurement for the speed of the vessel (currently only "km/hr" is supported).
* **Beam** *(float)* : The beam (width) of the ship in metres.
* **HullType** *(string)* : The hull profile of the ship (should be one of either "slender" or "blunt").
* **ForceLimit** *(float)* : The maximum allowed resistance force, specified in Newtons.
* **MaxIceConc** *(float)* : The maximum Sea Ice Concentration the vessel is able to travel through given as a percentage.
* **MinDepth** *(float)* : The minimum depth of water the vessel is able to travel through in metres. Negative values correspond to a depth below sea level.