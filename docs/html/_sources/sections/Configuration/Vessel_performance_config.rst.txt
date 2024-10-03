^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Configuration - Vessel Performance Modeller
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Vessel configuration file provides all the necessary information about the vessel that will execute
the routes such that performance parameters (e.g. speed or fuel consumption) can be calculated by the
`VesselPerformanceModeller` class. A file of this structure is also used as a command line argument for
the 'add_vehicle' entry point.

::

   {
     "vessel_type": "SDA",
     "max_speed": 26.5,
     "unit": "km/hr",
     "beam": 24.0,
     "hull_type": "slender",
     "force_limit": 96634.5,
     "max_ice_conc": 80,
     "min_depth": 10,
     "max_wave": 3,
     "excluded_zones": ["exclusion_zone"],
     "neighbour_splitting": true
   }

Above are a typical set of configuration parameters used for a vessel where the variables are as follows:

* **vessel_type** *(string)* : The specific vessel class to use for performance modelling.
* **max_speed** *(float)* : The maximum speed of the vessel in open water.
* **unit** *(string)* : The units of measurement for the speed of the vessel (currently only "km/hr" is supported).
* **beam** *(float)* : The beam (width) of the ship in metres.
* **hull_type** *(string)* : The hull profile of the ship (should be one of either "slender" or "blunt").
* **force_limit** *(float)* : The maximum allowed resistance force, specified in Newtons.
* **max_ice_conc** *(float)* : The maximum Sea Ice Concentration the vessel is able to travel through given as a percentage.
* **min_depth** *(float)* : The minimum depth of water the vessel is able to travel through in metres.
* **max_wave** *(float)* : The maximum significant wave height the vessel is able to travel through in metres.
* **excluded_zones** *(float)* : A list of of strings that name different boolean properties of a cell. Any cell with a value of True for any of the entered keys will be marked as unnavigable.
* **neighbour_splitting** *(bool)* : Used to enable or disable a feature that splits all accessible cells neighbouring inaccessible cells. This improves routing performance but can be disabled to speed up the vessel performance modelling.
