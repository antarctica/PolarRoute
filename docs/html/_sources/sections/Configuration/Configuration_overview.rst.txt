######################################
Configuration Overview
######################################

In this section we will outline the standard structure for a configuration file used in 
all portions of the PolarRoute software package.

Each stage of the route-planning process is configured by a separate configuration file. 
The configuration files are written in JSON, and are passed to each stage of the 
route-planning process as command-line arguments or through a Python script.

Example configuration files are provided in the `config` directory.

Descriptions of the configuration options for the Vessel Performance Modelling can 
be found in the :ref:`Configuration - Vessel Performance Modeller` section of the 
documentation.

Descriptions of the configuration options for Route Planning can be found in the 
:ref:`Configuration - Route Planning` section of the documentation.

.. toctree::
   :maxdepth: 1

   ./Vessel_performance_config
   ./Route_planning_config
