********
Methods - Vessel Specifics
********

Overview
##############

All of the functionality that relates to the specific vehicle traversing our meshed environment model is contained within the vessel_performance module.
This module contains a `VesselPerformance` class that determines which cells in the mesh are inaccessible for that particular vessel and what its performance will be in each of the accessible cells.

.. figure:: ./Figures/Mesh_Fuel_Speed.jpg
    :align: center
    :width: 700

    Maps of the sea ice concentration (a), speed (b) and fuel consumption (c) across the Weddell Sea.
    The latter two quantities are derived from the former.




Vessel Performance
##############

.. automodule:: polar_route.vessel_performance

.. autoclass:: polar_route.vessel_performance.VesselPerformance
   :special-members: __init__
   :members: to_json
