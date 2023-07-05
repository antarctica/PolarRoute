*************************
DUACS Currents Dataloader
*************************

DUACS  is a European operational multi-mission production system of altimeter data that provides (amongst other products)
global ocean current vectors. The system was developed by CNES/CLS and data is available from the copernicus marine data
service.

From their website:
   Altimeter satellite gridded Sea Level Anomalies (SLA) computed with respect to a twenty-year 1993, 2012 mean. The SLA
   is estimated by Optimal Interpolation, merging the L3 along-track measurement from the different altimeter missions
   available. Part of the processing is fitted to the Global Ocean. The product gives additional variables (i.e.
   Absolute Dynamic Topography and geostrophic currents).

Near real-time data can be downloaded from `here <https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046/description>`_.

Reanalysis data can be downloaded from `here. <https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description>`_


.. automodule:: polar_route.dataloaders.vector.duacs_current
   :special-members: __init__
   :members: