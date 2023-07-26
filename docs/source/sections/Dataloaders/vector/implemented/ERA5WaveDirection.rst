******************************
ERA5 Wave Direction Dataloader
******************************

ERA5 is a family of data products produced by the European Centre for Medium-Range Weather Forecasts (ECMWF).
It is the fifth generation ECMWF atmospheric reanalysis of the global climate covering the period from January 1950 to present.

From their website:

   ERA5 provides hourly estimates of a large number of atmospheric, 
   land and oceanic climate variables. The data cover the Earth on a 
   30km grid and resolve the atmosphere using 137 levels from the 
   surface up to a height of 80km. ERA5 includes information about 
   uncertainties for all variables at reduced spatial and temporal resolutions.

Instructions for how to download their data products are 
available `here <https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5>`_

This dataloader takes the mean wave direction variable, which gives the direction the waves are coming from as an angle
from north in degrees, and converts it to a unit vector with u and v components.


.. automodule:: polar_route.dataloaders.vector.era5_wave_direction
   :special-members: __init__
   :members: