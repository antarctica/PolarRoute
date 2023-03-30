***************
AMSR Dataloader
***************

The AMSR (Advanced Microwave Scanning Radiometer) dataset is a publically
available that provides Sea Ice Concentration scans of the earth's oceans.
It is produced by researchers at the University of Bremen.

The AMSR dataloader is currently the only 'standalone' dataloader, in that it
is defined independantly of the abstract base class. This is due to issues 
with :code:`pandas` calculating mean values differently depending on how the 
data is loaded. This caused issues with the regression tests passing. 
This issue will be rectified soon by updating the regression tests.

Data can be downloaded from `here <https://seaice.uni-bremen.de/data-archive/>`_


.. automodule:: polar_route.dataloaders.scalar.amsr
   :special-members: __init__
   :members: