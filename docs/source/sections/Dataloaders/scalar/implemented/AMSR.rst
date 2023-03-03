***************
AMSR Dataloader
***************

The AMSR dataloader is currently the only 'standalone' dataloader, in that it
is defined independantly of the abstract base class. This is due to issues 
with :code:`pandas` calculating mean values differently depending on how the 
data is loaded. This caused issues with the regression tests passing. 
This issue will be rectified soon by updating the regression tests.

.. automodule:: polar_route.Dataloaders.Scalar.AMSR
   :special-members: __init__
   :members: